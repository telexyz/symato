########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def __nop(ob): return ob

MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

FLOAT_MODE = os.environ["RWKV_FLOAT_MODE"] # "fp32", "tf32", "fp16", "bf16"
FLOAT_MODE_fp32 = ("32" in FLOAT_MODE)
FLOAT_MODE_fp16 = (FLOAT_MODE == "fp16")
FLOAT_MODE_bf16 = (FLOAT_MODE == "bf16")

T_MAX = int(os.environ["RWKV_T_MAX"]) # == args.ctx_len, TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load
wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["wkv_op.cpp", "wkv_cuda.cu"],
    verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", f"-DTmax={T_MAX}"])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= T_MAX
        if C > 32: assert (B * C) % 32 == 0, "Nếu C > 32 thì B * C phải chia hết cho 32"

        if FLOAT_MODE_fp32:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else: # biến thành f32
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y) # giá trị được lưu vào y

        if   FLOAT_MODE_fp32: return y
        elif FLOAT_MODE_fp16: return y.half()
        elif FLOAT_MODE_bf16: return y.bfloat16()


    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        assert T <= T_MAX
        if C > 32: assert (B * C) % 32 == 0

        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device=gy.device).contiguous()
        gu = torch.zeros((B, C), device=gy.device).contiguous()
        gk = torch.zeros((B, T, C), device=gy.device).contiguous()
        gv = torch.zeros((B, T, C), device=gy.device).contiguous()

        if not FLOAT_MODE_fp32: gy_ = gy.float() # biến đổi thành f32
        wkv_cuda.backward(B, T, C, w, u, k, v, gy_.contiguous(), gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        if   FLOAT_MODE_fp32: return (None, None, None, gw, gu, gk, gv)
        elif FLOAT_MODE_fp16: return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif FLOAT_MODE_bf16: return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd
        attn_sz = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0

            # fancy time_decay
            decay_speed = [-5 + 8*(h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(attn_sz) ]
            self.time_decay = nn.Parameter(torch.tensor(decay_speed))
            # time_decay => -5.00, -3.16, -1.89, -0.78,  0.23,  1.20,  2.11,  3.00

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            # zigzag     =>  0.00,  0.50, -0.50,  0.00,  0.50, -0.50,  0.00,  0.50
            # time_first => -1.20, -0.70, -1.70, -1.20, -0.70, -1.70, -1.20, -0.70

            # fancy time_mix
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            # time_mix_k => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87
            # time_mix_v => 0.01, 0.14, 0.27, 0.40, 0.52, 0.65, 0.77, 0.89
            # time_mix_r => 0.00, 0.36, 0.51, 0.62, 0.71, 0.79, 0.87, 0.93

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1)) # padding zero trước embd vector đầu tiên trong batch
        self.key = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, args.n_embd, bias=False)

    @MyFunction
    def jit_func(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x) # prev_x
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r) # sigmoid(receptance)
        # 
        return sr, k, v

    # Tại sao không jit toàn bộ forward mà chỉ jit phần tính sr, k, v
    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            # time_mix_k => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87
            # time_mix_r => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87

        hidden_sz = 4 * args.n_embd
        self.key = nn.Linear(args.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, args.n_embd, bias=False)


    @MyFunction # jit toàn bộ forward của channel-mix
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:  self.att = RWKV_TimeMix(args, layer_id)
        # Đôi khi thay att của tầng đầu (layer_id == 0) bằng ffn lại cho kết quả tốt hơn

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            # Thêm tiny attention để liên kết các tokens cực xa nhau tốt hơn!
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))


    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if hasattr(self, "ffnPre"): x = x + self.ffnPre(self.ln1(x))
        else: x = x + self.att(self.ln1(x))
        # Đôi khi thay att của tầng đầu (layer_id == 0) bằng ffn lại cho kết quả tốt hơn
        x = x + self.ffn(self.ln2(x))

        if hasattr(self, "tiny_ln"): # áp dụng tiny_attention nếu có
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()

            for n, p in self.named_parameters():
                if "time_mix" in n:
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif "time_decay" in n:
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)

            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))

            param_dict = {n: p for n, p in self.named_parameters()}
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
            ]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)


    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False


    def forward(self, idx):
        args = self.args
        GRAD_CP = (args.grad_cp == 1) # gradient checkpoint?
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        if args.tiny_att_dim > 0: # nếu có tiny_att thì sử dụng x_emb
            x_emb = x
            for block in self.blocks:
                if GRAD_CP: x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:       x = block(x, x_emb)
        else:
            for block in self.blocks:
                if GRAD_CP: x = deepspeed.checkpointing.checkpoint(block, x)
                else:       x = block(x)

        x = self.ln_out(x)

        if args.head_qk > 0: # apply head_qk trick nếu có
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            if   FLOAT_MODE_fp32: c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif FLOAT_MODE_fp16: c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif FLOAT_MODE_bf16: c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()
            return self.head(x) + c
        else:
            return self.head(x)


    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)


    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all


    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n:
                m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q."]:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if FLOAT_MODE_fp16:
                m[n] = m[n].half()
            elif FLOAT_MODE_bf16:
                m[n] = m[n].bfloat16()

        gc.collect()
        torch.cuda.empty_cache()
        return m
