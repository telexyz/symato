import numpy as np
import random, types, torch
from torch.nn import functional as F

########################################################################################################
# Step 0: Define the model in inference mode
########################################################################################################

class RWKV_RNN(torch.jit.ScriptModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode

        # Load tham số từ file vào vào bộ nhớ và biến đổi cho phù hợp
        w = torch.load(args.MODEL_NAME, map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # e^negative = decay it's actually `e^{-e^x}``
            else: w[k] = w[k].float() # convert to f32 type

        # Gán tham số vào biến số mô hình
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Tên tham số (string)                     Biến số (namespace)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # "emb.weight"                          => self.w.weight
        # "ln_out.weight"                       => self.w.ln_out.weight
        # "head.weight"                         => self.w.head.weight
        # "blocks.0.ln0.weight"                 => self.w.blocks[0].ln0.weight
        # "blocks.0.ln1.weight"                 => self.w.blocks[0].ln1.weight
        # "blocks.0.ln2.weight"                 => self.w.blocks[0].ln2.weight
        # "blocks.0.att.time_first"             => self.w.blocks[0].att.time_first
        # "blocks.0.att.time_decay"             ...
        # "blocks.0.att.time_mix_k"
        # "blocks.0.att.time_mix_v"
        # "blocks.0.att.time_mix_r"
        # "blocks.0.att.key.weight"
        # "blocks.0.att.value.weight"
        # "blocks.0.att.receptance.weight"
        # "blocks.0.att.output.weight"
        # "blocks.0.ffn.time_mix_k"
        # "blocks.0.ffn.time_mix_r"
        # "blocks.0.ffn.key.weight"
        # "blocks.0.ffn.value.weight"
        # "blocks.0.ffn.receptance.weight"
        #                                       ... tiếp tục cho tới block thứ n ...
        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for k in w.keys():
            parts = k.split('.') # Ví dụ k = "blocks.0.att.value.weight" => parts = ['block','0','ln0','weight']
            last = parts.pop() # => last = "weight"; parts = ['block','0','ln0']
            here = self.w
            for p in parts: # từng bước mở rộng namespace
                if p.isdigit(): # tầng thứ p
                    p = int(p) # dùng [] vì here (w.blocks) là dict object {}
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else: # dùng hasattr, setattr, getattr vì here là types.SimpleNamespace()
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k]) # gán giá trị vào namespace cuối cùng => self.w.blocks[0].ln0.weight = w[k]


    # state[] để lưu trạng thái của rnn, mỗi tầng i ghi lại 5 trạng thái: 
    # i+0 = ffn_xx : token của bước channel-mixing trước 
    # i+1 = att_xx : token của bước time-mixing trước
    # i+2 = att_aa : exp moving avg của kv 
    # i+3 = att_bb : exp moving avg của k
    # i+4 = att_pp : use pp to stove exponent of aa and bb
    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ''' channel-mixing giống FFN của transformer nhưng có thêm nhiều cải tiến:
        * Token-shift trộn vector hiện tại với vector đầu vào trước đó theo 1 tỉ lệ nhất định, mục đích là để
          mô hình mang thông tin từ quá khứ tới hiện tại tốt hơn, giúp việc suy diễn tốt hơn
        * Dùng relu square (giống primer)
        * Nhân với hàm sigmoid(r) giúp mô hình ổn định hơn.
        '''
        ffn_xx = 5*i+0 # feed-forward or channel mixing
        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + state[ffn_xx] * (1 - time_mix_k)
        xr = x * time_mix_r + state[ffn_xx] * (1 - time_mix_r)
        state[ffn_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr) # receptance factor thuộc 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        ''' Time-mixing hay còn gọi là linear-attention. Tuy nhiên công thức linear-attention này có thể được viết lại
        dưới dạng hồi quy nên sử dụng dạng công thức hồi quy để tính toán tiết kiệm hơn. Do hồi quy chỉ cần trạng thái
        hệ thống ở t-1 để tính ra trạng trái hệ thống ở t, không cần phải tính lại cho toàn bộ chuỗi đầu vào.
        '''
        att_xx = 5*i+1 # attention or time mixing
        # token-shift with diff mixing factors for k, v and r
        xk = x * time_mix_k + state[att_xx] * (1 - time_mix_k)
        xv = x * time_mix_v + state[att_xx] * (1 - time_mix_v)
        xr = x * time_mix_r + state[att_xx] * (1 - time_mix_r)
        state[att_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        # Công thức hồi quy của rnn mode, xem https://github.com/telexyz/symato/blob/main/docs/rwkv-illustrated.md
        aa = state[5*i+2] # exponential moving average of kv
        bb = state[5*i+3] # exponential moving average of k
        pp = state[5*i+4] # idea: use pp to store exponent of a and b

        ww = time_first + k # u + k_i
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b

        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq

        return ow @ (r * wkv)

    def forward(self, token_id, state, preprocess_only=False):
        with torch.no_grad():
            # 0/ Khởi tạo trạng thái hệ thống nếu chưa được khởi tạo
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # state[att_pp] = âm vô cực

            # 1/ Lấy vector nhúng của token_id
            x = self.w.emb.weight[token_id]
            # Và áp dụng layer-norm-0 ở tầng đầu tiên để small-init-emb trick hoạt động
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            
            # 2/ Với mỗi tầng áp dụng:
            for i in range(self.args.n_layer):
                # 2.1/ time-mixing
                att = self.w.blocks[i].att # trọng số của khối time-mixing
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)

                # 2.2/ channel-mixing
                ffn = self.w.blocks[i].ffn # trọng số của khối channel-mixing
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

            if preprocess_only: return state
            # 3/ Cuối cùng áp dụng bộ phân lớp cho ra next token probabilities
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

##########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
##########################################################################################################
from symato import Symato

TOKENIZER = Symato()
args = types.SimpleNamespace()
args.MODEL_NAME = "model/symato-2816-vlc-23m.pth"
args.n_layer = 6
args.n_embd = 512

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################
NUM_TRIALS = 10
LENGTH_PER_TRIAL = 250
TEMPERATURE = 0.9
top_p = 0.8

def sample_logits(out, temperature=1.0, top_p_usual=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    probs[256:274] = 0 # ko ngẫu nhiên sinh marktone
    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs) # [1,2,3] => [1,3,6]
    idx = np.argmax(cumulative_probs > top_p_usual) # vì là mảng True, False nên trả về idx của True đầu tiên
    cutoff = float(sorted_probs[idx]) # cutoff là tổng những prob lớn nhất đầu tiên vượt qua top_p_usual
    probs[probs < cutoff] = 0 # bỏ đi những prob < cutoff
    if temperature != 1.0: probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs) # chuẩn hóa lại probs sao cho tổng = 1
    return np.random.choice(a=len(probs), p=probs) # lấy mẫu

########################################################################################################
# Step 3: generate more tokens given the prompt
########################################################################################################

print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

def finetune_context(context):
    # tids = []
    # for syllable in context.split(): tids += list(TOKENIZER.syllable_to_tids(syllable))
    context = context.replace("\\", "|")
    context = "".join([f"{token}" for token in context.split()]) + " "
    out, init_state, finetune_tids = None, None, []
    # Nhồi context (a.k.a prompt) vào mô hình
    tids = TOKENIZER.encode(context)
    for i, token_id in enumerate(tids):
        finetune_tids.append(token_id)
        out, init_state = model.forward(token_id, init_state)
        # Tự động bỏ dấu thanh nếu chưa có
        if TOKENIZER.is_sym(token_id) and not TOKENIZER.is_marktone(tids[i + 1]):
            # TODO: tìm n-best marktone sequence
            best_marktone_id = 256 + out.numpy()[256:274].argmax()
            finetune_tids.append(best_marktone_id)
            out, init_state = model.forward(best_marktone_id, init_state)
    finetune_context = TOKENIZER.tids_to_utf8(finetune_tids)
    return out, init_state, finetune_context

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n\n- - - [ LẦN THỬ {TRIAL}: nhập tiếng Việt không dấu. Ví dụ "luat doanh nghiep" ] - - - - - - -\n')
    context = input("Hỏi: ")
    out, state, new_ctx = finetune_context(context)
    print(f'Đáp: {new_ctx}', end="")
    cap_id, prev_tid = None, 0

    for i in range(LENGTH_PER_TRIAL): # sinh thêm LENGTH_PER_TRIAL tokens nữa từ prompt đầu vào
        token_id = sample_logits(out, TEMPERATURE, top_p) # lấy mẫu ngẫu nhiên token tiếp theo
        if TOKENIZER.is_capitalized(token_id):
            cap_id = token_id
        else:
            token = TOKENIZER.decode(token_id)
            if TOKENIZER.is_sym(token_id):
                out, state = model.forward(token_id, state)
                best_marktone_id = 256 + out.numpy()[256:274].argmax()
                token = TOKENIZER.to_utf8(token_id, best_marktone_id)
                token_id = best_marktone_id
                if cap_id is not None:
                    if TOKENIZER.is_sym_capitalized(cap_id): token = token.upper()
                    else: token = token[0].upper() + token[1:]
                    cap_id = None
                if TOKENIZER.is_marktone(prev_tid): token = " " + token # thêm space giữa 2 âm tiết liền nhau

            if token_id == 32 and prev_tid == 32: continue # gộp nhiều spaces làm 1
            print(token, end="", flush=True)
        prev_tid = token_id
        out, state = model.forward(token_id, state)
print()
