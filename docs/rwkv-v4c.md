# Cải tiến

## Full self-attn

```py
@MyFunction
def self_attn(self, q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.att_mask == 0, float('-inf'))
    att = F.softmax(att, dim = -1)
    x = att @ v
    return x

@MyFunction
def token_shift_and_projection(self, x):
    # Mix x with the previous timestep to produce xk, xv, xr
    xx = self.time_shift(x)
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
    xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
    xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)

    # Use xk, xv, xr to produce k, v, r
    k = self.key(xk)
    v = self.value(xv)
    r = self.receptance(xr)
    sr = torch.sigmoid(r)
    
    qq = self.qq(xqq)
    kk = self.kk(xkk)
    vv = self.vv(xvv)

    return sr, k, v, qq, kk, vv

def forward(self, x):
    B, T, C = x.size()  # x = (Batch,Time,Channel)
    sr, k, v, qq, kk, vv = self.token_shift_and_projection(x)
    rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
    rwkv = self.output(rwkv) + self.oo(self.self_attn(qq, kk, vv))
    return rwkv # là phép cộng của rwkv time-mixing (thông thường) với self-attn (thông thường)
```

## Thay channel-mixing bằng MishGLU
```py
class MishGLU(MyModule):
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k) # time_mix_k, time_mix_r
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r) # là param vectors
        a, b = self.aa(xa), self.bb(xb) # aa, bb, value là linear layers
        return self.value(a * F.mish(b))
```
