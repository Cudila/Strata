#!/usr/bin/env python3
# tools/nemotron/attn_naive.py - Nemotron-H's attention block written out in float64 (stage 3 specification),
# checked against transformers' block on the same input, and dumped as test vectors for src/kernels/attn_parity.cpp.
#
#   xn = rms_norm(x) * norm.weight
#   q = Wq xn (32 x 128),  k = Wk xn, v = Wv xn (2 x 128)          no RoPE, no q/k norm, no bias
#   o[t, h] = sum_{j <= t} softmax_j(q[t,h] . k[j, h // 16] / sqrt(128)) v[j, h // 16]
#   out = Wo o ;  block output = x + out
#
# Two references are dumped: `o` with exact K/V, and `o_bf16kv` with K and V rounded to BF16 first - the engine's
# K/V cache - so the kernel is checked tightly against the second and the cost of a BF16 cache is the gap between them.
#
# Usage: python attn_naive.py <ref.npz from ref_layers.py, holding the block> <BF16 checkpoint dir> <layer> --dump DIR
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_layers import Ckpt, make_config  # noqa: E402


def attention(q, k, v, n_kv):
    T, H, Dh = q.shape
    group = H // n_kv
    scores = torch.einsum("thd,jhd->htj", q, k.repeat_interleave(group, dim=1)) / Dh ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, -1)
    return torch.einsum("htj,jhd->thd", p, v.repeat_interleave(group, dim=1))


def main():
    ref = np.load(sys.argv[1])
    ck = Ckpt(sys.argv[2])
    layer = int(sys.argv[3])
    cfg = make_config(sys.argv[2])
    pfx = f"language_model.backbone.layers.{layer}."
    W = {k: ck.weight(pfx + "mixer." + k + ".weight").double() for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    nw = ck.weight(pfx + "norm.weight").double()
    x = torch.from_numpy(ref[f"l{layer}_in"]).double()
    T = x.shape[0]
    H, KV, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.layer_norm_epsilon) * nw
    q = (xn @ W["q_proj"].T).view(T, H, Dh)
    k = (xn @ W["k_proj"].T).view(T, KV, Dh)
    v = (xn @ W["v_proj"].T).view(T, KV, Dh)
    o = attention(q, k, v, KV).reshape(T, H * Dh)
    bf = lambda t: t.to(torch.bfloat16).double()
    o16 = attention(q, bf(k), bf(v), KV).reshape(T, H * Dh)
    out = o @ W["o_proj"].T
    y = x + out
    yr = torch.from_numpy(ref[f"l{layer}_out"]).double()
    print(f"layer {layer} attention: transformers block vs naive fp64: rel {float((yr - y).norm() / out.norm()):.2e}; "
          f"BF16 K/V changes o by rel {float((o16 - o).norm() / o.norm()):.2e}")
    if "--dump" in sys.argv:
        d = sys.argv[sys.argv.index("--dump") + 1]
        os.makedirs(d, exist_ok=True)
        f32 = lambda t: t.float().contiguous().numpy()
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            ck.raw(pfx + "mixer." + name + ".weight").to(torch.bfloat16).view(torch.int16).numpy().tofile(
                os.path.join(d, f"w_{name}.bf16"))
        for name, t in {"x": xn, "q": q.reshape(T, -1), "k": k.reshape(T, -1), "v": v.reshape(T, -1), "o": o,
                        "o_bf16kv": o16, "out": out}.items():
            f32(t).tofile(os.path.join(d, f"{name}.f32"))
        json.dump({"T": T, "layer": layer, "n_embd": cfg.hidden_size, "n_head": H, "n_head_kv": KV, "head_dim": Dh},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"wrote test vectors to {d}")


if __name__ == "__main__":
    main()
