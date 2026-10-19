#!/usr/bin/env python3
# tools/nemotron/mamba_naive.py - Nemotron-H's Mamba-2 mixer written out from its formulas, token by token, in float64.
#
# The specification the engine's kernels follow (stage 2), independent of transformers' chunked and recurrent paths:
#
#   zxbcdt = in_proj @ x                       z 8192 | xBC 10240 | dt 128   (in that order)
#   xBC    = silu(conv1d_k4(xBC) + bias)       depthwise, causal, over the last 4 inputs (3 of history)
#   x, B, C = split(xBC)                       x 8192 = 128 heads x 64;  B, C 1024 = 8 groups x 128
#   dt     = softplus(dt + dt_bias)            no clamp (time_step_limit is 0..inf)
#   dA     = exp(dt * A),  A = -exp(A_log)     one scalar per head
#   h[hd, p, n] = dA * h + dt * x[hd, p] * B[g(hd), n]        g(hd) = hd // 16
#   y[hd, p]    = sum_n h[hd, p, n] * C[g(hd), n] + D[hd] * x[hd, p]
#   y      = rms_norm over 8 groups of 1024 (y * silu(z)) * norm.weight
#   out    = out_proj @ y
#
# Usage: python mamba_naive.py <ref.npz from ref_layers.py> <BF16 checkpoint dir> [layer] [--dump DIR]
#
# --dump DIR writes the stage 2 test vectors for src/kernels/mamba_parity.cpp: the layer's weights as the engine
# holds them (in_proj/out_proj BF16 bits, the rest f32, conv taps in ggml order c*4+i) and, per token, the mixer
# input, zxbcdt, the gated-norm output and the mixer output, plus the final SSM (H, P, N) and conv (C, 3) states -
# all from this float64 computation, written as raw little-endian files with meta.json.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_layers import Ckpt, make_config  # noqa: E402


def mamba_forward(x, W, cfg, state=None, conv=None):
    """x: (T, 4096) float64 -> (out (T, 4096), state (H, P, N), conv (3, conv_dim))."""
    H, P, N, G = cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size, cfg.n_groups
    d_in, conv_dim, K = H * P, H * P + 2 * G * N, cfg.conv_kernel
    eps = cfg.layer_norm_epsilon
    if state is None:
        state = torch.zeros(H, P, N, dtype=torch.float64)
    if conv is None:
        conv = torch.zeros(K - 1, conv_dim, dtype=torch.float64)
    A = -torch.exp(W["A_log"])
    cw = W["conv1d.weight"][:, 0, :]          # (conv_dim, K): weight[:, K-1] multiplies the newest input
    outs = []
    trace = {"zxbcdt": [], "y_core": []}
    for t in range(x.shape[0]):
        zxbcdt = W["in_proj.weight"] @ x[t]
        trace["zxbcdt"].append(zxbcdt)
        z, xbc, dt = zxbcdt[:d_in], zxbcdt[d_in:d_in + conv_dim], zxbcdt[d_in + conv_dim:]
        win = torch.cat([conv, xbc[None]], 0)    # (K, conv_dim), oldest first
        conv = win[1:]
        xbc = torch.nn.functional.silu((win.T * cw).sum(-1) + W["conv1d.bias"])
        xs, B, C = xbc[:d_in].view(H, P), xbc[d_in:d_in + G * N].view(G, N), xbc[d_in + G * N:].view(G, N)
        dt = torch.nn.functional.softplus(dt + W["dt_bias"])
        g = torch.arange(H) // (H // G)
        state = torch.exp(dt * A)[:, None, None] * state + (dt[:, None] * xs)[:, :, None] * B[g][:, None, :]
        y = (state * C[g][:, None, :]).sum(-1) + W["D"][:, None] * xs
        y = y.reshape(d_in) * torch.nn.functional.silu(z)
        yg = y.view(G, d_in // G)
        y = (yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + eps)).reshape(d_in) * W["norm.weight"]
        trace["y_core"].append(y)
        outs.append(W["out_proj.weight"] @ y)
    mamba_forward.trace = {k: torch.stack(v) for k, v in trace.items()}
    return torch.stack(outs), state, conv


def main():
    ref = np.load(sys.argv[1])
    ck = Ckpt(sys.argv[2])
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    cfg = make_config(sys.argv[2])
    p = f"language_model.backbone.layers.{layer}."
    W = {k: (ck.weight(p + "mixer." + k) if k.endswith("weight") else ck.raw(p + "mixer." + k)).double()
         for k in ["in_proj.weight", "conv1d.weight", "conv1d.bias", "dt_bias", "A_log", "D", "norm.weight",
                   "out_proj.weight"]}
    nw = ck.weight(p + "norm.weight").double()
    x = torch.from_numpy(ref[f"l{layer}_in"]).double()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.layer_norm_epsilon) * nw
    out, state, conv = mamba_forward(xn, W, cfg)
    y = x + out
    yr = torch.from_numpy(ref[f"l{layer}_out"]).double()
    rel = float((yr - y).norm() / out.norm())
    sr = torch.from_numpy(ref[f"l{layer}_ssm_state"]).double()[0]
    srel = float((sr - state).norm() / state.norm())
    print(f"layer {layer}: transformers chunked (fp32) vs naive fp64: mixer output rel {rel:.2e}, final state rel {srel:.2e}")
    np.savez(sys.argv[1].replace(".npz", f"_naive_l{layer}.npz"), out=y.numpy(), state=state.numpy(), conv=conv.numpy())
    if "--dump" in sys.argv:
        d = sys.argv[sys.argv.index("--dump") + 1]
        os.makedirs(d, exist_ok=True)
        bf = lambda name: ck.raw(p + "mixer." + name).to(torch.bfloat16).view(torch.int16).numpy()
        f32 = lambda t: t.float().contiguous().numpy()
        files = {
            "w_in_proj.bf16": bf("in_proj.weight"),                     # [18560, 4096] row-major = ggml [4096, 18560]
            "w_out_proj.bf16": bf("out_proj.weight"),                   # [4096, 8192]
            "conv_w.f32": f32(W["conv1d.weight"][:, 0, :]),             # [10240, 4]: c*4 + i, tap 3 the newest
            "conv_b.f32": f32(W["conv1d.bias"]),
            "dt_bias.f32": f32(W["dt_bias"]),
            "A.f32": f32(-torch.exp(W["A_log"])),
            "D.f32": f32(W["D"]),
            "norm_w.f32": f32(W["norm.weight"]),
            "x.f32": f32(xn),                                           # [T, 4096] the mixer's input (block-normed)
            "zxbcdt.f32": f32(mamba_forward.trace["zxbcdt"]),           # [T, 18560]
            "y_core.f32": f32(mamba_forward.trace["y_core"]),           # [T, 8192]
            "out.f32": f32(out),                                        # [T, 4096]
            "state.f32": f32(state),                                    # [128, 64, 128] = (H, P, N)
            "conv_state.f32": f32(conv.T),                              # [10240, 3]: per channel, oldest first
        }
        for k, v in files.items():
            v.tofile(os.path.join(d, k))
        import json
        json.dump({"T": int(x.shape[0]), "layer": layer, "n_embd": cfg.hidden_size, "H": cfg.mamba_num_heads,
                   "P": cfg.mamba_head_dim, "N": cfg.ssm_state_size, "G": cfg.n_groups, "d_conv": cfg.conv_kernel,
                   "eps": cfg.layer_norm_epsilon}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"wrote test vectors to {d}")


if __name__ == "__main__":
    main()
