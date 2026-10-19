#!/usr/bin/env python3
# tools/nemotron/moe_naive.py - Nemotron-H's LatentMoE block in float64 (stage 4 specification), checked against
# transformers' block on the same input, and dumped as test vectors for src/kernels/moe_parity.cpp.
#
#   xn      = rms_norm(x) * norm.weight
#   logits  = gate.weight @ xn                       (512)   FP32 in transformers: input and weight cast to f32
#   score   = sigmoid(logits);  choice = score + e_score_correction_bias
#   ids     = top-22 of choice (n_group = 1: the group step selects everything)
#   w       = score[ids] / sum(score[ids]) * 5.0     (the bias only CHOOSES; the weights are the plain sigmoid)
#   lat     = fc1_latent_proj @ xn                   (1024)
#   routed  = sum_j w_j * down_j(relu(up_j(lat))^2)  (experts in the latent space, no gate)
#   moe     = fc2_latent_proj @ routed + shared_down(relu(shared_up(xn))^2)
#   out     = x + moe
#
# With --ckpt NVFP4 the experts and the shared expert are the dequantized NVFP4 weights (decision 3 of the stage 0
# review: W4A16 is the engine's reference).  The dump keeps the experts in the checkpoint's own NVFP4 bytes, repacked to
# ggml's block_nvfp4 by the converter's own function, with each tensor's scale_2, so the C++ test reads exactly what the
# GGUF holds.
#
# Usage: python moe_naive.py <ref.npz> <checkpoint dir> <layer> [--dump DIR]
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp"))
from ref_layers import Ckpt, make_config  # noqa: E402


def relu2(t):
    return torch.relu(t) ** 2


def main():
    ref = np.load(sys.argv[1])
    ck = Ckpt(sys.argv[2])
    layer = int(sys.argv[3])
    cfg = make_config(sys.argv[2])
    pfx = f"language_model.backbone.layers.{layer}.mixer."
    E, K = cfg.n_routed_experts, cfg.num_experts_per_tok
    x = torch.from_numpy(ref[f"l{layer}_in"]).double()
    T = x.shape[0]
    nw = ck.weight(f"language_model.backbone.layers.{layer}.norm.weight").double()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.layer_norm_epsilon) * nw
    gate = ck.weight(pfx + "gate.weight").double()
    bias = ck.raw(pfx + "gate.e_score_correction_bias").double()
    fc1 = ck.weight(pfx + "fc1_latent_proj.weight").double()
    fc2 = ck.weight(pfx + "fc2_latent_proj.weight").double()
    sh_up = ck.weight(pfx + "shared_experts.up_proj.weight").double()
    sh_dn = ck.weight(pfx + "shared_experts.down_proj.weight").double()

    logits = xn @ gate.T
    score = torch.sigmoid(logits)
    choice = score + bias
    top = torch.topk(choice, K + 1, dim=-1)
    ids = top.indices[:, :K]
    margin = (top.values[:, K - 1] - top.values[:, K]).min()          # the closest 22nd/23rd call of the T tokens
    w = score.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
    lat = xn @ fc1.T
    used = sorted(set(ids.flatten().tolist()))
    ups = {e: ck.weight(pfx + f"experts.{e}.up_proj.weight").double() for e in used}
    dns = {e: ck.weight(pfx + f"experts.{e}.down_proj.weight").double() for e in used}
    routed = torch.zeros(T, lat.shape[1], dtype=torch.float64)
    for t in range(T):
        for j in range(K):
            e = int(ids[t, j])
            routed[t] += w[t, j] * (dns[e] @ relu2(ups[e] @ lat[t]))
    shared = relu2(xn @ sh_up.T) @ sh_dn.T
    moe = routed @ fc2.T + shared
    y = x + moe
    yr = torch.from_numpy(ref[f"l{layer}_out"]).double()
    print(f"layer {layer} moe: transformers block vs naive fp64: rel {float((yr - y).norm() / moe.norm()):.2e}; "
          f"{len(used)} distinct experts over {T} tokens; closest top-22 call margin {float(margin):.2e}")

    if "--dump" in sys.argv:
        from conversion.base import ModelBase
        d = sys.argv[sys.argv.index("--dump") + 1]
        os.makedirs(d, exist_ok=True)
        f32 = lambda t: t.float().contiguous().numpy()

        def nvfp4(name):
            """(ggml block_nvfp4 bytes, scale_2) of a checkpoint NVFP4 tensor."""
            raw, _ = ModelBase._nvfp4_pack(ck.raw(name + ".weight"), ck.raw(name + ".weight_scale"))
            return raw, float(ck.raw(name + ".weight_scale_2").float())

        bf = lambda name: ck.raw(name).to(torch.bfloat16).view(torch.int16).numpy()
        bf(pfx + "gate.weight").tofile(os.path.join(d, "w_gate.bf16"))
        bf(pfx + "fc1_latent_proj.weight").tofile(os.path.join(d, "w_fc1.bf16"))
        bf(pfx + "fc2_latent_proj.weight").tofile(os.path.join(d, "w_fc2.bf16"))
        f32(bias).tofile(os.path.join(d, "bias.f32"))
        s2 = []
        with open(os.path.join(d, "experts_up.nvfp4"), "wb") as fu, open(os.path.join(d, "experts_down.nvfp4"), "wb") as fd:
            for e in range(E):
                ru, su = nvfp4(pfx + f"experts.{e}.up_proj")
                rd, sd = nvfp4(pfx + f"experts.{e}.down_proj")
                fu.write(ru.tobytes())
                fd.write(rd.tobytes())
                s2.append((su, sd))
        np.array(s2, dtype=np.float32).tofile(os.path.join(d, "experts_scale2.f32"))        # (E, 2): up, down
        for nm, part in [("shared_up", "shared_experts.up_proj"), ("shared_down", "shared_experts.down_proj")]:
            r, sc = nvfp4(pfx + part)
            r.tofile(os.path.join(d, nm + ".nvfp4"))
            np.array([sc], dtype=np.float32).tofile(os.path.join(d, nm + "_scale2.f32"))
        for nm, t in {"x": xn, "logits": logits, "w": w, "lat": lat, "routed": routed, "shared": shared,
                      "moe": moe}.items():
            f32(t).tofile(os.path.join(d, nm + ".f32"))
        ids.int().numpy().astype(np.int32).tofile(os.path.join(d, "ids.i32"))
        json.dump({"T": T, "layer": layer, "n_embd": cfg.hidden_size, "n_expert": E, "k": K,
                   "latent": cfg.moe_latent_size, "ff": cfg.moe_intermediate_size,
                   "shared_ff": cfg.moe_shared_expert_intermediate_size, "scale": cfg.routed_scaling_factor},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"wrote test vectors to {d}")


if __name__ == "__main__":
    main()
