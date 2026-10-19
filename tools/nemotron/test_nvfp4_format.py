#!/usr/bin/env python3
# tools/nemotron/test_nvfp4_format.py - stage 1 check of how the ModelOpt NVFP4 checkpoint is read.
#
#   1. decode: an NVFP4 tensor decoded as  e2m1(nibble) * e4m3(weight_scale[row, col/16]) * weight_scale_2,
#      low nibble = even element, against the same tensor of the BF16 checkpoint.  A wrong nibble order, scale layout or
#      a missing factor shows as a relative error near 1 instead of NVFP4's ~0.1.
#   2. encode: convert.py's nvfp4_encode on the BF16 tensor against the checkpoint's bytes (how close our encoder is to
#      ModelOpt's: the FP8 -> NVFP4 re-encoding uses it).
#   3. ggml packing: the converter's _nvfp4_pack decoded with ggml's block_nvfp4 layout gives the same values.
#
# Usage: python test_nvfp4_format.py <NVFP4 dir> <BF16 dir> [tensor ...]
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import E2M1, nvfp4_encode  # noqa: E402

LUT = torch.cat([E2M1, -E2M1])


def load(d, name):
    shard = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"][name]
    with safe_open(os.path.join(d, shard), "pt") as f:
        return f.get_tensor(name)


def decode(packed, scale, scale2):
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    v = LUT[codes]
    s = scale.float().repeat_interleave(16, dim=1)
    return v * s * scale2.float()


def decode_ggml(raw, rows, cols):
    # block_nvfp4: 4 UE4M3 scales, then 32 bytes; byte j of sub-block s holds elements j (low) and j+8 (high)
    raw = torch.from_numpy(raw).reshape(rows, cols // 64, 36)
    d = raw[..., :4].contiguous().view(torch.float8_e4m3fn).float()            # UE4M3 = E4M3 with the sign bit 0
    qs = raw[..., 4:].reshape(rows, cols // 64, 4, 8).long()
    v = torch.cat([LUT[qs & 0x0F], LUT[qs >> 4]], dim=-1)                      # (rows, nsup, 4, 16)
    return (v * d.unsqueeze(-1)).reshape(rows, cols)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    nv, bf = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or [
        "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.0.down_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.311.up_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.311.down_proj.weight",
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight",
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight",
    ]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp"))
    from conversion.base import ModelBase  # noqa: E402

    bf_map = json.load(open(os.path.join(bf, "model.safetensors.index.json")))["weight_map"]
    names = [n for n in names if os.path.exists(os.path.join(bf, bf_map[n]))]   # the BF16 side is fetched by shard
    ok = bool(names)
    for name in names:
        stem = name[: -len(".weight")]
        w = load(nv, name)
        sc = load(nv, stem + ".weight_scale")
        s2 = load(nv, stem + ".weight_scale_2")
        ref = load(bf, name).float()
        dec = decode(w, sc, s2)
        e_dec = rel(dec, ref)
        # the alternative readings, to show the test can tell them apart
        e_swap = rel(decode(((w & 0x0F) << 4) | (w >> 4), sc, s2), ref)
        e_nos2 = rel(decode(w, sc, torch.tensor(1.0)), ref)

        pw, psc, ps2 = nvfp4_encode(ref)
        same_bytes = float((pw == w).float().mean())
        same_scale = float((psc.view(torch.uint8) == sc.view(torch.uint8)).float().mean())
        e_enc = rel(decode(pw, psc, ps2), ref)

        raw, shape = ModelBase._nvfp4_pack(w, sc)
        e_ggml = rel(decode_ggml(raw, *shape) * s2.float(), dec)

        print(f"{name.split('backbone.')[1]}: {list(ref.shape)}")
        print(f"  decode vs BF16 rel {e_dec:.4f}   (nibbles swapped {e_swap:.4f}, without scale_2 {e_nos2:.3g})")
        print(f"  our encoder: rel {e_enc:.4f}, same bytes {same_bytes:.4f}, same E4M3 scales {same_scale:.4f}, "
              f"scale_2 {float(ps2):.6g} vs {float(s2):.6g}")
        print(f"  ggml repack decodes to the same values: rel {e_ggml:.2e}")
        ok &= e_dec < 0.2 and e_swap > 2 * e_dec and e_ggml < 1e-6 and e_enc < 1.05 * e_dec + 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
