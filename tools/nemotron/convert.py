#!/usr/bin/env python3
# tools/nemotron/convert.py - Nemotron 3.5 Super VL (ModelOpt NVFP4 checkpoint) -> GGUF, through the vendored llama.cpp
# converter.
#
# The converter knows NemotronHForCausalLM and finds it through `llm_config.architectures`; what it does not do for the
# VL checkpoint is drop the vision tower, so `vision_model.*`, `mlp1.*` and `vision_projector.*` are filtered here.
#
# --fp8 picks what happens to the checkpoint's FP8 E4M3 per-tensor weights (Mamba in_proj/out_proj, attention q/k/v/o,
# lm_head):
#   bf16   the converter's own path: dequantized to BF16 (exact; the reference arm)
#   nvfp4  re-encoded as ModelOpt NVFP4 (E2M1, E4M3 scale per 16, F32 scale per tensor) from the dequantized values,
#          then packed by the converter's NVFP4 path like the experts
#
# Usage: python convert.py --fp8 nvfp4 <checkpoint dir> <out.gguf>
import argparse
import re
import os
import sys

LLAMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp")
sys.path.insert(0, LLAMA)
sys.path.insert(0, os.path.join(LLAMA, "gguf-py"))

import torch  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX = 448.0


def _round_e2m1(x):
    """Nearest E2M1 code of x (already divided by its block scale): magnitude index | sign bit, no negative zero."""
    idx = (x.abs().clamp(max=6.0).unsqueeze(-1) - E2M1.to(x.device)).abs().argmin(-1)   # ties go to the smaller index
    return idx | ((x < 0) & (idx > 0)).to(torch.int64) << 3


def nvfp4_encode(w, search=8, rows_per_chunk=2048):
    """ModelOpt's NVFP4 encoding of a 2-D float tensor: returns (packed U8 [out, in/2], E4M3 [out, in/16], F32 scale2).
    scale2 = amax / (6 * 448).  A block's scale is the E4M3 code, among the `search` codes either side of amax/6/scale2,
    with the smallest squared error after rounding to E2M1: on the checkpoint's own experts this reproduces 99.2% of
    ModelOpt's block scales (the plain amax/6 rounding: 32%)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.float()
    out, inn = w.shape
    assert inn % 16 == 0, inn
    scale2 = w.abs().max() / (6.0 * FP8_MAX)
    if scale2 == 0:
        scale2 = torch.tensor(1.0)
    lut = torch.cat([E2M1, -E2M1]).to(dev)
    packed = torch.empty(out, inn // 2, dtype=torch.uint8)
    bscale = torch.empty(out, inn // 16, dtype=torch.float8_e4m3fn)
    s2 = scale2.to(dev)
    for r0 in range(0, out, rows_per_chunk):
        blocks = w[r0:r0 + rows_per_chunk].to(dev).reshape(-1, inn // 16, 16)
        base = (blocks.abs().amax(-1) / 6.0 / s2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).int()
        best_err, best_code = None, None
        for d in sorted(range(-search, search + 1), key=abs):
            c = (base + d).clamp(0, 126).to(torch.uint8)                 # 126 = 448, the largest finite E4M3
            s = c.view(torch.float8_e4m3fn).float() * s2
            sd = torch.where(s == 0, torch.ones_like(s), s)
            q = lut[_round_e2m1(blocks / sd.unsqueeze(-1))] * s.unsqueeze(-1)
            err = ((q - blocks) ** 2).sum(-1)
            if best_err is None:
                best_err, best_code = err, c
            else:
                take = err < best_err                                    # strict: the code nearest base wins ties
                best_err = torch.where(take, err, best_err)
                best_code = torch.where(take, c, best_code)
        s = best_code.view(torch.float8_e4m3fn).float() * s2
        sd = torch.where(s == 0, torch.ones_like(s), s)
        code = _round_e2m1(blocks / sd.unsqueeze(-1)).reshape(blocks.shape[0], inn).to(torch.uint8)
        packed[r0:r0 + rows_per_chunk] = (code[:, 0::2] | (code[:, 1::2] << 4)).cpu()   # low nibble = even element
        bscale[r0:r0 + rows_per_chunk] = best_code.view(torch.float8_e4m3fn).cpu()
    return packed, bscale, scale2.reshape(()).float()


def fp8_names(tensors):
    """The FP8 weights: a `.weight` whose `.weight_scale` is a scalar."""
    out = []
    for name in list(tensors):
        if not name.endswith(".weight"):
            continue
        s = name[: -len(".weight")] + ".weight_scale"
        if s in tensors and name[: -len(".weight")] + ".weight_scale_2" not in tensors:
            out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", choices=["bf16", "nvfp4"], required=True)
    ap.add_argument("model")
    ap.add_argument("outfile")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--nvfp4-only", help="with --fp8 nvfp4: re-encode only the FP8 weights whose name matches this "
                    "regex (the others go to BF16), to localize what the re-encoding costs")
    a, rest = ap.parse_known_args()

    import conversion.base as base
    from conversion.base import LazyTorchTensor
    from conversion import nemotron

    cls = nemotron.NemotronHModel
    orig_filter = cls.filter_tensors.__func__

    def filter_tensors(c, item):
        if item[0].startswith(("vision_model.", "mlp1.", "vision_projector.")):
            return None
        # NemotronHModel.filter_tensors tests "mtp." before the "language_model." prefix is stripped, so --no-mtp
        # would let the MTP experts through as layer 1's (duplicated blk.1.ffn_*_exps)
        if c.no_mtp and item[0].startswith("language_model.mtp."):
            return None
        return orig_filter(c, item)

    cls.filter_tensors = classmethod(filter_tensors)

    orig_prepare = base.ModelBase.prepare_tensors

    def rename_mtp(self):
        """The NVFP4 path packs a tensor under its checkpoint name, before NemotronHModel.modify_tensors renames the MTP
        head (`mtp.layers.0/1.*` -> the folded block `self._mtp_bid`), so the MTP's NVFP4 experts and shared expert
        cannot be mapped.  Rename them, scale sidecars included, by modify_tensors' own rules first."""
        bid = self._mtp_bid
        if bid is None:
            return
        for name in [n for n in self.model_tensors if n.startswith(("mtp.layers.0.", "mtp.layers.1."))]:
            special = cls._MTP_SPECIAL_RENAMES.get(name)
            new = special.format(bid=bid) if special else f"backbone.layers.{bid}.{name.split('.', 3)[3]}"
            self.model_tensors[new] = self.model_tensors.pop(name)

    def reencode_fp8(self):
        names = fp8_names(self.model_tensors)
        if a.nvfp4_only:
            names = [n for n in names if re.search(a.nvfp4_only, n)]
        print(f"convert.py: re-encoding {len(names)} FP8 weights as NVFP4", file=sys.stderr)
        for name in names:
            stem = name[: -len(".weight")]
            wgen, sgen = self.model_tensors[name], self.model_tensors[stem + ".weight_scale"]

            def enc(wgen=wgen, sgen=sgen, cache={}):
                if "r" not in cache:
                    w = LazyTorchTensor.to_eager(wgen()).float() * LazyTorchTensor.to_eager(sgen()).float()
                    cache["r"] = nvfp4_encode(w)
                return cache["r"]

            self.model_tensors[name] = lambda enc=enc: enc()[0]
            self.model_tensors[stem + ".weight_scale"] = lambda enc=enc: enc()[1]
            self.model_tensors[stem + ".weight_scale_2"] = lambda enc=enc: enc()[2]
            # the FP8 input_scale is a W8A8 activation scale, meaningless for NVFP4: drop it
            self.model_tensors.pop(stem + ".input_scale", None)

    def prepare_tensors(self):
        if isinstance(self, cls):
            # _generate_nvfp4_tensors writes a layer's experts once it holds `num_local_experts`/`num_experts` of them;
            # Nemotron names the count n_routed_experts, so without this it held all 59 GiB of them (OOM at 110 GB)
            self.hparams.setdefault("num_local_experts", self.hparams["n_routed_experts"])
            rename_mtp(self)
            if a.fp8 == "nvfp4":
                reencode_fp8(self)
        return orig_prepare(self)

    base.ModelBase.prepare_tensors = prepare_tensors

    orig_dequant = base.ModelBase.dequant_model

    def dequant_model(self):
        # this checkpoint says quant_method "modelopt_mixed"; dequant_model's FP8 branch is keyed on "modelopt"
        qc = self.hparams.get("quantization_config") or {}
        if qc.get("quant_method") == "modelopt_mixed":
            qc["quant_method"] = "modelopt"
        return orig_dequant(self)

    base.ModelBase.dequant_model = dequant_model

    sys.argv = ["convert_hf_to_gguf.py", a.model, "--outfile", a.outfile] + (["--no-mtp"] if a.no_mtp else []) + rest
    import convert_hf_to_gguf
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()
