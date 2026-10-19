#!/usr/bin/env python3
# tools/nemotron/ref_layers.py - PyTorch references for single Nemotron-H blocks (stages 2-4).
#
# Runs the model's first blocks 0..L on real tokens, one `NemotronHBlock` at a time with that block's real weights
# (transformers' own modules, pure-PyTorch paths: the `kernels` package is not installed), and saves each block's input
# and output.  Only the shards holding those blocks are needed, so the BF16 checkpoint can be fetched shard by shard.
#
#   --ckpt BF16 dir     the reference ("what the model means")
#   --ckpt NVFP4 dir    the same blocks with the NVFP4 / FP8 weights dequantized (W4A16 / W8A16): decision 3 of the
#                       stage 0 review, the arm the engine's logits are compared with
#
# Every Mamba block is run twice: once over the whole sequence (the chunked SSD form) and once token by token through
# the cache (the recurrent form).  Their difference is reported - it is the reference's own floor - and the final SSM
# and conv states are saved for the engine's decode test.
#
# Usage: python ref_layers.py --ckpt DIR --layers 2 --tokens 512 --out ref.npz [--embed-dir NVFP4 dir] [--device cuda]
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_nvfp4_format import decode as nvfp4_decode  # noqa: E402

TEXT = (
    "The quick brown fox jumps over the lazy dog. Speculative decoding drafts several tokens and verifies them in one "
    "pass of the target model.\n"
    "def chunked_scan(x, dt, A, B, C, chunk=128):\n    out = []\n    for i in range(0, x.shape[1], chunk):\n"
    "        out.append(step(x[:, i:i + chunk]))\n    return torch.cat(out, 1)\n"
    "Модель состоит из чередующихся слоёв Mamba-2, внимания и смеси экспертов; каждый токен проходит через 22 из 512 "
    "экспертов. Состояние SSM имеет фиксированный размер и не растёт с длиной контекста.\n"
)


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def has(self, name):
        return name in self.map

    def raw(self, name):
        shard = self.map[name]
        if shard not in self.files:
            p = os.path.join(self.d, shard)
            if not os.path.exists(p):
                raise FileNotFoundError(f"{name} is in {shard}, which is not downloaded ({p})")
            self.files[shard] = safe_open(p, "pt")
        return self.files[shard].get_tensor(name)

    def weight(self, name):
        """A weight as float32, dequantized when the checkpoint stores it as NVFP4 or FP8."""
        w = self.raw(name)
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        if stem and self.has(stem + ".weight_scale_2"):
            return nvfp4_decode(w, self.raw(stem + ".weight_scale"), self.raw(stem + ".weight_scale_2"))
        if stem and self.has(stem + ".weight_scale") and w.dtype == torch.float8_e4m3fn:
            return w.float() * self.raw(stem + ".weight_scale").float()
        return w.float()


def make_config(d):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    cfg = json.load(open(os.path.join(d, "config.json")))
    llm = dict(cfg.get("llm_config", cfg))
    llm.pop("architectures", None)
    llm.pop("dtype", None)
    # the Hub config writes inf as {"__float__": "Infinity"}, which the strict dataclass rejects
    tsl = llm.get("time_step_limit")
    if tsl is not None:
        llm["time_step_limit"] = tuple(float(v["__float__"]) if isinstance(v, dict) else float(v) for v in tsl)
    return NemotronHConfig(**llm)


def load_block(block, ck, layer, n_experts):
    p = f"language_model.backbone.layers.{layer}."
    sd = {}
    for k in block.state_dict():
        if k.startswith("mixer.experts."):
            proj = k.split(".")[2]   # up_proj / down_proj
            sd[k] = torch.stack([ck.weight(f"{p}mixer.experts.{e}.{proj}.weight") for e in range(n_experts)])
        else:
            sd[k] = ck.weight(p + k) if k.endswith(".weight") else ck.raw(p + k).float()
    missing, unexpected = block.load_state_dict(sd, strict=True), None
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embed-dir", help="checkpoint dir holding embeddings + tokenizer (default: --ckpt)")
    ap.add_argument("--layers", type=int, default=2, help="run blocks 0..N-1")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-decode", action="store_true", help="skip the token-by-token pass of the Mamba blocks")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock

    torch.manual_seed(0)
    dev = torch.device(a.device)
    ed = a.embed_dir or a.ckpt
    cfg = make_config(a.ckpt)
    cfg._attn_implementation = "eager"
    ck, ek = Ckpt(a.ckpt), Ckpt(ed)

    tok = Tokenizer.from_file(os.path.join(ed, "tokenizer.json"))
    ids = []
    while len(ids) < a.tokens:
        ids += tok.encode(TEXT).ids
    ids = torch.tensor(ids[: a.tokens])
    x = ek.weight("language_model.backbone.embeddings.weight")[ids].to(dev).unsqueeze(0)

    out = {"tokens": ids.numpy(), "layer_types": np.array(cfg.layer_types)}
    report = []
    for layer in range(a.layers):
        kind = cfg.layer_types[layer]
        block = NemotronHBlock(cfg, layer_idx=layer)
        load_block(block, ck, layer, cfg.n_routed_experts)
        block = block.to(dev).float().eval()
        with torch.no_grad():
            pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
            if kind == "full_attention":
                from transformers.masking_utils import create_causal_mask
                mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None,
                                          position_ids=pos)
                y = block(x, attention_mask=mask, position_ids=pos)
            elif kind == "linear_attention":
                cache = DynamicCache(config=cfg)
                y = block(x, past_key_values=cache, use_cache=True)
                lc = cache.layers[layer]
                out[f"l{layer}_ssm_state"] = lc.recurrent_states[0].float().cpu().numpy()
                out[f"l{layer}_conv_state"] = lc.conv_states[0].float().cpu().numpy()
                if not a.no_decode:
                    c2 = DynamicCache(config=cfg)
                    steps = [block(x[:, t:t + 1], past_key_values=c2, use_cache=True) for t in range(x.shape[1])]
                    yd = torch.cat(steps, 1)
                    d_rel = float((yd - y).norm() / (y - x).norm())
                    s_rel = float((c2.layers[layer].recurrent_states[0] - lc.recurrent_states[0]).norm()
                                  / lc.recurrent_states[0].norm())
                    report.append(f"  layer {layer} mamba: chunked vs recurrent: output rel {d_rel:.2e}, "
                                  f"final state rel {s_rel:.2e}")
            else:
                y = block(x)
        out[f"l{layer}_in"] = x.float().cpu().numpy()[0]
        out[f"l{layer}_out"] = y.float().cpu().numpy()[0]
        delta = float((y - x).norm() / x.norm())
        report.append(f"layer {layer} {kind}: |in| rms {float(x.pow(2).mean().sqrt()):.4g}, "
                      f"|out-in|/|in| {delta:.4g}")
        x = y
        del block
    np.savez(a.out, **out)
    print("\n".join(report))
    print(f"saved {a.out}: {len(ids)} tokens, blocks 0..{a.layers - 1}")


if __name__ == "__main__":
    main()
