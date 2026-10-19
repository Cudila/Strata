#!/usr/bin/env python3
# tools/nemotron/tensor_map.py - the stage 1 tensor map: every language-model tensor of a Nemotron-H checkpoint
# (safetensors headers only), grouped by name pattern, with its shape, dtype, the layers it occurs on and their type.
#
# Usage: python tensor_map.py <checkpoint dir> [<second checkpoint dir>] > TENSOR_MAP.md
# With two dirs (e.g. BF16 and NVFP4) the dtype/shape columns show both.  Headers are read from the shards present;
# for a partly downloaded checkpoint pass --index-only to read the shapes from the HF headers dump instead.
import collections
import json
import os
import re
import struct
import sys


def headers(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for shard in sorted(set(idx.values())):
        p = os.path.join(d, shard)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = (v["dtype"], tuple(v["shape"]), v["data_offsets"][1] - v["data_offsets"][0])
    return out


def pattern(name):
    m = re.match(r"language_model\.(backbone|mtp)\.layers\.(\d+)\.(.*)$", name)
    if not m:
        return name.replace("language_model.", ""), None
    rest = re.sub(r"experts\.\d+\.", "experts.E.", m.group(3))
    return f"{m.group(1)}.layers.L.{rest}", int(m.group(2))


def ranges(ls):
    ls = sorted(ls)
    if not ls:
        return "-"
    out, a, b = [], ls[0], ls[0]
    for x in ls[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"{a}" if a == b else f"{a}-{b}")
        a = b = x
    out.append(f"{a}" if a == b else f"{a}-{b}")
    s = ",".join(out)
    return s if len(s) < 60 else f"{len(ls)} layers ({ls[0]}..{ls[-1]})"


def main():
    dirs = sys.argv[1:]
    cfg = json.load(open(os.path.join(dirs[0], "config.json")))
    llm = cfg.get("llm_config", cfg)
    types = llm["layers_block_type"]
    hs = [headers(d) for d in dirs]
    groups = collections.OrderedDict()
    for name in sorted(set().union(*hs), key=lambda k: (pattern(k)[1] is None, pattern(k)[0])):
        if not name.startswith("language_model."):
            continue
        p, layer = pattern(name)
        g = groups.setdefault(p, {"layers": set(), "count": 0, "info": [None] * len(hs), "bytes": [0] * len(hs)})
        g["count"] += 1
        if layer is not None:
            g["layers"].add(layer)
        for i, h in enumerate(hs):
            if name in h:
                g["info"][i] = h[name][:2]
                g["bytes"][i] += h[name][2]

    print(f"# Nemotron 3.5 Super VL: карта тензоров языковой модели\n")
    print(f"Сгенерировано `tools/nemotron/tensor_map.py` из заголовков safetensors: {', '.join(os.path.basename(d.rstrip('/')) for d in dirs)}.")
    print("Слои `backbone` нумеруются 0..87, тип слоя из `layers_block_type`; `mtp.layers.0` — attention-подслой MTP, "
          "`mtp.layers.1` — его MoE-подслой. `E` — номер эксперта 0..511. Объём — по всем вхождениям шаблона.\n")
    pat = "".join({"mamba": "M", "moe": "E", "attention": "*"}[t] for t in types)
    print(f"Порядок блоков: `{pat}`\n")
    hdr = "| Шаблон | Тип слоя | Слои | Кол-во | " + " | ".join(
        f"{os.path.basename(d.rstrip('/'))}: dtype, форма, объём" for d in dirs) + " |"
    print(hdr)
    print("|" + "---|" * (4 + len(dirs)))
    for p, g in groups.items():
        if p.startswith("backbone.layers"):
            kinds = sorted(set(types[l] for l in g["layers"]))
            kind = "/".join(kinds)
        elif p.startswith("mtp."):
            kind = "mtp-" + ("attention" if ".layers.L." in p and min(g["layers"]) == 0 and max(g["layers"]) == 0 else
                             "moe" if min(g["layers"]) == 1 else "both")
        else:
            kind = "global"
        cells = []
        for i in range(len(hs)):
            if g["info"][i] is None:
                cells.append("—")
            else:
                dt, sh = g["info"][i]
                cells.append(f"{dt} {list(sh)} {g['bytes'][i] / 2**20:.1f} MiB")
        print(f"| `{p}` | {kind} | {ranges(g['layers']) if g['layers'] else '-'} | {g['count']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
