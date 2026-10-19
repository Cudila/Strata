# tools/nemotron/vision_naive.py - strata-nemotron: a float32 reference of Nemotron 3.5 Super VL's image path, written
# from NVIDIA's modeling_radio.py / modeling_nemotron_h_omni.py / image_processing_nemotron_h_omni.py and run straight
# on the checkpoint's tensors (no transformers): the embeddings that replace an image's <image> tokens.
#
#   python tools/nemotron/vision_naive.py <checkpoint dir> <image> <out.npy> [--device cuda]
#
# Steps, as the checkpoint's code does them:
#   resize: one dynamic-resolution tile (1024..13312 patches of 16 px, a multiple of 2 patches each way), torch bicubic
#           with antialias; normalize with the CLIP mean/std (the model's own input conditioner is switched off)
#   RADIO ViT-H/16: patch projection (768 -> 1280), the 128x128 position table interpolated to the grid (CPE), 10 prefix
#           tokens (3 class + 7 registers), 32 pre-LN blocks (qkv bias, GELU MLP, layer scale 1), no final norm
#   vision_final_layernorm on the patch tokens (present because the language model has MTP), pixel shuffle 0.5
#           (ps_version v2), mlp1: RMSNorm(5120, eps 1e-5) -> Linear 20480 -> relu^2 -> Linear 4096
# Writes float32 (n_tokens, 4096), row i at (x = i % nx, y = i / nx) with nx = w_patches / 2, and prints the grid.
import json, math, os, struct, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PS, MERGE, MIN_P, MAX_P, MAX_LEN = 16, 2, 1024, 13312, 16384


def load(ckpt, names):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    out, by = {}, {}
    for n in names:
        by.setdefault(idx[n], []).append(n)
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
    for f, ns in by.items():
        with open(os.path.join(ckpt, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            for k in ns:
                a, b = h[k]["data_offsets"]
                fh.seek(8 + n + a)
                raw = bytearray(fh.read(b - a))
                out[k] = torch.frombuffer(raw, dtype=dt[h[k]["dtype"]]).reshape(h[k]["shape"]).float()
    return out


def target_patches(w, h, available):
    ph, pw = round(h / PS + 0.5), round(w / PS + 0.5)
    f = min(math.sqrt(available / (ph * pw)), 1.0)
    th, tw = math.floor(f * ph), math.floor(f * pw)
    if available > MIN_P and th * tw < MIN_P:
        up = math.sqrt(MIN_P / (th * tw))
        th, tw = math.ceil(up * th), math.ceil(up * tw)
    r = th % MERGE
    if r:
        th = th + MERGE - r if (th + MERGE - r) * tw <= available else max(MERGE, th - r)
    r = tw % MERGE
    if r:
        tw = tw + MERGE - r if th * (tw + MERGE - r) <= available else max(MERGE, tw - r)
    return tw, th


def main():
    ckpt, img_path, out_path = sys.argv[1:4]
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    img = Image.open(img_path).convert("RGB")
    available = max(min(max((MAX_LEN - 4) * MERGE * MERGE, MIN_P), MAX_P), MIN_P)
    wp, hp = target_patches(img.width, img.height, available)
    t = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float()
    if t.shape[-2:] != (hp * PS, wp * PS):
        t = F.interpolate(t, size=(hp * PS, wp * PS), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(cfg["norm_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["norm_std"]).view(1, 3, 1, 1)
    x = ((t / 255.0 - mean) / std).to(dev)

    p = "vision_model.radio_model.model."
    L = cfg["vision_config"]["num_hidden_layers"]
    names = [p + "patch_generator.embedder.weight", p + "patch_generator.pos_embed", p + "patch_generator.cls_token.token",
             "vision_projector.vision_final_layernorm.weight", "vision_projector.vision_final_layernorm.bias",
             "mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight"]
    for l in range(L):
        for s in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                  "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"):
            names.append(f"{p}blocks.{l}.{s}")
    W = {k: v.to(dev) for k, v in load(ckpt, names).items()}
    E, H = cfg["vision_config"]["hidden_size"], cfg["vision_config"]["num_attention_heads"]
    eps = cfg["vision_config"]["layer_norm_eps"]

    # patches: (rows*cols, C*16*16) in (c, py, px) order, then the projection
    pt = x.reshape(1, 3, hp, PS, wp, PS).permute(0, 2, 4, 1, 3, 5).reshape(hp * wp, 3 * PS * PS)
    h = pt @ W[p + "patch_generator.embedder.weight"].T
    pos = W[p + "patch_generator.pos_embed"].reshape(1, 128, 128, E).permute(0, 3, 1, 2)
    m = max(hp, wp)
    pos = F.interpolate(pos, size=(m, m), mode="bilinear", align_corners=False)[..., :hp, :wp]
    h = h + pos.flatten(2).permute(0, 2, 1)[0]
    pre = W[p + "patch_generator.cls_token.token"]
    h = torch.cat([pre, h], 0)
    D = E // H
    for l in range(L):
        b = f"{p}blocks.{l}."
        y = F.layer_norm(h, (E,), W[b + "norm1.weight"], W[b + "norm1.bias"], eps)
        qkv = y @ W[b + "attn.qkv.weight"].T + W[b + "attn.qkv.bias"]
        q, k, v = (z.reshape(-1, H, D).transpose(0, 1) for z in qkv.split(E, -1))
        a = torch.softmax((q @ k.transpose(1, 2)) * D ** -0.5, -1) @ v
        h = h + a.transpose(0, 1).reshape(-1, E) @ W[b + "attn.proj.weight"].T + W[b + "attn.proj.bias"]
        y = F.layer_norm(h, (E,), W[b + "norm2.weight"], W[b + "norm2.bias"], eps)
        y = F.gelu(y @ W[b + "mlp.fc1.weight"].T + W[b + "mlp.fc1.bias"])
        h = h + y @ W[b + "mlp.fc2.weight"].T + W[b + "mlp.fc2.bias"]
    feat = h[pre.shape[0]:]
    feat = F.layer_norm(feat, (E,), W["vision_projector.vision_final_layernorm.weight"],
                        W["vision_projector.vision_final_layernorm.bias"], eps)
    # pixel shuffle 0.5, ps_version v2 (the code's view/permute sequence on (n, w=rows, h=cols, c))
    z = feat.reshape(1, hp, wp, E)
    n, ww, hh, c = z.shape
    z = z.view(n, ww, hh // 2, c * 2).permute(0, 2, 1, 3).contiguous()
    z = z.view(n, hh // 2, ww // 2, c * 4).permute(0, 2, 1, 3).contiguous()
    z = z.reshape(-1, c * 4)
    z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5) * W["mlp1.0.weight"]
    z = F.relu(z @ W["mlp1.1.weight"].T).pow(2) @ W["mlp1.3.weight"].T
    np.save(out_path, z.float().cpu().numpy())
    print(f"{img_path}: {wp * PS}x{hp * PS} px, {wp}x{hp} patches, {z.shape[0]} tokens (grid {wp // 2}x{hp // 2})")


if __name__ == "__main__":
    main()
