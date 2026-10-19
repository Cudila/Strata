# Nemotron 3.5 Super VL: vision tower and mmproj (inventory, 2026-10-19)

The text path does not use the vision part. This page lists what it is and where llama.cpp stands on it.
Nothing here has been run on an image yet.

## What is in the checkpoint

From the headers of `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (`model.safetensors.index.json`): everything is
BF16 and none of it is quantized.

| part | tensors | size | what |
|---|---:|---:|---|
| `vision_model.radio_model` | 390 | 1.22 GiB | RADIO v2.5-H: ViT-Huge/16, 32 layers, hidden 1280, 16 heads, GELU MLP 5120, qkv bias, LayerNorm eps 1e-6 |
| `patch_generator.cls_token` | 1 | | 10 rows: 3 class tokens (one per teacher) + 7 registers, cut off before the projector |
| `patch_generator.pos_embed` | 1 | 40 MiB | (1, 16384, 1280): a 128×128 grid, interpolated to the image's patch grid (CPE, up to 2048 px) |
| `patch_generator.embedder` | 1 | | linear 768 → 1280 (16×16×3 patch) |
| `patch_generator.video_embedder` | 1 | | linear 1536 → 1280: two frames per patch (`video_temporal_patch_size` 2) |
| `input_conditioner.norm_mean/std` | 2 | | CLIP mean/std (0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758) |
| `vision_projector.vision_final_layernorm` | 2 | | LayerNorm(1280) on the ViT output, weight ≈ 1 (std 0.002), bias ≈ 0 (std 0.001) |
| `mlp1.0` / `mlp1.1` / `mlp1.3` | 3 | 0.35 GiB | RMSNorm(5120, eps 1e-5) → Linear 5120 → 20480 → ReLU² → Linear 20480 → 4096, no bias |

The pipeline (`modeling_nemotron_h_omni.py`) runs in this order:
1. RADIO features of the patch tokens;
2. `vision_final_layernorm`;
3. pixel shuffle 0.5 (2×2 patches → one 5120-wide token, `ps_version` v2);
4. `mlp1`;
5. the resulting embeddings take the place of the `<image>` tokens (id 18).

The final LayerNorm is there only because the language model has MTP. In Megatron, every block built from the shared
config, the vision tower included, gets a final LayerNorm when `mtp_num_layers` > 0. So it is part of the trained
model, not an option.

Images: the processor picks one dynamic-resolution tile per image. It keeps the aspect ratio and uses 1024 to 13312
patches, so an image becomes 256 to 3328 tokens after the pixel shuffle. It normalizes with the CLIP mean/std above.
Video: 1024 patches a frame, two frames per patch through `video_embedder`, and `video_pruning_rate` 0.7 drops
similar tokens across frames.

## mmproj in llama.cpp

**Our local llama.cpp tree** is `llama.cpp-glm5next` at b476d9a76, 2026-10-13. It has only the Nano 12B v2 VL path:
`conversion/nemotron.py` `NemotronNanoV2VLModel`, projector `nemotron_v2_vl`. Read against this checkpoint, that path
gets it wrong in four places:
1. it drops `vision_final_layernorm`;
2. it fixes images at 512×512 (`pos_embed` interpolated to 32×32 at conversion, so 256 tokens an image);
3. it writes ImageNet mean/std, because the checkpoint names its keys `norm_mean` / `norm_std`;
4. it uses RMSNorm eps 1e-6 instead of 1e-5.

**Unsloth's `mmproj-BF16.gguf`** (unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) comes from a newer llama.cpp.
Its header, read on 2026-10-19 from the first 8 MB of the file, shows none of these problems:
- projector `nemotron_omni`;
- `v.post_ln.weight/bias` (that is the final LayerNorm);
- the full `v.position_embd` (1280 × 16384, the 128×128 grid, interpolated at run time);
- `image_min_pixels` 262144 / `image_max_pixels` 3407872 (512² up to 13312 patches);
- CLIP `image_mean` / `image_std`;
- `v.patch_embd.weight.1` (16×16×6, the two-frame video embedder);
- 842M parameters in BF16.

So current llama.cpp handles this tower. The publication does not need its own mmproj. We can point to Unsloth's
file, or convert one with a current llama.cpp.

## To do before a vision publication

- A check on real images, because the header shows what is stored and not that it is computed right. Run the HF code
  in BF16 for the tower and the projector (1.6 GiB, fits on the GPU) and save the embeddings for a photo, a document
  and a screenshot. Compare them per token with llama.cpp on Unsloth's mmproj, and compare the answers to a question
  about each image.
- Update `llama.cpp-glm5next` (or another tree) to a version with `nemotron_omni`, so that a local slot can take images.
- In strata-nemotron, vision is a separate step: the tower plus the projector, and embeddings in place of the
  `<image>` tokens in the prompt path. Video adds `video_embedder` and token pruning across frames.
