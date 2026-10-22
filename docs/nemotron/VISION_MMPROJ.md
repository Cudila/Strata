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

## Vision in strata-nemotron (2026-10-19)

Images work end to end: through the server (OpenAI image_url parts, Anthropic image blocks) and through llama-swap.

**The pieces:**
- **Encoder**: `strata-vision` (tools/vision), built against llama.cpp `098ce526b` (2026-10-19; `nemotron_omni` since
  `cd0a5b6a1`) with `tools/vision/patches/llama.cpp-nemotron-omni-cpe.patch`, and Unsloth's `mmproj-BF16.gguf`.
- **The patch** fixes upstream's position embeddings for non-square pictures. RADIO's CPE resizes the 128×128 table to
  a square of the longer side and takes its top-left rows × cols. Upstream stretched it to rows × cols, which gave a
  per-token cosine down to 0.32 against NVIDIA's code on a 48×64-patch page.
- **NVIDIA's tile, in strata-vision.** For `nemotron_omni`, strata-vision picks NVIDIA's tile size itself
  (`_compute_target_patches`, with Python's round-half-to-even) and resamples like torch's antialiased bicubic. Only
  then does mtmd get the picture. mtmd's generic dynamic-size rule picks other tiles: a 640×488 photo became 300
  tokens there, against NVIDIA's 320.
- **Engine**: `GENI <max_new> [keys] <file> <ids>`. Each `<img><image>…</img>` run of the prompt takes one strata-vision
  record, in order. The rows replace the token rows in the main model and in the MTP block's input. The conversation
  reuse matches pictures by their hashes (`img_keys`), so a different picture of the same size is never reused.
- **Server**: for this tokenizer, each `<image>` the template writes becomes `<img>` + one `<image>` per image token
  + `</img>`, as in NVIDIA's processor.

**Correctness** (`tools/nemotron/vision_check.py`: strata-vision against `tools/nemotron/vision_naive.py`, a float32
transcription of NVIDIA's processor, RADIO and projector):

| image | tokens | CPU: rel err / min cos | GPU: rel err / min cos |
|---|---:|---|---|
| document 768×1024 | 768 | 0.0039 / 0.9996 | 0.0068 / 0.9986 |
| chart 1024×512 | 512 | 0.0049 / 0.9998 | 0.0066 / 0.9985 |
| photo 640×488 (resized to 640×512) | 320 | 0.0091 / 0.9967 | — |
| square 768×768 | 576 | 0.0029 / 0.9999 | — |

Before the patch, the same document gave rel err 0.258 and min cos 0.32.

**End to end** (greedy, thinking off, through the server). All answers were right:

| input | answer |
|---|---|
| invoice | the total ($98.60) and the customer |
| chart | the tallest bar (orange) |
| photo | "front page of The New York Times from July 21, 1969 … MEN WALK ON MOON" |
| two images in one message | the invoice number and the bar count |
| a 1920×1080 screenshot | the two failed tests |
| follow-up turn on the screenshot | 2061 of 2107 tokens reused, 146 ms |
| the same conversation with a mirrored chart | not reused, answered from the new picture |

**Cost:**
- One encoder call takes 36 ms (512 tokens) to 463 ms (3300 tokens). That is the whole call, `OK … ms` of
  `strata-vision`: image decode, resize, preprocessing, the GPU encode and the file write, not only the GPU. Since
  2026-10-22 (threaded resize) 1920×1080 takes 221 ms and 2560×1440 424 ms; the GPU part alone is ~181 and ~358 ms.
- The encoder holds 2.9 GB of VRAM, so the engine gets 116 expert slots a layer instead of 140. The 24 GB emulation
  showed that half the cache costs only a few %.
- Text after the change is identical: PPL 3.2120 on 8 × 1024, as before.

**Not done:**
- Video: `video_embedder` and the token pruning across frames.
- An elastic encoder, which would start with the first picture and give its VRAM back when idle. The engine has no
  `VRAM` command yet.
