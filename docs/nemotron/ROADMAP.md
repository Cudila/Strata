# strata-nemotron: plan of work (2026-10-19)

Two blocks. The main line comes first: what users of the published model get. The research block runs in parallel:
each item there starts with a question and a measurement, not with code.

## Main line

### 1. Video

The model takes video. NVIDIA's serving examples (vLLM/SGLang) sample **2 fps, at most 256 frames**, and the card
says mp4 up to 2 minutes. The model itself fixes no frame rate: the processor labels each frame with its timestamp
("Frame 1 sampled at 0.50 seconds and frame 2 sampled at 1.00 seconds: ").

What one frame costs:
- each frame is resized to ~1024 patches with its aspect ratio kept (`video_target_num_patches`);
- two frames make one temporal patch (`video_temporal_patch_size` 2): `video_embedder` (16×16×6 → 1280) reads both;
- one `<img>…</img>` chunk per pair: ~256 tokens after the pixel shuffle;
- `video_pruning_rate` 0.7 (0.5 in NVIDIA's vLLM command) drops tokens similar to the previous frame's
  (`compute_retention_mask`, EVS).

So 256 frames are 128 chunks × 256 = ~32K tokens before pruning and ~10–16K after.

To do:
- frame sampling and decoding in the server (fps 2, ≤ 256 frames);
- the two-frame path in strata-vision (upstream mtmd has `build_inp_with_temporal_merge`);
- the pruning, against NVIDIA's code: a float32 reference first, as for images;
- the timestamp labels in the prompt;
- a check on real clips.

### 2. Demo: per-model speed histograms in llama-swap's activity page

The page shows one prompt / generation histogram over all models, which mixes engines. Each model should get its own
histograms.

### 3. 16 GB VRAM + 64 GB RAM: cold experts on the SSD

**16 GB VRAM: done (2026-10-19).** `--vram-limit-mib`, the small-card profile, and a fix for a verify window that wants
more distinct experts than a layer has slots (such a miss is now skipped, not written past the slots). Numbers: README
*16 GB cards*. What is left is the RAM side: 64 GB.

Today the engine needs 32 GB of VRAM and 128 GB of RAM:
- on 16 GB only 13 expert slots a layer fit, and 22 are needed for one token;
- the 59 GiB arena of page-locked experts does not fit in 64 GB.

The plan:
- **experts in three tiers**: VRAM cache → a RAM cache of the warm experts → NVMe for the cold ones (O_DIRECT reads into
  pinned blocks, a fixed-size block per expert);
- **the VRAM floor**: the fixed part is ~11.5 GB at chunk 4096 (dense 6.6 GB, chunk buffers, KV, scratch). Smaller
  chunk buffers, and the MTP experts as ordinary cache entries instead of resident ones;
- **measure first**: the routing frequency per expert from real traffic, how much of it a RAM cache of 24–40 GB
  holds, and NVMe read rates at this block size.

## Research block (in parallel)

### 4. relu² zeros out of the compute graph: sparse GEMM/GEMV

After relu², most of an expert's hidden values are exactly zero: 92.5% according to an independent 8 GB-GPU port
(["Running NVIDIA's new 120B Nemotron-3.5-Super on an 8 GB GPU — skipping the zeros"](https://dev.to/helgard_orlm/running-nvidias-new-120b-nemotron-35-super-on-an-8-gb-gpu-skipping-the-zeros-and-finding-the-44o4), code:
[helgard-orlm/nemotron-super-8gb](https://github.com/helgard-orlm/nemotron-super-8gb)). There the `down` weights were
re-laid out on disk so that the zero neurons' rows are not read: decode went from 6.5 to 7.3–7.6 tok/s. The
kernels pass over them quickly already, but they still read every column of `down_proj` and do the arithmetic.

Questions:
- the zero share per layer and per token on our traffic, and how stable it is;
- decode: does reading only the `down_proj` columns of the nonzero hidden values cut memory traffic in proportion?
  This needs a layout where a hidden value's column is contiguous;
- prompt: does a gathered/sparse GEMM beat the dense FP4 tensor-core GEMM at this density? Dense FP4 runs at
  ~790 TFLOP/s, so the bar is high;
- what this does to the cache: fewer bytes per miss if experts are fetched by column.

The answer must be exact: a skipped value is exactly zero, so the output must not change.

### 5. Less memory without going below NVFP4

Ideas to measure:
- **experts that are never or rarely routed**: per-layer frequency profiles; whether some can stay on the SSD for good;
- **the dense part**: lm_head in NVFP4 costs +1.3% PPL (measured), so look for cheaper wins first: the MTP block's
  attention, the embeddings in RAM only;
- **buffers**: chunk buffers shared with the expert cache (partly done), the attention scratch, the K/V at int8 for V
  as well;
- **RAM**: the page-locked arena is a second copy next to the file's page cache. Read experts from the file into
  pinned blocks on demand instead.

Each idea goes in with its VRAM/RAM saving, its speed cost and its quality cost (PPL / KL against the published file).
