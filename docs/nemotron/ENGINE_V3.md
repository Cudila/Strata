# strata-nemotron: engine v3 — prefill 1270 → 7400 tok/s, 1M context (2026-10-19)

The goal: match Qwen3.8-Flash-Next on Strata in prefill (~4400 tok/s at 9K on this machine) and give the model a 1M
context. (Russian original: `ru/ENGINE_V3.md`.)

## Result (RTX 5090, 14700KF, DDR5-5200; prefill with the expert cache warm, MTP on)

| prompt length | v2, tok/s | v3 (first version), tok/s | v3 now, tok/s |
|---|---:|---:|---:|
| 128 (a short follow-up in chat, MTP on) | ~180 | 750 | 850 (150 ms) |
| 2K | — | 1950 | 2500 |
| 8K | 1270 | 7300 | 7800 |
| 16K | — | 8900 | 10900 |
| 32K (through the server) | — | 7600 | 9000 |
| 256K (through the server) | — | 5560 | 6600 |
| 1M (bench, cache cold) | — | 2750 | 2980 |

The second wave (2026-10-19):
- attention: tiles of 128 keys, exponentials in FP16, row sums through mma with ones, lazy rescaling, a simpler
  loader (266 → 312 TFLOP/s equivalent);
- experts: the down projection writes straight into (token, rank) slots with one sum in rank order (instead of a
  double accumulator); BF16 output of the up projection;
- activation quantization in one pass from registers;
- the FP4 GEMM reads the experts' ggml rows itself (no repacking);
- the shared expert is queued before the router syncs;
- the SSD's C·Bᵀ once per group;
- weight-based skipping with prediction up to 8192 rows;
- split-K attention with a piece length that does not depend on the window, so speculation is bit-identical to
  decode again.

The third wave (2026-10-19, cache warm, the second prompt in a row) reached: 128 tokens in 98 ms, 2K at 2650 tok/s,
16K at 12 700, 64K at 11 270, 256K at 7 440 (from 7 050). What it did:
- the NVFP4 weight is converted to N4 once per chunk, not on every 2048-row sub-chunk;
- W8A8 on int8 tensor cores for the Q8_0 attention and the BF16 fc1/fc2 (`nemo_q8.cu`, instead of BF16 + cuBLAS);
  the first prompt after load is ~0.5 s faster;
- the SSD without padding of the B/C rows (−7% kernel time).

Prefill at 16K–64K is now bound by compute (~1.17 s of kernels per 16K chunk) and PCIe (~1.0 s of copies) at once.
So making one kernel faster saves only a half to a quarter of its gain. Details and what was dropped:
OPTIMIZATION_NOTES.md. Stages 7–8: STAGE7-8_RESULTS.md.

Decode without speculation: 98–102 tok/s on short context, 77 tok/s at 256K. With speculation: 155–163 tok/s in chat
(temperature 1.0), up to 159 tok/s at 256K. A sweep of the MTP settings after all changes (length 4–7 × cut-off
0.7–0.9) gives a plateau of 155–163; `--spec 5 --spec-min-p 0.8` is on it.

Quality (pairwise ΔNLL per token, wikitext 40 × 1024):
- W4A4 (FP4 tensor cores in every NVFP4 GEMM of the prompt): +0.0028 ± 0.0019 (PPL ×1.003); decode is not affected.
- int8-K/FP16-V attention: prompt +0.0035…0.0049 (×1.004–1.005); the decode path against v2: −0.0033 ± 0.0020 (no
  worse).
- `--prompt-skip-w 0.3` (only chunks of ≤ 1024 rows): ×1.005; τ = 0.2 is free, τ = 0.4 gives ×1.014.
- The prompt is deterministic: the result does not depend on what is in the expert cache (the expert sum is in
  double). Speculative greedy is bit-identical to plain decode (300/300).

## What was done (in order of contribution)

1. **FP4 tensor cores (W4A4)**, `nemo_fp4.cu`, its own library for sm_120a:
   - `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`: NVFP4 × NVFP4 with ue4m3 scales per 16 values;
   - activations are quantized per row: a global scale amax/(448·6), e4m3 per 16 values, e2m1 RNE;
   - weights are repacked from ggml/tiled to "natural" nibble order (N4);
   - 128×128 blocks, k = 128 per stage, 3 cp.async stages, ldmatrix for the fragments: 792 TFLOP/s on the in_proj shape
     (mma peak 1610; cuBLAS BF16 ~150);
   - experts run as one grouped launch over a list of tiles;
   - test `nemo_fp4_parity`: quantization equals the host byte for byte; GEMM against float64.
2. **Expert copies in parallel with compute.**
   - Misses go on their own stream into a ring of 384 blobs, or into an empty cache slot.
   - The next MoE layer is planned ahead: a chunk of ≥ 512 rows needs almost all experts.
   - Hits are computed first.
   - A layer's metadata goes in one copy from pinned memory. Before, pageable copies for every group stopped the host.
3. **Attention on tensor cores.**
   - K in the cache is int8 with a scale per (cell, KV head), V is FP16.
   - Q·K is int8 mma (m16n8k32, 806 TOP/s); P·V is FP16 mma with FP16 sums per 64-key tile, then FP32.
   - A warp handles 16 heads of one token over their shared KV head (GQA 16:1).
   - 8K: 440 → 17 ms; 256K: 30 → 19 s.
   - Decode and the verify window on ≥ 16K keys use the same split-K code (a block reads each key once for all window
     tokens): decode at 256K 80 → 159 tok/s.
4. **Mamba.** A chunked SSD (Q = 32, FP32 on CUDA cores, state in registers, chunk prefetch) for T ≥ 64. A sequential
   scan, bit-identical to the decode step, for verify windows. 245 → 117 ms at 8K.
5. **Chunk buffer tails lent to the cache.** While the chunk is no longer than 8192, rows 8192–16383 of its buffers go
   to the expert cache (+4 slots). A long chunk takes them back.
6. **Prediction for short prompts with skipping** (≤ `--prompt-skip-max`, `STRATA_NEMO_PROMPT_PRED`). The next layer's
   router runs on the current residual, and its likely misses (weight ≥ τ) are copied ahead: 2048 tokens 933 → 819 ms,
   4096 tokens 1184 → 946 ms.
7. **Small things:**
   - the router through SGEMM (343 → 10 ms);
   - fast dequantization kernels;
   - MoE planning on the host by counting sort (9.5 → 4 ms per layer at chunk 16K);
   - chunk 16K (half the copies per token on long prompts);
   - group buffers ≥ the chunk: at 16K one expert can take > 8192 rows. That was an out-of-bounds write, now fixed.
8. **Skipping misses in a short prompt** (`--prompt-skip-w τ --prompt-skip-max N`). A miss whose largest weight in the
   chunk is < τ is not copied. A short prompt touches almost all 512 experts of a layer (~40 GB over PCIe), and its
   time is set by copying, not compute: 128 tokens 0.7 s → 85 ms.

## 1M context

Nemotron's attention is full: every decode step reads the whole KV (~6.5 GiB at 1M). Streaming the KV from RAM, as
Strata does for Flash-Next (which has sparse QSA attention), would cost ~0.2 s per token. So the KV lives in VRAM, but
elastically:
- every attention layer and the MTP get pages of 65 536 cells;
- above `--kv-base` (65 536), a page is added when the context grows into it, and its memory comes from the expert
  cache (bands of 4 slots of every layer);
- on the next conversation (`reset`), the pages go back to the cache;
- the kernels address a cell through a page table in their parameters (up to 64 pages).

Short conversations do not notice any of this. Expert cache: 140 slots a layer at 64K, 128 at 256K.

Needle in a haystack through the server (`tools/needle_bench.py`, depths 10/50/90%, greedy):

| length | found | prefill | slots a layer | KV |
|---|---|---:|---:|---:|
| 32K | 3/3 | 4.4 s (7600 tok/s) | 140 | 64K |
| 128K | 3/3 | 18 s (6950 tok/s) | 132 | 128K |
| 256K | 3/3 | 47 s (5560 tok/s) | 128 | 256K |
| 512K | 3/3 | 117–125 s (~4300 tok/s) | 112 | 512K |
| 1M (1 027 882 tokens) | 3/3 | 374 s (2750 tok/s) | 84 | 1M |

After a 1M request, a new short conversation takes the KV back to 64K and the cache to 136 slots. A request of
1 053 087 tokens is refused properly: 400, the prompt plus max_tokens exceed the window, nothing is cut.

The server reports the context as 1048576: `meta.n_ctx` in `/v1/models`, `n_ctx` in `/props`, `max_context` in
`/health`.

## Engine options (new)

`--kv-base N` (the initial KV), `--prompt-skip-w τ --prompt-skip-max N`, `--chunk 16384`.

Environment variables for probes:
- `STRATA_NEMO_FP4`: bits 1 dense, 2 experts' up, 4 down; 0 means BF16 + cuBLAS;
- `STRATA_NEMO_RING`;
- `STRATA_NEMO_ATTN_TC_KEYS`: the threshold for tensor-core attention in decode;
- `STRATA_NEMO_SEQ_SCAN`;
- `STRATA_NEMO_A4`: W4A4 emulation in the BF16 path;
- `STRATA_NEMO_HOSTPROF`.

Tests, all PASS:
- `nemo_fp4_parity`;
- `nemo_gemm_parity`, with the row-wise NVFP4 dequantization;
- `mamba_parity`: the SSD against the reference and against the sequential scan on 2000 rows;
- `attn_parity`: paged KV, int8 K;
- `nemo_iq_parity`;
- `moe_parity`.

Attention bench: `nemo_attn_bench [p0] [T]`.

## Not done / next

- Attention runs at ~40% of its theoretical ceiling. It is register-bound: one block per SM, so the mma and softmax
  phases do not overlap. At 1M, prefill takes minutes, and attention is most of that time.
- int8 K/Q scales per group of 32 would win back part of attention's ×1.004, at a cost of ~20–30% of its speed.
- 1K–8K prompts are still bound by ~1 s of expert copies (weight skipping gets in the way of the prefetch there).
- Repacking experts to N4 for every group (~90 ms at 8K) and quantizing the up input per row (repeated 22×) could
  both be removed by reading the ggml format directly in the GEMM.
