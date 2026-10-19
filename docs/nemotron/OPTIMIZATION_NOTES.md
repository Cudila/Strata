# Nemotron 3.5 Super on one RTX 5090: what it takes to optimize, and what costs performance

These are the findings from porting Nemotron 3.5 Super VL (120B-A12B: a hybrid of Mamba-2, MoE and attention) to the
Strata engine on one machine: RTX 5090 32 GB (sm_120), 128 GB DDR5-5200, PCIe 5, i7-14700KF. Numbers are measured on
this machine unless said otherwise. (Russian original: `ru/OPTIMIZATION_NOTES.md`.)

## 1. Hardware: what is actually available

The desktop on this machine runs on the same card. Its compositor and the screen-capture portal lose buffers when VRAM
runs out, and the portal also loses its capture permission. So 1 GB of VRAM is always kept free, and the engine runs
with `--vram-reserve-mib 1024`.

| resource | figure | note |
|---|---:|---|
| PCIe 5 host→device | ~41 GB/s | at any copy size (measured): the ceiling for experts streamed from RAM |
| VRAM | ~1.5 TB/s | decode GEMVs are bound by this |
| mma BF16 / FP16, FP32 accumulate | 256 TFLOP/s | GeForce: FP32 accumulation runs at full rate only for BF16/FP16 |
| mma FP16, FP16 accumulate | 512 TFLOP/s | twice as fast; the sums are FP16 |
| mma FP8 e4m3, FP32 accumulate | ~450 TFLOP/s | |
| mma FP8 e4m3, FP16 accumulate | ~830–1020 TFLOP/s | |
| mma int8, int32 accumulate | ~800–1020 TOP/s | exact sums |
| mma NVFP4 (`kind::mxf4nvf4.block_scale`) | ~1610 TFLOP/s | sm_120a only; ue4m3 scales per 16 values, exactly the NVFP4 format |
| FP64 | ~1/64 of FP32 | double accumulators cost both memory and compute |

Conclusion: on GeForce Blackwell, format is speed. The BF16 path through cuBLAS reaches ~150–235 TFLOP/s; the FP4
path gives ~800 on the real shapes. A model that already ships in NVFP4 should run on FP4 mma, not be widened to BF16.

## 2. Architecture choices that cost performance

1. **Full attention in a hybrid.** 8 attention layers out of 88 blocks is little compute on short context, but every
   decode step reads the whole KV cache: ~6.5 GiB per token at 1M (int8 K / FP16 V). So the KV cannot be kept in RAM
   and streamed, as Strata does for Qwen3.8-Flash-Next. That model has sparse QSA attention, and decode reads only the
   selected cells. Over PCIe this would cost ~0.2 s per token. The KV has to live in VRAM, where it takes space from
   the expert cache: at 1M, 140 → 84 slots per layer. Prefill on long context is quadratic: at 1M, attention is most of
   the time (minutes).
2. **MoE with 512 experts, top-22, ~3.1 MB per expert (NVFP4).** 63 GB of experts do not fit in 32 GB. A decode token
   needs 22 × 40 = 880 expert-layers. Top-22 is very wide routing: even a short prompt (100–500 tokens) touches almost
   all 512 experts of every layer, which is ~30–40 GB over PCIe, about 1 s. The time of a short prompt is set by
   copying, not by compute. The wider the top-k for the same number of experts, the worse short-prompt prefill gets on
   cards with little VRAM.
3. **Latent MoE (4096 → 1024 → experts → 1024 → 4096).** The upside: experts are small (1024×2688), with 1024-wide input
   and output. The downside: two extra dense matrices (fc1/fc2) per MoE layer, and narrow expert GEMMs (K = 1024) with
   lower arithmetic intensity.
4. **relu² in the experts and the shared expert.** The activations after relu² have a heavy tail (the input of the down
   projection), which is awkward for 4-bit activation quantization. In practice W4A4 cost ×1.003 PPL. The input of the
   experts' up projection (the latent) turned out to be the most sensitive: ×1.004 of the total ×1.006 in emulation.
5. **Mamba-2 with 128 heads × 64, state 128, 8 groups.** The sequential scan on prefill is slow and latency-bound, so
   a chunked SSD is needed. The state has to be saved and restored when MTP drafts are rolled back. MTP speculation
   needs the conv/scan of the accepted rows replayed: Mamba cannot be rolled back any cheaper than that.
6. **NoPE attention, GQA 32:2, head_dim 128.** The upside: no RoPE, and 16 heads share one KV head, so a warp handles
   16 heads of one token and reads the key once. The downside: only 2 KV heads give little head parallelism in decode,
   so split-K over the keys is needed.
7. **Mixed-precision checkpoint.** It has NVFP4 experts, FP8 per-tensor Mamba in/out and attention, and BF16 for the
   rest. The FP8 layers have no direct fast path in llama.cpp formats. Our choice: in/out_proj in NVFP4, attention in
   Q8_0, which together cost +1.6% PPL.
8. **The MTP block has its own 512 experts** (1.6 GB resident). It speeds decode up 1.7–1.9×, but takes ~13 cache
   slots per layer, which hurts short prompts noticeably (they are bound by copies).

## 3. Details you need to understand to optimize this

### Decode
- The routing weight is the best criterion for skipping a miss. An expert that is not in VRAM and has weight < 0.4 in
  every row of the window is not copied (PPL ×1.005). Skipping by rank, as Strata does, is worse at the same speed.
- LFU with decay 0.97 gives −7% misses against LRU. Static placement by frequency is worse than LRU. Belady gives −55%
  (a ceiling, not reachable).
- Prefetch by prediction (the next layer's router on the current residual) helps without miss skipping. Together with
  skipping it does twice the harm, because the predicted experts are the ones that would have been skipped anyway.
- A tiled format for tensor-core GEMV: the weights are repacked at load so that each lane of a warp reads exactly its
  B fragment of mma.m16n8k16. Then M = 1..8 costs almost as much as M = 1, and the MTP verify window is nearly free.
- The verify window is bit-identical to decode only if each row is computed independently of M, in its own order of
  sums. This property can be tested (speculative greedy = plain decode), and the test catches many bugs.
- Attention in decode on long context: a CUDA-core kernel with a warp reduction per (key, window token) is
  compute-bound, not memory-bound. Tensor-core split-K, where a block reads each key once for all window tokens, makes
  the window twice as fast.

### Prefill
- Expert copies must run in parallel with compute: their own stream, a ring of blobs, release events. Compute the hits
  first (they are in VRAM) and the misses as they arrive. Plan the next layer ahead: on a large chunk almost all
  experts are needed, so copy them blind.
- If hits are computed first, the order in which expert contributions are summed depends on what is in the cache, and
  the prompt becomes nondeterministic: PPL drifts by 0.1% between runs, and speculation stops matching. Summing in
  double fixes this, because a sum of 22 exact products does not depend on order.
- A pageable `cudaMemcpyAsync` is synchronous. One such metadata copy per expert group stopped the host and left the
  GPU idle: 2.4 s of idle time at 8K. Anything copied on every step has to go through pinned memory.
- FP4 GEMM: what matters most is the k depth per stage between `__syncthreads`. 64 values per stage gave 390
  TFLOP/s, 128 gave 790. When both operands are in "natural" nibble order, their fragments can be loaded with
  `ldmatrix` as b16 matrices.
- A short prompt is bound by PCIe, a long one by compute. Chunk size: 16K halves the copies per token on long prompts,
  but the chunk's buffers take ~1 GB from the cache. The tails of the buffers can be lent to the cache while the chunk
  is short.
- One expert can get every row of the chunk, so group buffers must hold the whole chunk. At chunk 16K this was an
  out-of-bounds write.
- Mamba scan: the sequential version is bound not by shuffles but by dependent loads of the tile from global memory.
  Prefetching the next tile into registers gave more than any change to the arithmetic. A chunked SSD in FP32 gave
  another ×2.
- Check W4A4 activation quantization by emulation (fake-quant in the BF16 path) before writing the kernel. The
  emulation, with double rounding, is even more pessimistic than the real kernel (×1.0064 against ×1.003).

### Attention on tensor cores
- int8 for Q·K with one scale per vector: ~0.3% output error (BF16: 0.13%, FP8 e4m3: 1.6%). P in 8 bits (u8 or
  e4m3) gives 1.7–2%, so P·V runs in FP16 with FP16 sums per tile (0.32%).
- int8 K in the cache also makes the KV a quarter smaller and halves the K reads in decode.
- The kernel is register-bound (~200 per thread): one block of 8 warps per SM. `__syncthreads` on every tile lines
  the warps up, so mma and softmax do not overlap (~40% of the theoretical ceiling).

### Where the time actually goes (measured, not guessed)
- Switching parts of a kernel off is a cheap profiler when the counters are not available (ncu needs privileges).
  Attention without either mma still took 43 of 68 ms, so the limit is not the tensor cores. Timing the phases with
  `clock64` showed 25% "waiting for data". That turned out to be the loader's address arithmetic: divisions and page
  recomputation for every 16-byte piece. After simplifying it, the share fell to 16%, but the total barely changed:
  with 2 warps per scheduler the phases are latency-bound, and the time saved flows into the neighbouring phases.
- Limiting registers (`__launch_bounds__(256, 2)`) to get 2 blocks per SM causes spills and is not faster.
- Activation quantization was bound by memory, not compute. The experts' up-projection output in FP32 is ~3.9 GB per
  layer at chunk 16K. A BF16 output halves that round trip (relu² is already applied, and it goes to 4 bits next
  anyway).
- Summing expert contributions "as the groups go" moves an 8 KB accumulator row for every (token, group) pair, all
  for 1–2 rows of contribution. Writing the down GEMM's output straight into a (token, rank) slot and summing the 22
  slots once in rank order is both faster (10.4K tok/s against 8.6K at 32K) and deterministic without double.
- Repacking expert weights into another format for every group is expensive on short prompts: there, almost all
  experts are hits with 1–5 rows, so the GEMM should read the storage format itself. On long prompts it pays to repack
  once and use the fast loader. 4-byte cp.async (ggml's 36-byte blocks are not 16-byte aligned) is worse than both.
- PCIe: plain pinned memory gives 45–46 GB/s, a page-locked file mapping ~41–45. That is the ceiling for "experts
  from RAM" on this platform.

### MTP
- An attention window for the MTP block (the draft sees only the last W cells) made prefill 4.5% faster at W = 8K. But
  acceptance became unstable: on a long text with repeats, W = 32K dropped acceptance from 4.6 to 2.6 tokens a window.
  So it is off.
- The settings have a plateau: draft length 4–7 × cut-off 0.7–0.9 all give 155–163 tok/s, and the differences are
  within sampling noise.

### Third wave (2026-10-19): what helped and what did not
- **Where prefill stops at 16K–64K.** Profile of a 16K chunk, cache warm: ~1.17 s of kernels, ~1.0 s of expert copies
  over PCIe (46 GB), 25–45 ms idle. Both resources are almost fully busy. So making one kernel N ms faster saves only
  N/2–N/4 in total: 77 ms saved on the attention/fc GEMMs became ~6 ms per prompt. A 32K chunk (half the copies per
  token) does not help even after the kernels got faster (64K: −1%, 256K: +1%), so the chunk stays at 16K.
- **Repacking weights on every sub-chunk.** Mamba and the shared expert run in sub-chunks of 2048 rows, and the
  NVFP4 weight was converted to N4 8 times per chunk. A two-weight cache in `wb` (the loops alternate in/out and
  up/down) saves 39 ms of kernels per chunk: +3% at 32K.
- **W8A8 on int8 tensor cores** (`nemo_q8.cu`) for the Q8_0 attention and the BF16 fc1/fc2, instead of a BF16
  expansion plus cuBLAS. The kernel reaches 390 TOP/s against 233 TFLOP/s for cuBLAS on a 16K×4096×4096 shape.
  ΔNLL −0.0004 ± 0.0022 (activations in blocks of 32, as MMQ does). On a warm prompt it saves 16 ms per chunk, because
  the narrow k/v with N = 256 load the card poorly. The first prompt after load is faster by ~0.5 s: there is no
  cuBLAS warm-up. The kernel is bound by issuing instructions for the block scales (4 FFMA/FMUL per k32 mma), not by
  barriers. A stage depth of 64 → 128 values gave +1.5%; the "magic number" trick instead of int→float gave nothing.
- **Quantizing the experts' latent once per token** instead of 22 times (once per expert), with the rows gathered in
  the GEMM. Quantization got 26 ms faster, but the grouped GEMM with row gathering got 25 ms slower, so the net is zero
  (probe `STRATA_NEMO_LATQ=1`).
- **The router on int8** (the FP32 weight in Q8 blocks): +0.12% PPL for ~1% speed. Not taken (probe: bit 4 of
  `STRATA_NEMO_Q8`).
- **Mamba SSD.** 256 blocks (head × half of the columns) with 60 KB of shared memory give one block per SM and one and
  a half waves on 170 SMs. Two blocks per SM (PS = 16) are slower (0.71 against 0.54 ms per 2048 tokens), because each
  block reloads B/C and recomputes the dt scan. B/C rows without padding (bank conflicts at most 2-way): 0.538 → 0.500
  ms. Splitting the sequence into segments with a state correction does not pay off at one block per SM, by
  estimate.

## 4. What not to do (tried and dropped)

- Ready-made 2-bit experts (AD-Q2_0_L): PPL +8–12%, and NVFP4 with miss skipping is faster anyway.
- An MTP expert cache smaller than 512 together with miss skipping: draft acceptance falls more than the cache gains.
- FP8 for all of attention: 12–15× the error of BF16.
- Streaming the KV from RAM for full attention (see 2.1).
- Skipping misses by weight on 2K–8K prompts without prediction: it disables the prefetch and slows things down. With
  prediction (the next layer's router on the current residual), it speeds them up: 2K 1049 → 816 ms.
- TF32 for the router: ΔNLL +0.002 and no speed gain.
- A 32K chunk for long prompts: half the copies per token, but the same time (±1%).
- Gathering latent rows inside the expert GEMM instead of quantizing per expert: the GEMM eats the quantization gain.

## 5. Decode: skipping misses, quality against speed (2026-10-19)

Measured with the decode path (`--ppl-decode`, wikitext, 4 × 1024, PPL against no skipping at all). The slot runs
`--skip-w 0.4 --prefetch 1`:

| setting | decode PPL | × | decode with MTP at 4K, tok/s | routings skipped |
|---|---:|---:|---:|---:|
| no skipping (`--skip-w 0`) | 2.2428 | 1 | 43 | 0 |
| `--skip-w 0.1`, prefetch off | 2.2484 | 1.003 | 45 | |
| `--skip-w 0.2`, prefetch off | 2.2820 | 1.017 | 73 | |
| `--skip-w 0.3`, prefetch off | 2.3465 | 1.046 | 113 | |
| `--skip-w 0.4`, prefetch off | 2.4459 | **1.091** | 129 | 37% |
| **`--skip-w 0.4`, prefetch on (the slot)** | 2.2611 | **1.008** | 85 | 10% |
| `--skip-w 0.2`, prefetch on | 2.2573 | 1.006 | 64 | |

Why the threshold bites: the router's weights are normalized and then multiplied by 5, so a token's 22 weights average
~0.23. A threshold of 0.4 is above the average, and without the prefetch a non-resident expert below it is simply
dropped: 37% of all routings. Giving the dropped weight to the kept experts (renormalizing) makes it worse: 2.4150 at
0.3, 2.6427 at 0.4.

The prompt path is not affected (its skipping is `--prompt-skip-w`, measured at ×1.005 on chunks ≤ 1024 rows).

More decode speed has to come from exact methods: the CPU computing missed experts while the GPU works (Strata's CPU
pool for swift15), and fewer bytes per miss (the sparse `down_proj` below, ROADMAP item 4).

## 6. Research measurements (2026-10-19)

**relu² zeros** (`STRATA_NEMO_ZERO_STATS=1`, decode path, 1.6M expert rows):
- 92.1% of the hidden values after relu² are exactly zero (layers 85–99%);
- only 33.7% of the 16-value blocks (NVFP4's scale block) are all zero, and 5.9% of the 64-value ggml superblocks;
- so skipping whole blocks in today's layout saves about a third of `down_proj`'s reads. Reaching the 92% needs a
  column layout of `down_proj`: the 4-bit codes of a hidden value's column contiguous, the scales of a block read
  once for its 16 columns. Estimated ~84% less `down` traffic, bit-exact (the same codes and scales, read
  differently). `up_proj` cannot be skipped: its input, the latent, is dense.

**Routing skew** (`STRATA_NEMO_ROUTE_TRACE`, a 64K-token prompt + 512 decoded tokens, a static cache of the most
used (layer, expert) pairs):

| RAM cache | share of decode routings | share of prompt routings |
|---:|---:|---:|
| 16 GiB (27% of the experts) | 71% | 54% |
| 24 GiB | 84% | 68% |
| 32 GiB | 92% | 80% |
| 40 GiB (68%) | 97% | 89% |

9% of the experts were never used in decode on this text. This is one text and an ideal static cache, so mixed
traffic will do worse.

**NVMe** (XPG MARS 980 BLADE, random 3 MiB `O_DIRECT` reads of the GGUF): 5.5 GB/s at QD 1 and ~7.0 GB/s at QD 2–16,
which is ~2,200 experts/s or 0.45 ms per expert.

**A small VRAM cache** (`--expert-slots N` on the 32 GB card, prefetch off, skip 0.4): 140 / 48 / 24 slots gave decode
PPL 2.4357 / 2.6425 / 2.8442. With a small cache, skipping removes even more of the routing. A 16 GB card needs exact
miss handling (CPU lane, SSD tier), not more skipping.
