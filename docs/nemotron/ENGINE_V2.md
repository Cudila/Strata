# strata-nemotron: engine v2 — decode 21 → 150–190 tok/s (2026-10-18)

The goal: match Qwen3.8-Flash-Next on Strata in decode (156–270 tok/s with MTP on this machine). (Russian original:
`ru/ENGINE_V2.md`.)

## Result (RTX 5090, 14700KF, DDR5-5200, NVFP4 experts, sampling at temperature 1.0)

| mode | decode, tok/s |
|---|---:|
| v1 | 21 |
| v2 without speculation (`--skip-w 0.4`, `--prefetch 0`) | 105–110 |
| v2 + MTP (`--spec 5 --spec-min-p 0.8`), bench, 4 prompts × 500 tokens | 172–186 |
| through the server, 1500 tokens | 152–191 |

Quality: the pairwise ΔNLL of the decode path (wikitext, 8 × 1024, 4088 tokens) with `--skip-w 0.4` is +0.0046 ±
0.0027 (PPL ×1.005). Speculation is exact: greedy without skipping is bit-identical to plain decode (300/300).

## What was done (in order of contribution)

1. **Skipping cache misses by routing weight** (`--skip-w τ`).
   - An expert that is not in VRAM and whose weight is < τ in every row of the window is not copied. It counts as a zero
     expert.
   - Misses are mostly rare experts with small weights, so the cost is small.
   - Skipping by rank (as `ROUTE_TAIL_SKIP` in Strata) is worse in quality at the same speed.
   - τ = 0.3/0.4/0.5: PPL ×1.0054/×1.0046/×1.0044 (within 2σ); speculation speed 136 → 150 → up to 180.
2. **MTP speculation** (`spec_step`).
   - Drafts come from the MTP block (blk.88): eh_proj, NoPE attention, MoE over 512 resident NVFP4 experts, a
     LayerNorm head, the shared lm_head, and its own KV.
   - The verify window of ≤ 8 rows goes through the decode path.
   - Mamba rollback: a snapshot of the states, plus a replay of conv/scan for the accepted rows from the saved in_proj
     rows. The MTP's KV is written with the true h.
   - Sampling is exact: each row's token is drawn from the main model's distribution, and a draft is accepted when it
     matches.
   - The chain stops at a draft whose MTP probability is < `--spec-min-p`.
   - A greedy probe: of 4 drafts, 2.5–3.25 are accepted on average.
3. **Tiled tensor-core GEMVs** (`nemo_tiled_*`).
   - Dense NVFP4/Q8_0/BF16 weights are repacked at load so that each lane reads exactly its B fragments of
     mma.m16n8k16.
   - M = 1..8 costs almost as much as M = 1: in_proj 36 → 38 µs for M 1 → 8, against 361 µs on CUDA cores.
   - A row does not depend on M, so the verify window is bit-identical to decode.
4. **Fast CUDA-core GEMVs** (`nemo_gemv`).
   - The input sits in shared memory, padded against bank conflicts; grid-stride loops; a LUT in shared memory.
   - A wide variant for matrices with few rows.
   - Experts are batched (`nemo_gemv_many`).
5. **Cache.**
   - A copy stream; the hit experts are computed first.
   - LFU with decay 0.97 (`--evict lfu`): −7% misses against LRU on a trace. Belady gives −55%, which is not reachable.
6. **Prefetch** (the next layer's router on the current residual) helped without skipping. With `--skip-w` it does
   twice the harm, so it is off (`--prefetch 0`).

## Tried and dropped

- Ready-made 2-bit experts, AtomicChat AD-Q2_0_L. The IQ3_XXS/IQ4_XS/IQ4_NL/Q2_0 kernels exist (`--experts-gguf`).
  PPL is +8–12% against NVFP4, and with weight skipping NVFP4 is faster too. Not used.
- An MTP expert cache smaller than 512 with miss skipping: the share of accepted drafts falls more than the cache
  grows.
- PCIe 5 on this machine gives 41 GB/s at any copy size (measured), so resizing or merging copies does not help.

## Engine options

`--spec N --spec-min-p p --skip-w τ --tail-skip R --prefetch 0|1 --evict lru|lfu --mtp-slots N --experts-gguf F`.

Bench: `--bench a.txt,b.txt --spec-list 4,5,6 --min-p-list 0.7,0.8 --temp 1.0`, `--mtp-probe N`,
`--ppl-decode --ll-dump F`.

Tests, all PASS: `nemo_gemm_parity` (with tc/tiled), `nemo_iq_parity`, mamba/attn/moe parity.

## Next

- Prefill: 1270 tok/s at 8K against ~4400 for Flash-Next. Expert copies in parallel with compute, FP4 MMA.
- Verify window: experts in one pass over the union (now each row reads its own expert).
- Prefetch only the experts with weight ≥ τ.
- A CPU lane for misses.
