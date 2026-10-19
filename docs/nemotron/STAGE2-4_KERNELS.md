# strata-nemotron, stages 2–4: Mamba-2, attention and MoE kernels against per-layer references

Date: 2026-10-17. (Russian original: `ru/STAGE2-4_KERNELS.md`.)

All tests run on real weights and real activations. The input of block L comes from running blocks 0..L-1 of the
model on BF16 weights (`tools/nemotron/ref_layers.py`). The reference is a float64 implementation of the equations,
itself checked against the transformers block.

## References

| reference | checked against transformers | note |
|---|---|---|
| `tools/nemotron/mamba_naive.py` (fp64, per token) | = transformers' recurrent path; transformers' chunked fp32 path differs by 2e-3 (up to 6e-3 by token 28) | transformers' chunked path is not usable as a reference |
| `tools/nemotron/attn_naive.py` (fp64) | block 7: 1.9e-6 (64 tokens), 4.6e-7 (2048 tokens) | plus a variant with K/V rounded to BF16 (the engine's cache) |
| `tools/nemotron/moe_naive.py` (fp64) | block 1 on BF16: 4.0e-7 | the experts are dumped as the checkpoint's NVFP4 bytes in ggml layout |

## Kernels (library `strata_nemotron_kernels`; the qwen4exp path does not link it)

| file | what |
|---|---|
| `src/kernels/cuda/mamba.cu`, `include/strata/kernels/mamba.hpp` | decode: conv + bias + SiLU, the SSM step (a warp per (h, p) row, FP32 state (H, P, N)), gated RMSNorm over groups of 1024; prefill: conv over T tokens, a scan with the state in registers, the norm per row |
| `src/kernels/cuda/nemo_attn.cu`, `nemo_attn.hpp` | NoPE GQA 32/2 × 128, causal, BF16 K/V cache, split-K + merge; one kernel for decode, the verify window and the prompt |
| `src/kernels/cuda/nemo_moe.cu`, `nemo_moe.hpp` | router (sigmoid + bias → top-22, weights = sigmoid, normalized, ×5); NVFP4 W4A16 GEMV (ggml block_nvfp4, the tensor's scale_2); a token's k experts in one launch, plus a deterministic sum |

In the tests, the projections (in/out_proj, q/k/v/o, router, latent) use Strata's existing `bf16_gemv_fp32_mmvf`
(FP32 activations, exact products). Pack formats for them (NVFP4 / Q8_0) come in stage 5.

## Results (relative error ||got − want|| / ||want||, worst token)

**Mamba-2, block 0, 64 tokens** (`mamba_parity`):

| check | error |
|---|---:|
| the kernel between the projections (y after the gated norm) | 5.0e-7 |
| final SSM state / conv state | 1.1e-7 / 0 |
| the whole mixer with BF16 projections | 2.3e-7 |
| prefill in two pieces (37 + 27) against the reference | 5.0e-7 |
| prefill against decode of the same tokens | **bit-identical** (output, state, conv) |
| negative test (conv taps reversed) | 0.81 → caught |

Tolerance: 2e-5 (kernel), 1e-4 (whole mixer).

The prefill scan on an 8K piece takes 7.2–7.6 ms per layer: ~24K tok/s over 40 layers for the Mamba parts without
projections (`mamba_parity --bench`). At the target prefill of ~5K tok/s that is ~20% of the time. So the chunked SSD
(tensor cores) is postponed until a profile of the whole model.

**Attention, block 7** (`attn_parity`), three modes: decode token by token, the prompt in two pieces, verify windows of 4:

| | 64 tokens | 2048 tokens (up to 8 split-K blocks) |
|---|---:|---:|
| the kernel against the reference with BF16 K/V | 5.1e-7 | 1.8e-6 |
| the whole block (our projections) against the exact reference | 2.4e-3 | 1.2e-3 |

The second row's difference is the cost of the BF16 K/V cache; the reference shows the same (2.4e-3 / 1.0e-3).
Kernel tolerance: 1e-5.

Found and fixed: the scratch size is not monotonic in T. A 2011-token request needs 2 splits and a 2048-token one
needs 1. `nemo_attn_scratch_bytes` now returns the maximum over all T' ≤ T.

**MoE, blocks 1 and 5, 64 tokens** (`moe_parity`; NVFP4 experts through pointers, as the cache will hand them over):

| | block 1 | block 5 |
|---|---:|---:|
| router logits | 6.5e-8 | 5.8e-8 |
| top-22 sets that differ from the reference | 0 of 64 | 0 of 64 |
| routing weights | 2.2e-7 | 2.3e-7 |
| sum of the experts in the latent | 1.9e-6 | 3.1e-7 |
| MoE output (+ latent projections + shared) | 5.4e-7 | 2.4e-7 |

Tolerance: 1e-5.

In the reference, the choice between the 22nd and 23rd expert comes closest to flipping at 3.2e-5 / 5.3e-5 of the
selection score. That is the threshold below which any two implementations can pick different sets.

An observation for stages 4/6: over 64 tokens a layer uses 467 (block 1) and 448 (block 5) of its 512 experts.
Routing locality is low, so the union of misses in a verify window will be large.

## What was left for these stages

- A new cache type (SSM 160 MiB + conv per sequence, next to the KV), coming with the engine's session (block
  composition, graphs), as for GDN in Strata.
- An expert frequency profile (after `--dump-routing` / `tools/make_profile.py`), once the engine runs the whole
  model.
- A chunked SSD for prefill, depending on the profile.
