// include/strata/kernels/nemo_moe.hpp - strata-nemotron stage 4: Nemotron-H's LatentMoE, the parts that are not
// plain BF16 GEMVs (those - the router's logits, fc1/fc2_latent_proj - are bf16_gemv_fp32_mmvf).
//
//   router    sigmoid(logits) + selection bias -> top-k ids; weights = sigmoid at those ids, normalized, x scale
//   experts   relu(up_e(lat))^2 -> down_e, NVFP4 W4A16 (weights decoded in registers, FP32 activations), one launch
//             for all k experts of a token, then the weighted sum
//
// NVFP4 is ggml's block_nvfp4 (64 values: 4 UE4M3 scales, then 32 bytes; byte j of sub-block s holds elements j (low
// nibble) and j+8 (high) of that 16-element sub-block) with the tensor's FP32 scale_2 applied to the row's dot -
// exactly the bytes tools/nemotron/convert.py writes.  The checkpoint's input_scale (W4A4) is not used.
#pragma once

#include "strata/kernels/nemo_gemm.hpp"

#include <cstdint>

namespace strata::kernels {

/// Router for `n_tok` tokens: `logits` (n_tok, n_expert) f32, `bias` (n_expert,).  Writes `ids` (n_tok, k), sorted by
/// descending choice score (ties: lower expert id first), and `weights` (n_tok, k).  n_expert <= 1024, k <= 32.
void nemo_router(const float* logits, const float* bias, int32_t* ids, float* weights, int64_t n_tok,
                 int64_t n_expert, int64_t k, float scale, bool norm, void* stream);

/// y[r] = scale2 * sum_c W[r, c] x[c] for an NVFP4 matrix (rows, cols), cols % 64 == 0.  `relu2` applies relu(.)^2
/// to the result.
void nvfp4_gemv(const uint8_t* w, float scale2, const float* x, float* y, int64_t rows, int64_t cols, bool relu2,
                void* stream);

/// The k experts of ONE token: h_j = relu(scale_up_j * up_j @ lat)^2, out_j = scale_down_j * down_j @ h_j,
/// routed = sum_j weights_j * out_j.  `up` / `down` (k,) device arrays of pointers to each expert's NVFP4 rows
/// (ff x latent and latent x ff), `s_up` / `s_down` (k,) their scale_2.  `h` is k * (ff + latent) floats of scratch
/// (the activations, then the per-expert outputs, summed in expert order so the result does not depend on timing).
void nemo_experts(const uint8_t* const* up, const uint8_t* const* down, const float* s_up, const float* s_down,
                  const float* weights, const float* lat, float* h, float* routed, int64_t k, int64_t latent,
                  int64_t ff, void* stream);

/// nemo_experts in two steps, so the experts already in VRAM can run while the others are copied: `part` computes
/// n of the k experts - entry i is expert idx[i] (device), its up/down pointers and scale_2 at index i - into the
/// scratch rows idx[i] of h (k * ff, then the k parts); `sum` adds the k parts by the weights, in expert order.
/// Entry i reads latent row xidx[i] (a verify window's token; null = row 0) - `lat` has rows of `latent`.
void nemo_experts_part(NemoWType tu, NemoWType td, const uint8_t* const* up, const uint8_t* const* down,
                       const float* s_up, const float* s_down, const int32_t* idx, const int32_t* xidx, int64_t n,
                       const float* lat, float* h, int64_t k, int64_t latent, int64_t ff, void* stream);
void nemo_experts_sum(const float* weights, const float* h, float* routed, int64_t k, int64_t latent, int64_t ff,
                      void* stream);

}  // namespace strata::kernels
