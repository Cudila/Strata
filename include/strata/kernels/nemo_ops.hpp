// include/strata/kernels/nemo_ops.hpp - strata-nemotron: the small row operations between the model's kernels.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * w, for T rows of n.  `out` may not alias `x`.
void nemo_rms_norm_rows(const float* x, const float* w, float* out, int64_t T, int64_t n, float eps, void* stream);

/// out (T, n) f32 from BF16 rows (T, n).
void nemo_bf16_rows_to_f32(const uint16_t* in, float* out, int64_t T, int64_t n, void* stream);

/// out[i] = in[idx[i]] for `count` rows of n.
void nemo_gather_rows(const float* in, const int32_t* idx, float* out, int64_t count, int64_t n, void* stream);

/// out[idx[i]] = in[i] for `count` rows of n.
void nemo_scatter_rows(const float* in, const int32_t* idx, float* out, int64_t count, int64_t n, void* stream);

/// out[t] = sum_j w[t*k + j] * parts[(t*k + j)] for T rows of n, j in order (deterministic).
void nemo_weighted_rows(const float* parts, const float* w, float* out, int64_t T, int64_t k, int64_t n,
                        void* stream);
/// The same for BF16 parts (n % 8 == 0): the prompt MoE's expert outputs, each in its (t, j) row.
void nemo_weighted_rows_bf16(const uint16_t* parts, const float* w, float* out, int64_t T, int64_t k, int64_t n,
                             void* stream);

/// routed[seg_token[s]] += sum over the segment's rows r, in their order, of w[wid[r]] * out[row[r]] (rows of n).
/// One block per segment; a segment's rows are one token's rows of a group (the prompt's MoE, group by group).
void nemo_accum_rows(const float* out, const int32_t* seg_start, const int32_t* seg_token, const int32_t* row,
                     const int32_t* wid, const float* w, float* routed, int64_t nseg, int64_t n, void* stream);

/// The same into a double accumulator: the products are exact in double and a token's 22 sums do not depend on
/// their order (which groups a token's experts fall in depends on what the cache holds) - so the prompt's result does
/// not depend on the cache.
void nemo_accum_rows_f64(const float* out, const int32_t* seg_start, const int32_t* seg_token, const int32_t* row,
                         const int32_t* wid, const float* w, double* routed, int64_t nseg, int64_t n, void* stream);
void nemo_f64_to_f32(const double* x, float* y, int64_t n, void* stream);

/// LayerNorm without bias (the MTP head's): out[t] = (x[t] - mean) * rsqrt(var + eps) * w.
void nemo_layer_norm_rows(const float* x, const float* w, float* out, int64_t T, int64_t n, float eps, void* stream);

/// out[t] = the first index of row t's largest value (T rows of n); with `prob`, also its softmax probability.
void nemo_argmax_rows(const float* x, int64_t T, int64_t n, int32_t* out, void* stream, float* prob = nullptr);

/// For k routed ids (device) of experts resident at base + e * stride (up, then down at + ub): ptr[j] / ptr[k + j]
/// and s2[j] / s2[k + j] (their scale_2) - nemo_experts_part's inputs without a trip to the host.
void nemo_resident_expert_ptrs(const int32_t* ids, const uint8_t* base, uint64_t stride, uint64_t ub,
                               const float* s_up, const float* s_dn, const uint8_t** ptr, float* s2, int64_t k,
                               void* stream);

}  // namespace strata::kernels
