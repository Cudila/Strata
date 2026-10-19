// include/strata/kernels/nemo_tc.hpp - strata-nemotron: the prompt's projections on tensor cores (cuBLAS BF16).
//
// A chunk of the prompt multiplies every dense matrix by thousands of rows, which nemo_gemm's CUDA-core passes do
// at a few TFLOP/s; here the weight is expanded to BF16 once per call and cuBLAS does the product (FP32 accumulate).
//
// THE EXPANSION IS EXACT FOR NVFP4: a value is e2m1 (1 mantissa bit) x ue4m3 (3 bits), at most 4 significant bits and
// inside BF16's range, so BF16 holds it exactly; the tensor's scale_2 is NOT multiplied in (it is ~1e-5 and would push
// small weights towards BF16's subnormals) but applied as the GEMM's alpha.  Q8_0 (f16 d x int8, up to 17 bits) is
// rounded to BF16 (2^-9 relative): only the attention projections are Q8_0 in the packs this reads.  What the path
// does round is the activations, to BF16 - the precision every tensor-core W4A16 kernel has.
//
// The experts of a chunk go in groups through cublasGemmGroupedBatchedEx: each expert of the group its own m (its
// tokens), all of them one call, its scale_2 as its alpha.
#pragma once

#include "strata/kernels/nemo_gemm.hpp"

#include <cstdint>

namespace strata::kernels {

/// W (rows, cols) in `t` -> BF16 (rows, cols), without NVFP4's scale_2.
void nemo_dequant_bf16(NemoWType t, const void* w, uint16_t* out, int64_t rows, int64_t cols, void* stream);

/// `n` matrices at once (the experts of a group): ptr[i] (rows, cols) NVFP4 -> out + i * rows * cols.
void nemo_dequant_bf16_many(const uint8_t* const* ptr, uint16_t* out, int64_t n, int64_t rows, int64_t cols,
                            void* stream);

/// The same for any expert format (nemo_gemv.cu holds the decoders): NVFP4 without scale_2 (exact), the ggml formats
/// with their block scales (rounded to BF16).
void nemo_dequant_bf16_many_t(NemoWType t, const uint8_t* const* ptr, uint16_t* out, int64_t n, int64_t rows,
                              int64_t cols, void* stream);

/// x (M, ldx) f32 -> out (M, cols) BF16; `relu2` takes relu(x)^2 first.
void nemo_f32_to_bf16(const float* x, int64_t ldx, uint16_t* out, int64_t M, int64_t cols, bool relu2, void* stream);

/// x (M, cols) BF16, in place: each row through NVFP4 (per-row global scale, e4m3 per 16 values, e2m1) and back -
/// the W4A4 path's activation rounding, for measuring what it costs.
void nemo_fq_nvfp4_bf16(uint16_t* x, int64_t M, int64_t cols, void* stream);

/// y (M, ldy) f32 = relu(y)^2, in place.
void nemo_relu2_rows(float* y, int64_t ldy, int64_t M, int64_t cols, void* stream);

/// The cuBLAS handle these use (one per process, on `stream` at each call).
void* nemo_cublas();

/// Y = alpha * X W^T (+ Y when `accumulate`): X (M, cols) BF16 row-major, W (rows, cols) BF16 row-major, Y (M, ldy) f32.
bool nemo_gemm_bf16(const uint16_t* x, const uint16_t* w, float alpha, float* y, int64_t ldy, int64_t rows,
                    int64_t cols, int64_t M, bool accumulate, void* stream);

/// The same in FP32 (cuBLAS SGEMM, no reduced-precision math): X (M, ldx) f32, W (rows, cols) f32 - the router.
bool nemo_gemm_f32(const float* x, int64_t ldx, const float* w, float* y, int64_t ldy, int64_t rows, int64_t cols,
                   int64_t M, bool accumulate, void* stream);

/// The grouped form: for i < n, Y_i (m[i], rows) = alpha[i] * X_i (m[i], cols) W_i^T, with X_i, W_i and Y_i given by
/// host pointer arrays (device addresses).  Every group has the same rows and cols.
bool nemo_gemm_bf16_grouped(const uint16_t* const* x, const uint16_t* const* w, float* const* y, const int* m,
                            const float* alpha, int n, int64_t rows, int64_t cols, void* stream);

}  // namespace strata::kernels
