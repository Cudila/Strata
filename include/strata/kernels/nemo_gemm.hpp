// include/strata/kernels/nemo_gemm.hpp - strata-nemotron: Y = X W^T for M tokens at once, W in a GGUF format,
// X FP32 - the dense projections of the verify window and the prompt (and of one token: M = 1).
//
// Weight-stationary on CUDA cores: a warp owns an output row, its lanes decode a tile of the row once per pass and
// multiply it with up to MT tokens staged in shared memory, so W is read ceil(M / MT) times instead of M.  The
// arithmetic per (row, token) is a sum of per-lane partial dots and one warp reduction - the same for every M, so a
// token's result does not depend on how many tokens share its pass.  Tensor cores for the prompt come in stage 5.
//
//   NVFP4  ggml block_nvfp4 (64 values, 36 bytes), the tensor's FP32 scale_2 applied to the dot   cols % 64 == 0
//   Q8_0   ggml block_q8_0 (32 values: f16 d, 32 int8)                                        cols % 32 == 0
//   BF16   plain row-major                                                                    cols % 16 == 0
//   F32    plain row-major (the converter keeps the router's weight in F32)                    cols % 16 == 0
//
// X is (M, ldx) f32, Y is (M, ldy) f32 (rows of each are tokens); W's rows are contiguous.  `relu2` writes
// relu(.)^2.  `accumulate` adds into Y instead of storing (a residual add fused into the projection).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The ggml type ids.  Q2_0 / IQ4_NL / IQ4_XS / IQ3_XXS are expert formats only (nemo_gemv, nemo_gemv_many and the
/// prompt's dequantization; nemo_gemm itself has the first four).
enum class NemoWType : int { NVFP4 = 40, Q8_0 = 8, BF16 = 30, F32 = 0, Q2_0 = 42, IQ4_NL = 20, IQ4_XS = 23, IQ3_XXS = 18 };

/// Values per block of `t` (a row's length is a multiple of it).
int64_t nemo_block_values(NemoWType t);

/// Bytes of one row of `cols` values in type `t`.
uint64_t nemo_row_bytes(NemoWType t, int64_t cols);

void nemo_gemm(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
               int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream);

}  // namespace strata::kernels
