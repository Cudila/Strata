// include/strata/kernels/nemo_gemv.hpp - strata-nemotron: the decode GEMVs (one token, or a verify window of up to
// NEMO_GEMV_MAX_M tokens) for NVFP4 weights, bandwidth-bound on CUDA cores.
//
// nemo_gemm's pass kernel stages a 512-column tile per __syncthreads, so a lane has one 9-byte unit in flight and a
// one-token projection reads its weights at a third of the card's bandwidth; here the token rows are staged once per
// block, lanes take whole 36-byte blocks, two per step, and the blocks stride over the rows.
//
// Same contract as nemo_gemm (Y = X W^T * scale_2, `relu2`, `accumulate`), W in ggml block_nvfp4, cols % 64 == 0.
#pragma once

#include "strata/kernels/nemo_gemm.hpp"

#include <cstdint>

namespace strata::kernels {

constexpr int NEMO_GEMV_MAX_M = 8;

/// Whether nemo_gemv_nvfp4 takes this shape.  x rows wider than shared memory are read through L1: x and ldx must
/// then keep float4 alignment (ldx % 4 == 0).
bool nemo_gemv_nvfp4_ok(int64_t cols, int64_t M);

void nemo_gemv_nvfp4(const uint8_t* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
                     int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream);

/// The same for every nemo_gemm format (Q8_0, BF16 and F32 too; cols % 32 == 0).
bool nemo_gemv_ok(NemoWType t, int64_t cols, int64_t M);
void nemo_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
               int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream);

/// A verify window's projections on tensor cores (mma.sync, BF16 operands, FP32 sums): NVFP4, Q8_0, BF16 weights,
/// M <= 8 tokens.  x is rounded to BF16, Q8_0 weights too (NVFP4 is exact) - the prompt path's precision, not
/// nemo_gemv's FP32.  A token's row does not depend on M (each row's sum is its own, the same order for every M).
bool nemo_tc_gemv_ok(NemoWType t, int64_t cols, int64_t M);
void nemo_tc_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
                  int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream);

/// The tiled layout of a dense matrix (NVFP4, Q8_0, BF16; repacked once at load, see nemo_gemv.cu): its bytes, the
/// host-side repack from the GGUF rows, the tensor-core GEMV on it (M = 1..8, the precision of nemo_tc_gemv) and its
/// dequantization to row-major BF16 for the prompt path.
uint64_t nemo_tiled_bytes(NemoWType t, int64_t rows, int64_t cols);
bool nemo_tiled_repack(NemoWType t, const uint8_t* src, int64_t rows, int64_t cols, uint8_t* dst);
void nemo_tiled_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
                     int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream);
void nemo_tiled_dequant_bf16(NemoWType t, const void* w, uint16_t* out, int64_t rows, int64_t cols, void* stream);

/// nemo_gemv_nvfp4_many for any expert format (NVFP4, Q2_0, IQ4_NL, IQ4_XS, IQ3_XXS).
/// `xidx` (device, optional): matrix i reads input row xidx[i] (and still writes row idx[i]).
void nemo_gemv_many(NemoWType t, const uint8_t* const* w, const float* scale2, const float* x, int64_t x_stride,
                    float* y, int64_t y_stride, int64_t rows, int64_t cols, int64_t n, bool relu2, void* stream,
                    const int32_t* idx = nullptr, const int32_t* xidx = nullptr);

/// `n` matrices of one shape at once (a token's experts): y + i * y_stride (rows) = (x + i * x_stride) W_i^T *
/// scale2[i], W_i = w[i]; `w` and `scale2` are device arrays.  x_stride 0 = one input for all.  With `idx` (device,
/// n entries) matrix i reads input row idx[i] and writes output row idx[i] instead of row i.
void nemo_gemv_nvfp4_many(const uint8_t* const* w, const float* scale2, const float* x, int64_t x_stride, float* y,
                          int64_t y_stride, int64_t rows, int64_t cols, int64_t n, bool relu2, void* stream,
                          const int32_t* idx = nullptr);

}  // namespace strata::kernels
