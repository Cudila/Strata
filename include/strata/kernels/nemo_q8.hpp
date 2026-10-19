// include/strata/kernels/nemo_q8.hpp - strata-nemotron: the prompt's Q8_0 and BF16 GEMMs on int8 tensor cores (W8A8).
//
// The attention projections are Q8_0 (int8, an FP16 scale per 32 values) and the latent fc1/fc2 BF16; on the prompt
// they went through a BF16 expansion and cuBLAS (~160 TFLOP/s on the 5090).  Here both operands are int8 with an
// FP32 scale per 32 values - the activations quantized per row and block as llama.cpp's MMQ does (q8_1), the Q8_0
// weights taken as they are, the BF16 weights rounded to Q8_0 - and mma.sync m16n8k32 s8 multiplies them exactly;
// each block's int32 sum is scaled into the FP32 accumulator.
//
// Layout of both operands ("Q8N"): rows of K int8 values, the scales as rows of K/32 floats.
// Shapes: K % 128 == 0, N % 128 == 0; any M.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// x rows (f32, row r at x + r * ldx) -> q (M, K) int8, s (M, K/32) scales (amax / 127 per 32 values).
void nemo_q8_quant_rows(const float* x, int64_t ldx, int64_t M, int64_t K, int8_t* q, float* s, void* stream);

/// The tiled Q8_0 layout (nemo_gemv's) -> Q8N, exact.
void nemo_q8_from_tiled_q8(const uint8_t* w, int64_t rows, int64_t cols, int8_t* q, float* s, void* stream);

/// The tiled BF16 layout -> Q8N, each 32 values rounded to Q8_0 (amax / 127, nearest).
void nemo_q8_from_tiled_bf16(const uint8_t* w, int64_t rows, int64_t cols, int8_t* q, float* s, void* stream);

/// Y (M, ldy) = alpha * A . B^T (+ Y when `accumulate`): A (M, K), B (N, K) in Q8N.
void nemo_q8_gemm(const int8_t* aq, const float* as, const int8_t* bq, const float* bs, float alpha, float* y,
                  int64_t ldy, int64_t M, int64_t N, int64_t K, bool accumulate, void* stream);

}  // namespace strata::kernels
