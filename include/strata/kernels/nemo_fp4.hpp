// include/strata/kernels/nemo_fp4.hpp - strata-nemotron: the prompt's NVFP4 GEMMs on sm_120's FP4 tensor cores (W4A4).
//
// mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 multiplies e2m1 by e2m1 with a ue4m3 scale per 16
// values of each operand - NVFP4 exactly - at four times the BF16 rate.  The weights already are NVFP4; the
// activations are quantized to it per row: a global scale amax / (448 * 6) (FP32, applied in the epilogue with the
// weight's scale_2), an e4m3 scale per 16 values, e2m1 values rounded to nearest-even.  The products and their FP32
// sums are exact for the operands; what the path rounds is the activations (measured with STRATA_NEMO_A4: PPL x1.006
// for every NVFP4 GEMM of the prompt, the decode path untouched).
//
// Both operands go to the kernel in one layout, "N4": rows of K/2 bytes with the values in natural order (value k in
// byte k/2, the low nibble for even k), and the scales as rows of K/16 bytes.  Weights are converted into it per call
// from their stored layouts (a quarter of what the BF16 expansion wrote).
//
// Shapes: K % 64 == 0, N % 128 == 0 (every NVFP4 matrix of the model); any M.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Whether the FP4 path is compiled in (sm_120a) and the device runs it.
bool nemo_fp4_ok();

/// x rows (f32, row r = x + (gather ? gather[r] : r) * ldx; relu(x)^2 first when `relu2`) -> N4: q (M, K/2),
/// s (M, K/16), gs (M) the rows' global scales.
void nemo_fp4_quant_rows(const float* x, int64_t ldx, const int32_t* gather, int64_t M, int64_t K, bool relu2,
                         uint8_t* q, uint8_t* s, float* gs, void* stream);

/// n ggml block_nvfp4 matrices (ptr[i], rows x cols, device array of pointers) -> N4 at q + i * rows * cols / 2,
/// s + i * rows * cols / 16.
void nemo_fp4_from_ggml_many(const uint8_t* const* ptr, int64_t n, int64_t rows, int64_t cols, uint8_t* q, uint8_t* s,
                             void* stream);

/// The tiled NVFP4 layout (nemo_gemv's) -> N4.
void nemo_fp4_from_tiled(const uint8_t* w, int64_t rows, int64_t cols, uint8_t* q, uint8_t* s, void* stream);

/// Y (M, ldy) = alpha * gs_a[r] * A_r . B_n (+ Y when `accumulate`; relu(.)^2 of the result when `relu2`).
void nemo_fp4_gemm(const uint8_t* aq, const uint8_t* as, const float* ags, const uint8_t* bq, const uint8_t* bs,
                   float alpha, float* y, int64_t ldy, int64_t M, int64_t N, int64_t K, bool relu2, bool accumulate,
                   void* stream);

/// A group of problems sharing N and K (a group of experts): tile i = (problem p, rows [r0, r1)) of tiles (device,
/// 3 ints a tile, r1 - r0 <= 128), rows of the packed A and of Y (ld N); problem p's B at bq + p * N * K / 2,
/// bs + p * N * K / 16, its alpha at alpha[p] (device).
void nemo_fp4_gemm_grouped(const uint8_t* aq, const uint8_t* as, const float* ags, const uint8_t* bq,
                           const uint8_t* bs, const float* alpha, const int32_t* tiles, int ntiles, float* y,
                           int64_t N, int64_t K, bool relu2, void* stream, uint16_t* yb = nullptr,
                           const int32_t* yrow = nullptr, const uint8_t* const* bptr = nullptr,
                           const int32_t* arow = nullptr);
/// (yb: Y as BF16 instead, y unused - the up projection's relu^2 output, read again only to be quantized; yrow
/// (device, a row per packed A row): the output row of each, the down projection's (t, j) slot; bptr (device, a
/// pointer per problem): B in ggml block_nvfp4 rows instead of bq/bs - no nemo_fp4_from_ggml_many needed; arow
/// (device, a row per tile row): the A row (and global scale) each reads - one quantized latent row per token for all
/// of its experts)

/// nemo_fp4_quant_rows for BF16 rows (ldx % 8 == 0, 16-byte aligned, K <= 12,288; false: not taken).
bool nemo_fp4_quant_rows_bf16(const uint16_t* x, int64_t ldx, int64_t M, int64_t K, uint8_t* q, uint8_t* s, float* gs,
                              void* stream);

}  // namespace strata::kernels
