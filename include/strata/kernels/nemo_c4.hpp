// include/strata/kernels/nemo_c4.hpp - strata-nemotron: NVFP4 matrices in a column layout ("C4") for inputs that are
// mostly exact zeros - the experts' down_proj, whose input is relu^2's output (~92% zeros in decode).
//
// The same codes and scales as ggml's block_nvfp4 rows, permuted (lossless, the same size): for an R x K matrix,
//   codes:  K columns of R / 2 bytes - column k, row r is byte k * R/2 + r/2, the low nibble for an even r;
//   scales: K / 16 block columns of R bytes - the UE4M3 scale of row r's 16 inputs 16c .. 16c + 15 is byte c * R + r.
// A product with a sparse input reads the codes of its nonzero inputs only (R / 2 bytes each) and the scales of the
// 16-input blocks that hold one (R bytes each); a row's sum runs over its nonzero inputs in increasing order.
// Shapes: R % 128 == 0, K % 16 == 0, K <= 4096.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Bytes of an R x K matrix in C4 (the same as in block_nvfp4 rows).
inline uint64_t nemo_c4_bytes(int64_t rows, int64_t cols) { return (uint64_t) rows * cols / 2 + (uint64_t) rows * cols / 16; }

/// ggml block_nvfp4 rows (device) -> C4 (device).
void nemo_c4_from_ggml(const uint8_t* ggml, int64_t rows, int64_t cols, uint8_t* c4, void* stream);

/// For n input rows x_j = x + j * x_stride (cols floats): the nonzero columns in increasing order (list + j * cols),
/// their values (vals + j * cols) and how many (count[j]).
void nemo_c4_compact(const float* x, int64_t x_stride, int64_t cols, int64_t n, uint16_t* list, float* vals,
                     int* count, void* stream);

/// y[idx[j]] (rows floats, row stride y_stride) = scale2[j] * W_j . x_j for n matrices W_j in C4 (pointers w[j]),
/// the inputs given by nemo_c4_compact's lists; `part` is scratch of nemo_c4_part_floats(rows, n) floats.
/// rows == 1024 (the experts' down_proj).
void nemo_c4_gemv_many(const uint8_t* const* w, const float* scale2, const uint16_t* list, const float* vals,
                       const int* count, float* part, float* y, int64_t y_stride, int64_t rows, int64_t cols, int64_t n,
                       const int32_t* idx, void* stream);
int64_t nemo_c4_part_floats(int64_t rows, int64_t n);

}  // namespace strata::kernels
