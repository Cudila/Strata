// src/kernels/cuda/nemo_tc.cu - see include/strata/kernels/nemo_tc.hpp.
//
// Row-major Y (M, rows) = X (M, cols) W^T is, read column-major, Y^T (rows, M) = W (rows, cols) X^T: cuBLAS's
// C = op(A) op(B) with A = W stored as a column-major (cols, rows) matrix (transposed: CUBLAS_OP_T, lda = cols),
// B = X as a column-major (cols, M) matrix (CUBLAS_OP_N, ldb = cols), C = Y (ldc = ldy).
#include "strata/kernels/nemo_tc.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

__constant__ float kE2M1t[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float ue4m3t(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? ldexpf(1.0f + (float) m * 0.125f, (int) e - 7) : ldexpf((float) m * 0.125f, -6);
}

__device__ __forceinline__ uint16_t bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

/// One thread per 8 values of matrix blockIdx.y: half h of a 16-value NVFP4 sub-block (h = 0: the low nibbles of its
/// 8 bytes, values 0..7; h = 1: the high ones, 8..15), one 16-byte store - consecutive threads write consecutive bytes.
__global__ void dequant_nvfp4_kernel(const uint8_t* const* __restrict__ ptrs, const uint8_t* __restrict__ one,
                                     uint16_t* __restrict__ out, int64_t rows, int64_t cols) {
    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = kE2M1t[threadIdx.x];
    __syncthreads();
    const uint8_t* w = ptrs ? ptrs[blockIdx.y] : one;
    uint4* o = reinterpret_cast<uint4*>(out + (size_t) blockIdx.y * rows * cols);
    const int64_t per_row = cols / 8;
    const int64_t units = rows * per_row;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = u / per_row, c8 = u % per_row;   // c8: the 8-value group in the row
        const uint8_t* blk = w + (size_t) r * (cols / 64) * 36 + (size_t) (c8 / 8) * 36;
        const int s = (int) ((c8 >> 1) & 3), h = (int) (c8 & 1);
        const float d = ue4m3t(blk[s]);
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(blk + 4 + s * 8);   // 4-byte aligned (36-byte blocks)
        const uint32_t q0 = qp[0] >> (4 * h), q1 = qp[1] >> (4 * h);
        uint32_t pk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t qa = j < 2 ? q0 : q1, sh = 16 * (j & 1);
            const uint32_t lo = bf16_rne(d * tab[(qa >> sh) & 15u]);          // exact: <= 4 significant bits
            const uint32_t hi = bf16_rne(d * tab[(qa >> (sh + 8)) & 15u]);
            pk[j] = lo | (hi << 16);
        }
        o[u] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
}

/// One thread per 32-value Q8_0 block.
__global__ void dequant_q8_kernel(const uint8_t* __restrict__ w, uint16_t* __restrict__ out, int64_t rows,
                                  int64_t cols) {
    const int64_t units = rows * (cols / 32);
    const int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= units) return;
    const int64_t r = u / (cols / 32), c0 = (u % (cols / 32)) * 32;
    const uint8_t* blk = w + (size_t) r * (cols / 32) * 34 + (size_t) (c0 / 32) * 34;
    const float d = __half2float(__ushort_as_half((unsigned short) (blk[0] | (blk[1] << 8))));
    uint16_t* dst = out + (size_t) r * cols + c0;
    for (int j = 0; j < 32; ++j) dst[j] = bf16_rne(d * (float) (int8_t) blk[2 + j]);
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, int64_t ldx, uint16_t* __restrict__ out,
                                   int64_t cols, int relu2) {
    const int64_t m = blockIdx.y;
    for (int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += (int64_t) gridDim.x * blockDim.x) {
        float v = x[m * ldx + c];
        if (relu2) {
            v = fmaxf(v, 0.0f);
            v *= v;
        }
        out[m * cols + c] = bf16_rne(v);
    }
}

__global__ void relu2_kernel(float* __restrict__ y, int64_t ldy, int64_t cols) {
    const int64_t m = blockIdx.y;
    for (int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += (int64_t) gridDim.x * blockDim.x) {
        const float v = fmaxf(y[m * ldy + c], 0.0f);
        y[m * ldy + c] = v * v;
    }
}

__device__ __forceinline__ float e2m1_rne(float a) {   // |a| to the nearest e2m1 magnitude (ties to even)
    if (a <= 0.25f) return 0.0f;
    if (a < 0.75f) return 0.5f;
    if (a <= 1.25f) return 1.0f;
    if (a < 1.75f) return 1.5f;
    if (a <= 2.5f) return 2.0f;
    if (a < 3.5f) return 3.0f;
    if (a <= 5.0f) return 4.0f;
    return 6.0f;
}

/// One block per row: the row's NVFP4 quantization (per-row global scale amax / (448 * 6), per 16 values an e4m3
/// scale, e2m1 values), written back as the BF16 values it stands for.
__global__ void fq_nvfp4_kernel(uint16_t* __restrict__ x, int64_t cols) {
    uint16_t* r = x + (size_t) blockIdx.x * cols;
    __shared__ float red[32];
    float m = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(__uint_as_float((uint32_t) r[c] << 16)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float gs = red[0] / (448.0f * 6.0f);
    if (gs == 0.0f) return;
    for (int64_t b = threadIdx.x; b < cols / 16; b += blockDim.x) {
        float v[16], am = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = __uint_as_float((uint32_t) r[b * 16 + j] << 16);
            am = fmaxf(am, fabsf(v[j]));
        }
        const __nv_fp8_e4m3 s8(am / (6.0f * gs));
        const float sb = (float) s8 * gs;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float q = sb > 0.0f ? copysignf(e2m1_rne(fabsf(v[j]) / sb), v[j]) * sb : 0.0f;
            r[b * 16 + j] = bf16_rne(q);
        }
    }
}

std::mutex g_mu;
cublasHandle_t g_handle = nullptr;

}  // namespace

void nemo_dequant_bf16(NemoWType t, const void* w, uint16_t* out, int64_t rows, int64_t cols, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    if (t == NemoWType::NVFP4) {
        const int64_t units = rows * (cols / 8);
        dequant_nvfp4_kernel<<<dim3((unsigned) std::min<int64_t>((units + 255) / 256, 4096), 1), 256, 0, st>>>(
            nullptr, (const uint8_t*) w, out, rows, cols);
    } else if (t == NemoWType::Q8_0) {
        const int64_t units = rows * (cols / 32);
        dequant_q8_kernel<<<(unsigned) ((units + 255) / 256), 256, 0, st>>>((const uint8_t*) w, out, rows, cols);
    } else if (t == NemoWType::BF16) {
        cudaMemcpyAsync(out, w, (size_t) rows * cols * 2, cudaMemcpyDeviceToDevice, st);
    }
}

void nemo_dequant_bf16_many(const uint8_t* const* ptr, uint16_t* out, int64_t n, int64_t rows, int64_t cols,
                            void* stream) {
    if (n <= 0) return;
    const int64_t units = rows * (cols / 8);
    dequant_nvfp4_kernel<<<dim3((unsigned) std::min<int64_t>((units + 255) / 256, 512), (unsigned) n), 256, 0,
                           (cudaStream_t) stream>>>(ptr, nullptr, out, rows, cols);
}

void nemo_f32_to_bf16(const float* x, int64_t ldx, uint16_t* out, int64_t M, int64_t cols, bool relu2, void* stream) {
    if (M <= 0) return;
    const unsigned bx = (unsigned) ((cols + 255) / 256 < 64 ? (cols + 255) / 256 : 64);
    f32_to_bf16_kernel<<<dim3(bx, (unsigned) M), 256, 0, (cudaStream_t) stream>>>(x, ldx, out, cols, relu2 ? 1 : 0);
}

void nemo_relu2_rows(float* y, int64_t ldy, int64_t M, int64_t cols, void* stream) {
    if (M <= 0) return;
    const unsigned bx = (unsigned) ((cols + 255) / 256 < 64 ? (cols + 255) / 256 : 64);
    relu2_kernel<<<dim3(bx, (unsigned) M), 256, 0, (cudaStream_t) stream>>>(y, ldy, cols);
}

void nemo_fq_nvfp4_bf16(uint16_t* x, int64_t M, int64_t cols, void* stream) {
    if (M <= 0) return;
    fq_nvfp4_kernel<<<(unsigned) M, 256, 0, (cudaStream_t) stream>>>(x, cols);
}

void* nemo_cublas() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_handle) {
        if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return nullptr;
        cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH);
    }
    return g_handle;
}

bool nemo_gemm_bf16(const uint16_t* x, const uint16_t* w, float alpha, float* y, int64_t ldy, int64_t rows,
                    int64_t cols, int64_t M, bool accumulate, void* stream) {
    cublasHandle_t h = (cublasHandle_t) nemo_cublas();
    if (!h) return false;
    cublasSetStream(h, (cudaStream_t) stream);
    const float beta = accumulate ? 1.0f : 0.0f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) rows, (int) M, (int) cols, &alpha, w, CUDA_R_16BF, (int) cols,
                        x, CUDA_R_16BF, (int) cols, &beta, y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
}

bool nemo_gemm_f32(const float* x, int64_t ldx, const float* w, float* y, int64_t ldy, int64_t rows, int64_t cols,
                   int64_t M, bool accumulate, void* stream) {
    cublasHandle_t h = (cublasHandle_t) nemo_cublas();
    if (!h) return false;
    cublasSetStream(h, (cudaStream_t) stream);
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    // STRATA_NEMO_ROUTER_TF32=1: TF32 tensor cores (a probe of what the router's precision costs)
    static const bool tf32 = std::getenv("STRATA_NEMO_ROUTER_TF32") && std::atoi(std::getenv("STRATA_NEMO_ROUTER_TF32"));
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) rows, (int) M, (int) cols, &alpha, w, CUDA_R_32F, (int) cols, x,
                        CUDA_R_32F, (int) ldx, &beta, y, CUDA_R_32F, (int) ldy,
                        tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
}

bool nemo_gemm_bf16_grouped(const uint16_t* const* x, const uint16_t* const* w, float* const* y, const int* m,
                            const float* alpha, int n, int64_t rows, int64_t cols, void* stream) {
    // one cublasGemmEx per group member: cublasGemmGroupedBatchedEx refuses BF16 inputs with an FP32 output
    // (CUBLAS_STATUS_NOT_SUPPORTED, cuBLAS of CUDA 13.3), and the FP32 output is what keeps the experts' sums exact
    // to their accumulation
    for (int i = 0; i < n; ++i)
        if (m[i] > 0 && !nemo_gemm_bf16(x[i], w[i], alpha[i], y[i], rows, rows, cols, m[i], false, stream)) return false;
    return true;
}

}  // namespace strata::kernels
