// src/kernels/cuda/nemo_moe.cu - strata-nemotron stage 4: router and NVFP4 expert GEMVs.  See nemo_moe.hpp.
//
// The NVFP4 GEMV is a warp per output row; the unit of work is one 16-element sub-block (its UE4M3 scale and 8 bytes
// of nibbles), lanes taking consecutive units so a warp reads consecutive 36-byte blocks.  The input vector is staged
// in shared memory.  FP32 CUDA cores, bandwidth-bound: the decode baseline; the FP4 tensor-core path is for the
// prompt (stage 5).
#include "strata/kernels/nemo_moe.hpp"
#include "strata/kernels/nemo_gemv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace strata::kernels {
namespace {

__constant__ float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// UE4M3 (E4M3 with the sign bit clear): exponent bias 7, subnormals below exponent 1.
__device__ __forceinline__ float ue4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? ldexpf(1.0f + (float) m * 0.125f, (int) e - 7) : ldexpf((float) m * 0.125f, -6);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// Dot of one NVFP4 row with `xs` (shared), by one warp.  `row` points at the row's first block.
__device__ __forceinline__ float nvfp4_row_dot(const uint8_t* __restrict__ row, const float* __restrict__ xs,
                                               int cols, int lane) {
    const int units = cols / 16;
    float acc = 0.0f;
    for (int u = lane; u < units; u += 32) {
        const int sb = u >> 2, s = u & 3;
        const uint8_t* blk = row + (size_t) sb * 36;
        const float d = ue4m3(blk[s]);
        const uint32_t q0 = *reinterpret_cast<const uint32_t*>(blk + 4 + s * 8);
        const uint32_t q1 = *reinterpret_cast<const uint32_t*>(blk + 8 + s * 8);
        const float* xv = xs + sb * 64 + s * 16;
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t b0 = (q0 >> (8 * j)) & 0xffu, b1 = (q1 >> (8 * j)) & 0xffu;
            a += kE2M1[b0 & 15u] * xv[j] + kE2M1[b0 >> 4] * xv[j + 8];
            a += kE2M1[b1 & 15u] * xv[j + 4] + kE2M1[b1 >> 4] * xv[j + 12];
        }
        acc += d * a;
    }
    return warp_sum(acc);
}

constexpr int WARPS = 8;

__global__ void nvfp4_gemv_kernel(const uint8_t* __restrict__ w, float scale2, const float* __restrict__ x,
                                  float* __restrict__ y, int rows, int cols, int relu2) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= rows) return;
    float v = scale2 * nvfp4_row_dot(w + (size_t) r * (cols / 64) * 36, xs, cols, lane);
    if (relu2) {
        v = fmaxf(v, 0.0f);
        v *= v;
    }
    if (lane == 0) y[r] = v;
}

/// Grid (rows / WARPS, k): expert j = blockIdx.y.  `xin` is per expert (stride `ld_x`, 0 for a shared input).
__global__ void nvfp4_experts_kernel(const uint8_t* const* __restrict__ w, const float* __restrict__ s2,
                                     const float* __restrict__ xin, int64_t ld_x, float* __restrict__ y,
                                     int rows, int cols, int relu2) {
    extern __shared__ float xs[];
    const int j = blockIdx.y;
    const float* x = xin + (size_t) j * ld_x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= rows) return;
    float v = s2[j] * nvfp4_row_dot(w[j] + (size_t) r * (cols / 64) * 36, xs, cols, lane);
    if (relu2) {
        v = fmaxf(v, 0.0f);
        v *= v;
    }
    if (lane == 0) y[(size_t) j * rows + r] = v;
}

__global__ void weighted_sum_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                    float* __restrict__ out, int k, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += w[j] * parts[(size_t) j * n + i];   // expert order: deterministic
    out[i] = acc;
}

/// One block per token, one thread per expert; k rounds of a block-wide argmax (ties to the lower id).
__global__ void router_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                              int32_t* __restrict__ ids, float* __restrict__ weights, int n_expert, int k,
                              float scale, int norm) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float sel_score[32];
    const int t = blockIdx.x, e = threadIdx.x;
    const float lg = e < n_expert ? logits[(size_t) t * n_expert + e] : 0.0f;
    const float sc = 1.0f / (1.0f + expf(-lg));
    float choice = e < n_expert ? sc + bias[e] : -INFINITY;
    const int warp = e / 32, lane = e % 32, nwarps = blockDim.x / 32;
    for (int r = 0; r < k; ++r) {
        float v = choice;
        int id = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (lane == 0) { bv[warp] = v; bi[warp] = id; }
        __syncthreads();
        if (warp == 0) {
            v = lane < nwarps ? bv[lane] : -INFINITY;
            id = lane < nwarps ? bi[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, id, o);
                if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
            }
            if (lane == 0) bi[0] = id;
        }
        __syncthreads();
        const int win = bi[0];
        if (e == win) {
            ids[(size_t) t * k + r] = win;
            sel_score[r] = sc;
            choice = -INFINITY;
        }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.0f;
        for (int r = 0; r < k; ++r) sum += sel_score[r];
        const float f = norm ? scale / (sum + 1e-20f) : scale;
        for (int r = 0; r < k; ++r) weights[(size_t) t * k + r] = sel_score[r] * f;
    }
}

}  // namespace

void nemo_router(const float* logits, const float* bias, int32_t* ids, float* weights, int64_t n_tok,
                 int64_t n_expert, int64_t k, float scale, bool norm, void* stream) {
    if (n_tok <= 0 || n_expert > 1024 || k > 32) return;
    const int threads = (int) ((n_expert + 31) / 32 * 32);
    router_kernel<<<(unsigned) n_tok, threads, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights, (int) n_expert,
                                                                          (int) k, scale, norm ? 1 : 0);
}

void nvfp4_gemv(const uint8_t* w, float scale2, const float* x, float* y, int64_t rows, int64_t cols, bool relu2,
                void* stream) {
    const unsigned blocks = (unsigned) ((rows + WARPS - 1) / WARPS);
    nvfp4_gemv_kernel<<<blocks, WARPS * 32, (size_t) cols * sizeof(float), (cudaStream_t) stream>>>(
        w, scale2, x, y, (int) rows, (int) cols, relu2 ? 1 : 0);
}

void nemo_experts(const uint8_t* const* up, const uint8_t* const* down, const float* s_up, const float* s_down,
                  const float* weights, const float* lat, float* h, float* routed, int64_t k, int64_t latent,
                  int64_t ff, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    float* parts = h + (size_t) k * ff;
    nemo_gemv_nvfp4_many(up, s_up, lat, 0, h, ff, ff, latent, k, true, st);
    nemo_gemv_nvfp4_many(down, s_down, h, ff, parts, latent, latent, ff, k, false, st);
    weighted_sum_kernel<<<(unsigned) ((latent + 255) / 256), 256, 0, st>>>(parts, weights, routed, (int) k,
                                                                           (int) latent);
}

}  // namespace strata::kernels

namespace strata::kernels {

void nemo_experts_part(NemoWType tu, NemoWType td, const uint8_t* const* up, const uint8_t* const* down,
                       const float* s_up, const float* s_down, const int32_t* idx, const int32_t* xidx, int64_t n,
                       const float* lat, float* h, int64_t k, int64_t latent, int64_t ff, void* stream) {
    float* parts = h + (size_t) k * ff;
    nemo_gemv_many(tu, up, s_up, lat, xidx ? latent : 0, h, ff, ff, latent, n, true, stream, idx, xidx);
    nemo_gemv_many(td, down, s_down, h, ff, parts, latent, latent, ff, n, false, stream, idx);
}

void nemo_experts_sum(const float* weights, const float* h, float* routed, int64_t k, int64_t latent, int64_t ff,
                      void* stream) {
    const float* parts = h + (size_t) k * ff;
    weighted_sum_kernel<<<(unsigned) ((latent + 255) / 256), 256, 0, (cudaStream_t) stream>>>(parts, weights, routed,
                                                                                             (int) k, (int) latent);
}

}  // namespace strata::kernels
