// src/kernels/cuda/nemo_gemm.cu - see include/strata/kernels/nemo_gemm.hpp.
//
// A pass: block = 8 warps = 8 output rows, MT tokens.  The columns go in tiles of 32 lanes x U values; a lane decodes
// its U weights of the tile once and multiplies them with the MT tokens' values, which the block staged in shared
// memory LANE-INTERLEAVED (value j of lane l at xs[m][j*32 + l]), so the 32 lanes of a warp read 32 consecutive words.
#include "strata/kernels/nemo_gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {
namespace {

__constant__ float kE2M1g[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float ue4m3f(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? ldexpf(1.0f + (float) m * 0.125f, (int) e - 7) : ldexpf((float) m * 0.125f, -6);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// NVFP4: unit = one 16-value sub-block; column c0 is a multiple of 16.
struct DecNvfp4 {
    static constexpr int U = 16;
    __device__ static void unit(const uint8_t* row, int c0, float* w) {
        const uint8_t* blk = row + (size_t) (c0 >> 6) * 36;
        const int s = (c0 >> 4) & 3;
        const float d = ue4m3f(blk[s]);
        const uint32_t q0 = *reinterpret_cast<const uint32_t*>(blk + 4 + s * 8);
        const uint32_t q1 = *reinterpret_cast<const uint32_t*>(blk + 8 + s * 8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t b0 = (q0 >> (8 * j)) & 0xffu, b1 = (q1 >> (8 * j)) & 0xffu;
            w[j] = d * kE2M1g[b0 & 15u];
            w[j + 8] = d * kE2M1g[b0 >> 4];
            w[j + 4] = d * kE2M1g[b1 & 15u];
            w[j + 12] = d * kE2M1g[b1 >> 4];
        }
    }
};

/// Q8_0: unit = one 32-value block (34 bytes: f16 d, 32 int8); c0 is a multiple of 32.
struct DecQ8 {
    static constexpr int U = 32;
    __device__ static void unit(const uint8_t* row, int c0, float* w) {
        const uint8_t* blk = row + (size_t) (c0 >> 5) * 34;
        const float d = __half2float(__ushort_as_half((unsigned short) (blk[0] | (blk[1] << 8))));
        const uint16_t* q = reinterpret_cast<const uint16_t*>(blk + 2);   // 2-byte aligned: 34 is even
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint16_t p = q[j];
            w[2 * j] = d * (float) (int8_t) (p & 0xff);
            w[2 * j + 1] = d * (float) (int8_t) (p >> 8);
        }
    }
};

/// BF16: unit = 16 values (32 bytes, 16-byte aligned when cols % 8 == 0).
struct DecBf16 {
    static constexpr int U = 16;
    __device__ static void unit(const uint8_t* row, int c0, float* w) {
        const uint4* p = reinterpret_cast<const uint4*>(row + (size_t) c0 * 2);
        const uint4 a = p[0], b = p[1];
        const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[2 * j] = __uint_as_float(v[j] << 16);
            w[2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u);
        }
    }
};

/// F32: unit = 16 values (64 bytes).
struct DecF32 {
    static constexpr int U = 16;
    __device__ static void unit(const uint8_t* row, int c0, float* w) {
        const float4* p = reinterpret_cast<const float4*>(row + (size_t) c0 * 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 a = p[j];
            w[4 * j] = a.x; w[4 * j + 1] = a.y; w[4 * j + 2] = a.z; w[4 * j + 3] = a.w;
        }
    }
};

constexpr int WARPS = 8;

template <class Dec, int MT>
__global__ void gemm_kernel(const uint8_t* __restrict__ w, uint64_t row_bytes, float scale2,
                            const float* __restrict__ x, int64_t ldx, float* __restrict__ y, int64_t ldy, int rows,
                            int cols, int M, int relu2, int accumulate) {
    constexpr int U = Dec::U, TILE = 32 * U;
    extern __shared__ float xs[];   // [MT][TILE], lane-interleaved
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * WARPS + warp;
    const int m0 = blockIdx.y * MT;
    const int mt = min(MT, M - m0);
    const uint8_t* row = w + (size_t) (r < rows ? r : 0) * row_bytes;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int c0 = 0; c0 < cols; c0 += TILE) {
        const int tc = min(TILE, cols - c0);
        for (int i = threadIdx.x; i < MT * TILE; i += blockDim.x) {
            const int m = i / TILE, c = i % TILE;
            float v = 0.0f;
            if (m < mt && c < tc) v = x[(size_t) (m0 + m) * ldx + c0 + c];
            xs[m * TILE + (c % U) * 32 + c / U] = v;
        }
        __syncthreads();
        if (r < rows && lane * U < tc) {
            float wv[U];
            Dec::unit(row, c0 + lane * U, wv);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < mt) {
                    float a = 0.0f;
#pragma unroll
                    for (int j = 0; j < U; ++j) a += wv[j] * xs[m * TILE + j * 32 + lane];
                    acc[m] += a;
                }
            }
        }
        __syncthreads();
    }
    if (r >= rows) return;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= mt) break;
        float v = warp_sum(acc[m]) * scale2;
        if (relu2) {
            v = fmaxf(v, 0.0f);
            v *= v;
        }
        if (lane == 0) {
            float* out = y + (size_t) (m0 + m) * ldy + r;
            *out = accumulate ? *out + v : v;
        }
    }
}

template <class Dec, int MT>
void launch(const void* w, uint64_t rb, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
            int64_t cols, int64_t M, bool relu2, bool acc, cudaStream_t st) {
    const size_t smem = (size_t) MT * 32 * Dec::U * sizeof(float);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(gemm_kernel<Dec, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr = true;
    }
    const dim3 grid((unsigned) ((rows + WARPS - 1) / WARPS), (unsigned) ((M + MT - 1) / MT));
    gemm_kernel<Dec, MT><<<grid, WARPS * 32, smem, st>>>((const uint8_t*) w, rb, s2, x, ldx, y, ldy, (int) rows,
                                                         (int) cols, (int) M, relu2 ? 1 : 0, acc ? 1 : 0);
}

template <class Dec, int MTMAX>
void dispatch(const void* w, uint64_t rb, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
              int64_t cols, int64_t M, bool relu2, bool acc, cudaStream_t st) {
    // the pass width follows M so a one-token call is not a 32-token kernel; every width does the same arithmetic per
    // (row, token), so the choice changes speed, not bits
    if (M <= 1) launch<Dec, 1>(w, rb, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
    else if (M <= 4) launch<Dec, 4>(w, rb, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
    else if (M <= 8) launch<Dec, 8>(w, rb, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
    else launch<Dec, MTMAX>(w, rb, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
}

}  // namespace

int64_t nemo_block_values(NemoWType t) {
    switch (t) {
    case NemoWType::NVFP4: case NemoWType::Q2_0: return 64;
    case NemoWType::IQ4_XS: case NemoWType::IQ3_XXS: return 256;
    default: return 32;
    }
}

uint64_t nemo_row_bytes(NemoWType t, int64_t cols) {
    switch (t) {
    case NemoWType::Q2_0: return (uint64_t) cols / 64 * 18;
    case NemoWType::IQ4_NL: return (uint64_t) cols / 32 * 18;
    case NemoWType::IQ4_XS: return (uint64_t) cols / 256 * 136;
    case NemoWType::IQ3_XXS: return (uint64_t) cols / 256 * 98;
    case NemoWType::NVFP4: return (uint64_t) cols / 64 * 36;
    case NemoWType::Q8_0: return (uint64_t) cols / 32 * 34;
    case NemoWType::BF16: return (uint64_t) cols * 2;
    case NemoWType::F32: return (uint64_t) cols * 4;
    }
    return 0;
}

void nemo_gemm(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
               int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream) {
    if (M <= 0 || rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const uint64_t rb = nemo_row_bytes(t, cols);
    switch (t) {
    case NemoWType::NVFP4: dispatch<DecNvfp4, 32>(w, rb, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::Q8_0: dispatch<DecQ8, 16>(w, rb, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::BF16: dispatch<DecBf16, 32>(w, rb, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::F32: dispatch<DecF32, 32>(w, rb, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    default: break;   // the expert-only formats: nemo_gemv
    }
}

}  // namespace strata::kernels
