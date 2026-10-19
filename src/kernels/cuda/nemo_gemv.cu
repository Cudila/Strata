// src/kernels/cuda/nemo_gemv.cu - see include/strata/kernels/nemo_gemv.hpp.
//
// A warp-group of LPR lanes owns an output row; each lane takes whole 64-value NVFP4 blocks (36 bytes: 4 UE4M3
// scales, 32 bytes of nibbles - nine 4-byte loads, the block is only 4-byte aligned), so a warp reads LPR consecutive
// blocks per step and keeps several steps in flight (unrolled).  The input rows are staged once per block in shared
// memory (4 floats of padding per 64 values, against bank conflicts) and the blocks stride over the rows (grid-stride), so the staging is paid once per ~rows/grid rows, not per
// 8 rows.  E2M1 is decoded through a 16-entry table in shared memory: lanes reading different entries hit different
// banks, so the lookups do not serialize the way a __constant__ table would.
//
// The sum per (row, token) is: per lane, its blocks in order, each block's four sub-blocks d_s * sum(e2m1 * x); then
// a butterfly over the LPR lanes; then * scale_2.  It does not depend on M (each token has its own accumulators), so
// a verify window's rows match the one-token rows bit for bit.
#include "strata/kernels/nemo_gemv.hpp"
#include "strata/kernels/nemo_tc.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::kernels {
namespace {

__constant__ float kE2M1v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// UE4M3 -> float without ldexpf: exponent field e (bias 7), mantissa m; e == 0 is m * 2^-9.
__device__ __forceinline__ float ue4m3v(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float) m * (1.0f / 512.0f);
}

constexpr int WARPS = 8;

/// The table's words in shared memory, rounded up so the staged rows stay 16-byte aligned (at least 4).
template <class D>
__host__ __device__ constexpr int TABW() { return D::TAB < 4 ? 4 : (D::TAB + 3) / 4 * 4; }

/// The weight formats, one ggml block per lane step.  dot<MT>(blk, xr, xo, tab, acc): acc[m] += block . x_m, the
/// block's values at xr[m] + xo (staged, padded) in the block's order.
struct DNvfp4 {   // 64 values, 36 bytes: 4 UE4M3 scales, 32 bytes of nibbles (sub-block s: byte j = values j, j + 8)
    static constexpr int VALS = 64, BYTES = 36, TAB = 16;
    __device__ static void init(float* t) {
        if (threadIdx.x < 16) t[threadIdx.x] = kE2M1v[threadIdx.x];
    }
    template <int MT>
    __device__ static void dot(const uint8_t* __restrict__ blk, const float* const* xr, int xo,
                               const float* __restrict__ tab, float* acc) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(blk);
        uint32_t q[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) q[i] = __ldg(p + i);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3v((q[0] >> (8 * s)) & 0xffu);
            const uint32_t w0 = q[1 + 2 * s], w1 = q[2 + 2 * s];
            float wv[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t b0 = (w0 >> (8 * j)) & 0xffu, b1 = (w1 >> (8 * j)) & 0xffu;
                wv[j] = tab[b0 & 15u];
                wv[j + 8] = tab[b0 >> 4];
                wv[j + 4] = tab[b1 & 15u];
                wv[j + 12] = tab[b1 >> 4];
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const float4* xv = reinterpret_cast<const float4*>(xr[m] + xo + s * 16);
                float a = 0.0f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const float4 x4 = xv[v];
                    a += wv[4 * v] * x4.x + wv[4 * v + 1] * x4.y + wv[4 * v + 2] * x4.z + wv[4 * v + 3] * x4.w;
                }
                acc[m] += d * a;
            }
        }
    }
};

struct DQ8 {   // 32 values, 34 bytes: f16 d, 32 int8 (2-byte aligned)
    static constexpr int VALS = 32, BYTES = 34, TAB = 0;
    __device__ static void init(float*) {}
    template <int MT>
    __device__ static void dot(const uint8_t* __restrict__ blk, const float* const* xr, int xo, const float*, float* acc) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(blk);
        uint16_t q[17];
#pragma unroll
        for (int i = 0; i < 17; ++i) q[i] = __ldg(p + i);
        const float d = __half2float(__ushort_as_half(q[0]));
        float wv[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            wv[2 * j] = (float) (int8_t) (q[1 + j] & 0xff);
            wv[2 * j + 1] = (float) (int8_t) (q[1 + j] >> 8);
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float4* xv = reinterpret_cast<const float4*>(xr[m] + xo);
            float a = 0.0f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float4 x4 = xv[v];
                a += wv[4 * v] * x4.x + wv[4 * v + 1] * x4.y + wv[4 * v + 2] * x4.z + wv[4 * v + 3] * x4.w;
            }
            acc[m] += d * a;
        }
    }
};

struct DBf16 {   // 32 values, 64 bytes (16-byte aligned)
    static constexpr int VALS = 32, BYTES = 64, TAB = 0;
    __device__ static void init(float*) {}
    template <int MT>
    __device__ static void dot(const uint8_t* __restrict__ blk, const float* const* xr, int xo, const float*, float* acc) {
        const uint4* p = reinterpret_cast<const uint4*>(blk);
        uint32_t q[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 u = __ldg(p + i);
            q[4 * i] = u.x; q[4 * i + 1] = u.y; q[4 * i + 2] = u.z; q[4 * i + 3] = u.w;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float4* xv = reinterpret_cast<const float4*>(xr[m] + xo);
            float a = 0.0f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float4 x4 = xv[v];
                a += __uint_as_float(q[2 * v] << 16) * x4.x + __uint_as_float(q[2 * v] & 0xffff0000u) * x4.y +
                     __uint_as_float(q[2 * v + 1] << 16) * x4.z + __uint_as_float(q[2 * v + 1] & 0xffff0000u) * x4.w;
            }
            acc[m] += a;
        }
    }
};

struct DF32 {   // 32 values, 128 bytes
    static constexpr int VALS = 32, BYTES = 128, TAB = 0;
    __device__ static void init(float*) {}
    template <int MT>
    __device__ static void dot(const uint8_t* __restrict__ blk, const float* const* xr, int xo, const float*, float* acc) {
        const float4* p = reinterpret_cast<const float4*>(blk);
        float4 q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = __ldg(p + i);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float4* xv = reinterpret_cast<const float4*>(xr[m] + xo);
            float a = 0.0f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float4 x4 = xv[v];
                a += q[v].x * x4.x + q[v].y * x4.y + q[v].z * x4.z + q[v].w * x4.w;
            }
            acc[m] += a;
        }
    }
};


/// The ggml i-quant / 2-bit formats of the expert files (llama.cpp's layouts and grids, ggml-common.h): deq(blk, sb,
/// tab, w) decodes 32-value sub-block sb (scale included) and dot32 multiplies it into the token rows.
template <int MT>
__device__ __forceinline__ void dot32(const float* w, const float* const* xr, int xo, float* acc) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float4* xv = reinterpret_cast<const float4*>(xr[m] + xo);
        float a = 0.0f;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const float4 x4 = xv[v];
            a += w[4 * v] * x4.x + w[4 * v + 1] * x4.y + w[4 * v + 2] * x4.z + w[4 * v + 3] * x4.w;
        }
        acc[m] += a;
    }
}

__device__ __forceinline__ float f16at(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p)));
}

template <class Self>
struct BySub {
    template <int MT>
    __device__ static void dot(const uint8_t* __restrict__ blk, const float* const* xr, int xo,
                               const float* __restrict__ tab, float* acc) {
#pragma unroll
        for (int sb = 0; sb < Self::VALS / 32; ++sb) {
            float w[32];
            Self::deq(blk, sb, tab, w);
            dot32<MT>(w, xr, xo + sb * 32, acc);
        }
    }
};

struct DQ2_0 : BySub<DQ2_0> {   // 64 values, 18 bytes: f16 d, 16 bytes of 2-bit codes (value j: byte j/4, bits 2(j%4))
    static constexpr int VALS = 64, BYTES = 18, TAB = 0;
    __device__ static void init(float*) {}
    __device__ static void deq(const uint8_t* blk, int sb, const float*, float* w) {
        const float d = f16at(blk);
        const uint16_t* q = reinterpret_cast<const uint16_t*>(blk + 2 + sb * 8);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t u = __ldg(q + i);
#pragma unroll
            for (int j = 0; j < 8; ++j) w[i * 8 + j] = d * (float) ((int) ((u >> (2 * j)) & 3u) - 1);
        }
    }
};

struct DIq4nl : BySub<DIq4nl> {   // 32 values, 18 bytes: f16 d, 16 bytes (byte j: value j low nibble, j + 16 high)
    static constexpr int VALS = 32, BYTES = 18, TAB = 16;
    __device__ static void init(float* t) {
        if (threadIdx.x < 16) t[threadIdx.x] = (float) kvalues_iq4nl[threadIdx.x];
    }
    __device__ static void deq(const uint8_t* blk, int, const float* tab, float* w) {
        const float d = f16at(blk);
        const uint16_t* q = reinterpret_cast<const uint16_t*>(blk + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t u = __ldg(q + i);
            w[2 * i] = d * tab[u & 15u];
            w[2 * i + 16] = d * tab[(u >> 4) & 15u];
            w[2 * i + 1] = d * tab[(u >> 8) & 15u];
            w[2 * i + 17] = d * tab[(u >> 12) & 15u];
        }
    }
};

struct DIq4xs : BySub<DIq4xs> {   // 256 values, 136 bytes: f16 d, u16 scales_h, 4 scales_l, 128 bytes of nibbles
    static constexpr int VALS = 256, BYTES = 136, TAB = 16;
    __device__ static void init(float* t) {
        if (threadIdx.x < 16) t[threadIdx.x] = (float) kvalues_iq4nl[threadIdx.x];
    }
    __device__ static void deq(const uint8_t* blk, int ib, const float* tab, float* w) {
        const float d = f16at(blk);
        const uint32_t sh = *reinterpret_cast<const uint16_t*>(blk + 2);
        const uint32_t sl = blk[4 + ib / 2];
        const int ls = (int) (((sl >> 4 * (ib % 2)) & 0xfu) | (((sh >> 2 * ib) & 3u) << 4));
        const float dl = d * (float) (ls - 32);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 8 + ib * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t u = __ldg(q + i);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[4 * i + j] = dl * tab[(u >> (8 * j)) & 15u];
                w[4 * i + j + 16] = dl * tab[(u >> (8 * j + 4)) & 15u];
            }
        }
    }
};

struct DIq3xxs : BySub<DIq3xxs> {   // 256 values, 98 bytes: f16 d, 64 grid indices, 32 bytes of scales and signs
    static constexpr int VALS = 256, BYTES = 98, TAB = 256 + 32;   // iq3xxs_grid, then ksigns_iq2xs (128 bytes)
    __device__ static void init(float* t) {
        uint32_t* g = reinterpret_cast<uint32_t*>(t);
        for (int i = threadIdx.x; i < 256; i += blockDim.x) g[i] = iq3xxs_grid[i];
        uint8_t* ks = reinterpret_cast<uint8_t*>(g + 256);
        for (int i = threadIdx.x; i < 128; i += blockDim.x) ks[i] = ksigns_iq2xs[i];
    }
    __device__ static void deq(const uint8_t* blk, int ib, const float* tab, float* w) {
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(tab);
        const uint8_t* ks = reinterpret_cast<const uint8_t*>(grid + 256);
        const float d = f16at(blk);
        const uint8_t* qs = blk + 2 + ib * 8;
        const uint16_t* ssp = reinterpret_cast<const uint16_t*>(blk + 2 + 64 + 4 * ib);
        const uint32_t aux = (uint32_t) ssp[0] | ((uint32_t) ssp[1] << 16);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t signs = ks[(aux >> (7 * l)) & 127u];
            const uint32_t g1 = grid[qs[2 * l]], g2 = grid[qs[2 * l + 1]];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v1 = (float) ((g1 >> (8 * j)) & 0xffu), v2 = (float) ((g2 >> (8 * j)) & 0xffu);
                w[8 * l + j] = db * ((signs >> j) & 1u ? -v1 : v1);
                w[8 * l + j + 4] = db * ((signs >> (j + 4)) & 1u ? -v2 : v2);
            }
        }
    }
};

/// Rows of one matrix (or of matrix blockIdx.y: `wp` non-null) against MT token rows staged in shared memory.
/// LPR lanes per row (cols / 64 >= LPR, a power of two <= 32).
template <class D, int MT, int LPR, bool SH>
__global__ void __launch_bounds__(WARPS * 32) gemv_kernel(
    const uint8_t* __restrict__ w1, const uint8_t* const* __restrict__ wp, float s2_1, const float* __restrict__ s2p,
    const float* __restrict__ x, int64_t ldx, int64_t x_stride, float* __restrict__ y, int64_t ldy, int64_t y_stride,
    const int32_t* __restrict__ idx, const int32_t* __restrict__ xidx, int rows, int cols, int M, int relu2,
    int accumulate) {
    extern __shared__ float sm[];
    float* tab = sm;                 // D::TAB words
    float* xs = sm + TABW<D>();      // MT staged rows
    // matrix j writes output row jo and reads input row jx (xidx[j], else jo)
    const int j = blockIdx.y, jo = idx ? idx[j] : j, jx = xidx ? xidx[j] : jo;
    const uint8_t* w = wp ? wp[j] : w1;
    const float s2 = s2p ? s2p[j] : s2_1;
    const float* xin = x + (size_t) jx * x_stride;
    D::init(tab);
    if (SH)
        for (int i = threadIdx.x; i < MT * cols; i += blockDim.x) {
            const int m = i / cols, c = i % cols;
            xs[m * (cols + cols / D::VALS * 4) + c + c / D::VALS * 4] = m < M ? xin[(size_t) m * ldx + c] : 0.0f;
        }
    __syncthreads();
    // staged, or (too wide for shared memory) read through L1 - the rows past M then repeat row 0, which is computed
    // and dropped
    const float* xr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m)
        xr[m] = SH ? xs + m * (cols + cols / D::VALS * 4) : xin + (size_t) (m < M ? m : 0) * ldx;
    constexpr int GPW = 32 / LPR;   // rows per warp at a time
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int sub = lane % LPR, grp = lane / LPR;
    const int nb = cols / D::VALS;
    constexpr int XB = SH ? D::VALS + 4 : D::VALS;   // a block's stride in the staged rows: 4 floats of padding, so
                                                     // the lanes' 16-byte loads (one block apart) hit different banks
    const uint64_t rb = (uint64_t) nb * D::BYTES;
    for (int r0 = (blockIdx.x * WARPS + warp) * GPW; r0 < rows; r0 += gridDim.x * WARPS * GPW) {
        const int r = r0 + grp;
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
        if (r < rows) {
            const uint8_t* row = w + (size_t) r * rb;
            int b = sub;
#pragma unroll 1
            for (; b + LPR < nb; b += 2 * LPR) {
                D::template dot<MT>(row + (size_t) b * D::BYTES, xr, b * XB, tab, acc);
                D::template dot<MT>(row + (size_t) (b + LPR) * D::BYTES, xr, (b + LPR) * XB, tab, acc);
            }
            if (b < nb) D::template dot<MT>(row + (size_t) b * D::BYTES, xr, b * XB, tab, acc);
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int o = LPR / 2; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        if (r < rows && sub == 0) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= M) break;
                float v = acc[m] * s2;
                if (relu2) {
                    v = fmaxf(v, 0.0f);
                    v *= v;
                }
                float* out = y + (size_t) jo * y_stride + (size_t) m * ldy + r;
                *out = accumulate ? *out + v : v;
            }
        }
    }
}


/// Few rows (the router's 512, k / v, fc1): the 8 warps of a block share one row - warp w takes the row's blocks
/// w * 32 + lane, w * 32 + lane + 256, ... - and their sums meet in shared memory in warp order (deterministic), so
/// a 512-row matrix is 512 blocks, not 64.  x is read through L1 (no staging: a block covers one row at a time).
template <class D, int MT>
__global__ void __launch_bounds__(WARPS * 32) gemv_wide_kernel(
    const uint8_t* __restrict__ w, float s2, const float* __restrict__ x, int64_t ldx, float* __restrict__ y,
    int64_t ldy, int rows, int cols, int M, int relu2, int accumulate) {
    extern __shared__ float sm[];
    float* tab = sm;
    __shared__ float part[WARPS][MT];
    D::init(tab);
    __syncthreads();
    const float* xr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) xr[m] = x + (size_t) (m < M ? m : 0) * ldx;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nb = cols / D::VALS;
    const uint64_t rb = (uint64_t) nb * D::BYTES;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
        const uint8_t* row = w + (size_t) r * rb;
        for (int b = warp * 32 + lane; b < nb; b += WARPS * 32)
            D::template dot<MT>(row + (size_t) b * D::BYTES, xr, b * D::VALS, tab, acc);
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        if (lane == 0)
#pragma unroll
            for (int m = 0; m < MT; ++m) part[warp][m] = acc[m];
        __syncthreads();
        if (threadIdx.x < MT && threadIdx.x < M) {
            const int m = threadIdx.x;
            float v = 0.0f;
            for (int q = 0; q < WARPS; ++q) v += part[q][m];
            v *= s2;
            if (relu2) {
                v = fmaxf(v, 0.0f);
                v *= v;
            }
            float* out = y + (size_t) m * ldy + r;
            *out = accumulate ? *out + v : v;
        }
        __syncthreads();
    }
}


// ---- tensor cores for a verify window (M <= 8 tokens): mma.sync m16n8k16, BF16 in, FP32 accumulate ----
//
// D (16 x 8) = A (16 x 16) B (16 x 8): A = the tokens' x (rows 0..7; 8..15 zero), B = 8 output rows' weights for 16
// columns, decoded to BF16 in registers (NVFP4 exactly; Q8_0 rounded).  Lane (g = lane / 4, t = lane % 4) holds
// A[g][2t, 2t+1], A[g][2t+8, 2t+9]; B[2t, 2t+1][g], B[2t+8, 2t+9][g] (= output row g's weights at those columns);
// D[g][2t, 2t+1].  An NVFP4 sub-block is one k-step: its byte j holds columns j and j + 8, so lane t's u16 at bytes
// 2t, 2t+1 is its whole B fragment.  A block of 8 warps owns NT * 8 output rows and splits the k-steps among its
// warps; the warps' sums meet in shared memory in warp order.  x is read as FP32 and rounded to BF16 (the W4A16
// activations of the prompt path).
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

struct TNvfp4 {
    static constexpr int VALS = 64, BYTES = 36;
    __device__ static void frag(const uint8_t* row, int ks, int t, const float* tab, uint32_t& b0, uint32_t& b1) {
        const uint8_t* blk = row + (size_t) (ks >> 2) * 36;
        const int s = ks & 3;
        const float d = ue4m3v(blk[s]);
        const uint32_t q = *reinterpret_cast<const uint16_t*>(blk + 4 + 8 * s + 2 * t);
        b0 = bf2(tab[q & 15u] * d, tab[(q >> 8) & 15u] * d);
        b1 = bf2(tab[(q >> 4) & 15u] * d, tab[(q >> 12) & 15u] * d);
    }
};

struct TQ8 {
    static constexpr int VALS = 32, BYTES = 34;
    __device__ static void frag(const uint8_t* row, int ks, int t, const float*, uint32_t& b0, uint32_t& b1) {
        const uint8_t* blk = row + (size_t) (ks >> 1) * 34;
        const float d = f16at(blk);
        const uint8_t* q = blk + 2 + 16 * (ks & 1);
        const uint32_t q0 = *reinterpret_cast<const uint16_t*>(q + 2 * t), q1 = *reinterpret_cast<const uint16_t*>(q + 8 + 2 * t);
        b0 = bf2(d * (float) (int8_t) (q0 & 0xff), d * (float) (int8_t) (q0 >> 8));
        b1 = bf2(d * (float) (int8_t) (q1 & 0xff), d * (float) (int8_t) (q1 >> 8));
    }
};

struct TBf16 {
    static constexpr int VALS = 16, BYTES = 32;
    __device__ static void frag(const uint8_t* row, int ks, int t, const float*, uint32_t& b0, uint32_t& b1) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(row + (size_t) ks * 32);
        b0 = p[t];
        b1 = p[4 + t];
    }
};

template <class TB, int NT>
__global__ void __launch_bounds__(256) tc_gemv_kernel(const uint8_t* __restrict__ w, float s2, const float* __restrict__ x,
                                                      int64_t ldx, float* __restrict__ y, int64_t ldy, int rows,
                                                      int cols, int M, int relu2, int accumulate) {
    __shared__ float tab[16];
    __shared__ float red[8][NT][8][8];
    if (threadIdx.x < 16) tab[threadIdx.x] = kE2M1v[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.x * 8 * NT;
    const uint64_t rb = (uint64_t) (cols / TB::VALS) * TB::BYTES;
    const int KS = cols / 16;
    const int k0 = (int) ((int64_t) KS * warp / 8), k1 = (int) ((int64_t) KS * (warp + 1) / 8);
    const bool gv = g < M;
    const float* xg = x + (size_t) (gv ? g : 0) * ldx;
    const uint8_t* rowp[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        int r = r0 + nt * 8 + g;
        rowp[nt] = w + (size_t) (r < rows ? r : rows - 1) * rb;
    }
    float c[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
#pragma unroll 2
    for (int ks = k0; ks < k1; ++ks) {
        uint32_t a0 = 0, a2 = 0;
        if (gv) {
            const float2 xa = *reinterpret_cast<const float2*>(xg + ks * 16 + 2 * t);
            const float2 xb = *reinterpret_cast<const float2*>(xg + ks * 16 + 8 + 2 * t);
            a0 = bf2(xa.x, xa.y);
            a2 = bf2(xb.x, xb.y);
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            uint32_t b0, b1;
            TB::frag(rowp[nt], ks, t, tab, b0, b1);
            mma_bf16(c[nt], a0, a2, b0, b1);
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        red[warp][nt][g][2 * t] = c[nt][0];
        red[warp][nt][g][2 * t + 1] = c[nt][1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * 64; i += blockDim.x) {
        const int nt = i / 64, m = (i / 8) % 8, rr = i % 8, r = r0 + nt * 8 + rr;
        if (m >= M || r >= rows) continue;
        float v = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) v += red[q][nt][m][rr];
        v *= s2;
        if (relu2) {
            v = fmaxf(v, 0.0f);
            v *= v;
        }
        float* out = y + (size_t) m * ldy + r;
        *out = accumulate ? *out + v : v;
    }
}

template <class TB>
void tc_launch(const void* w, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, int64_t cols,
               int64_t M, bool relu2, bool acc, cudaStream_t st) {
    constexpr int NT = 4;
    const int64_t blocks = (rows + 8 * NT - 1) / (8 * NT);
    tc_gemv_kernel<TB, NT><<<(unsigned) blocks, 256, 0, st>>>((const uint8_t*) w, s2, x, ldx, y, ldy, (int) rows,
                                                             (int) cols, (int) M, relu2 ? 1 : 0, acc ? 1 : 0);
}


// ---- the tiled layout: the tensor-core kernels' own (dense matrices are repacked at load) ----
//
// Rows in tiles of 8 (padded with zero rows), columns in the format's blocks; tile T's block b at (T * nb + b) * TBB:
// 32 lanes x 8 bytes - lane (g, t) gets exactly its B fragments for the block's k-steps, so a warp's load is 256
// contiguous bytes - then the 8 rows' scales.
//   NVFP4 (64 columns, 4 k-steps): lane: 4 x u16 = row g's bytes 4 + 8s + 2t, +1; then 8 x 4 scale bytes   (288)
//   Q8_0  (32 columns, 2 k-steps): lane: [h][2] u16 = bytes 2 + 16h + 2t, +1 and 2 + 16h + 8 + 2t, +1; 8 f16 (272)
//   BF16  (16 columns, 1 k-step):  lane: u32 columns 2t, 2t+1; u32 columns 2t+8, 2t+9                        (256)
struct LNvfp4 {
    static constexpr int VALS = 64, KS = 4, TBB = 288;
    template <int NTMAX>
    __device__ static void frags(const uint8_t* tb, int g, int t, const float* tab, uint32_t (*b)[2]) {
        const int lane = g * 4 + t;
        const uint2 q = *reinterpret_cast<const uint2*>(tb + lane * 8);
        const uint32_t sc = *reinterpret_cast<const uint32_t*>(tb + 256 + g * 4);
        const uint32_t qq[2] = {q.x, q.y};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint32_t u = (qq[s >> 1] >> (16 * (s & 1))) & 0xffffu;
            const float d = ue4m3v((sc >> (8 * s)) & 0xffu);
            b[s][0] = bf2(tab[u & 15u] * d, tab[(u >> 8) & 15u] * d);
            b[s][1] = bf2(tab[(u >> 4) & 15u] * d, tab[(u >> 12) & 15u] * d);
        }
    }
};

struct LQ8 {
    static constexpr int VALS = 32, KS = 2, TBB = 272;
    template <int NTMAX>
    __device__ static void frags(const uint8_t* tb, int g, int t, const float*, uint32_t (*b)[2]) {
        const int lane = g * 4 + t;
        const uint2 q = *reinterpret_cast<const uint2*>(tb + lane * 8);
        const float d = f16at(tb + 256 + g * 2);
        const uint32_t qq[2] = {q.x, q.y};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t u = qq[h];
            b[h][0] = bf2(d * (float) (int8_t) (u & 0xff), d * (float) (int8_t) ((u >> 8) & 0xff));
            b[h][1] = bf2(d * (float) (int8_t) ((u >> 16) & 0xff), d * (float) (int8_t) (u >> 24));
        }
    }
};

struct LBf16 {
    static constexpr int VALS = 16, KS = 1, TBB = 256;
    template <int NTMAX>
    __device__ static void frags(const uint8_t* tb, int g, int t, const float*, uint32_t (*b)[2]) {
        const uint2 q = *reinterpret_cast<const uint2*>(tb + (g * 4 + t) * 8);
        b[0][0] = q.x;
        b[0][1] = q.y;
    }
};

template <class TL, int NT>
__global__ void __launch_bounds__(256) tc_tiled_kernel(const uint8_t* __restrict__ w, float s2,
                                                       const float* __restrict__ x, int64_t ldx, float* __restrict__ y,
                                                       int64_t ldy, int rows, int cols, int M, int relu2,
                                                       int accumulate) {
    __shared__ float tab[16];
    __shared__ float red[8][NT][8][8];
    if (threadIdx.x < 16) tab[threadIdx.x] = kE2M1v[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    const int nb = cols / TL::VALS, ntile = (rows + 7) / 8;
    const int T0 = blockIdx.x * NT;
    const int b0 = (int) ((int64_t) nb * warp / 8), b1 = (int) ((int64_t) nb * (warp + 1) / 8);
    const bool gv = g < M;
    const float* xg = x + (size_t) (gv ? g : 0) * ldx;
    float c[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
    for (int bb = b0; bb < b1; ++bb) {
        uint32_t a0[TL::KS], a2[TL::KS];
#pragma unroll
        for (int k = 0; k < TL::KS; ++k) {
            a0[k] = a2[k] = 0;
            if (gv) {
                const float* xk = xg + (size_t) bb * TL::VALS + k * 16;
                const float2 xa = *reinterpret_cast<const float2*>(xk + 2 * t);
                const float2 xb = *reinterpret_cast<const float2*>(xk + 8 + 2 * t);
                a0[k] = bf2(xa.x, xa.y);
                a2[k] = bf2(xb.x, xb.y);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int T = T0 + nt < ntile ? T0 + nt : ntile - 1;
            uint32_t bf[TL::KS][2];
            TL::template frags<NT>(w + ((size_t) T * nb + bb) * TL::TBB, g, t, tab, bf);
#pragma unroll
            for (int k = 0; k < TL::KS; ++k) mma_bf16(c[nt], a0[k], a2[k], bf[k][0], bf[k][1]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        red[warp][nt][g][2 * t] = c[nt][0];
        red[warp][nt][g][2 * t + 1] = c[nt][1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * 64; i += blockDim.x) {
        const int nt = i / 64, m = (i / 8) % 8, rr = i % 8, r = (T0 + nt) * 8 + rr;
        if (m >= M || r >= rows) continue;
        float v = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) v += red[q][nt][m][rr];
        v *= s2;
        if (relu2) {
            v = fmaxf(v, 0.0f);
            v *= v;
        }
        float* out = y + (size_t) m * ldy + r;
        *out = accumulate ? *out + v : v;
    }
}

template <class TL, int NT>
void tiled_launch_nt(const void* w, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
                     int64_t cols, int64_t M, bool relu2, bool acc, cudaStream_t st) {
    const int64_t ntile = (rows + 7) / 8;
    tc_tiled_kernel<TL, NT><<<(unsigned) ((ntile + NT - 1) / NT), 256, 0, st>>>(
        (const uint8_t*) w, s2, x, ldx, y, ldy, (int) rows, (int) cols, (int) M, relu2 ? 1 : 0, acc ? 1 : 0);
}

int g_tc_sms = 0;

template <class TL>
void tiled_launch(const void* w, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
                  int64_t cols, int64_t M, bool relu2, bool acc, cudaStream_t st) {
    if (!g_tc_sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_tc_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    // the most row tiles a block that still gives every SM ~3 blocks (a block's 8 warps split the columns)
    const int64_t ntile = (rows + 7) / 8;
    if (ntile >= (int64_t) g_tc_sms * 24) tiled_launch_nt<TL, 4>(w, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
    else if (ntile >= (int64_t) g_tc_sms * 6) tiled_launch_nt<TL, 2>(w, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
    else tiled_launch_nt<TL, 1>(w, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
}

/// The tiled layout back to row-major BF16 (the prompt path's cuBLAS input): a warp per (tile, block) - its lanes'
/// fragments into shared memory, then the 8 rows out in 16-byte stores.
template <class TL>
__global__ void __launch_bounds__(256) tiled_deq_kernel(const uint8_t* __restrict__ w, uint16_t* __restrict__ out,
                                                        int rows, int cols) {
    __shared__ float tab[16];
    __shared__ __align__(16) uint16_t sm[8][8][TL::VALS];   // [warp][row][column]
    if (threadIdx.x < 16) tab[threadIdx.x] = kE2M1v[threadIdx.x];
    __syncthreads();
    const int nb = cols / TL::VALS, wp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int64_t units = (int64_t) ((rows + 7) / 8) * nb;
    for (int64_t u = (int64_t) blockIdx.x * 8 + wp; u < units; u += (int64_t) gridDim.x * 8) {
        const int T = (int) (u / nb), bb = (int) (u % nb);
        uint32_t bf[TL::KS][2];
        TL::template frags<1>(w + (size_t) u * TL::TBB, g, t, tab, bf);
#pragma unroll
        for (int k = 0; k < TL::KS; ++k) {
            // b0 = columns 2t, 2t+1 of k-step k; b1 = columns 2t+8, 2t+9
            *reinterpret_cast<uint32_t*>(&sm[wp][g][k * 16 + 2 * t]) = bf[k][0];
            *reinterpret_cast<uint32_t*>(&sm[wp][g][k * 16 + 2 * t + 8]) = bf[k][1];
        }
        __syncwarp();
        constexpr int Q = TL::VALS / 8;   // 16-byte pieces a row
        for (int q = lane; q < 8 * Q; q += 32) {
            const int r = T * 8 + q / Q;
            if (r < rows)
                *reinterpret_cast<uint4*>(out + (size_t) r * cols + (size_t) bb * TL::VALS + (q % Q) * 8) =
                    *reinterpret_cast<const uint4*>(&sm[wp][q / Q][(q % Q) * 8]);
        }
        __syncwarp();
    }
}

__device__ __forceinline__ uint16_t bf16_rne_g(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

/// The prompt path: matrix blockIdx.y (ptrs[y], rows x cols) to BF16, one thread per 32-value sub-block.
template <class D>
__global__ void deq_bf16_kernel(const uint8_t* const* __restrict__ ptrs, uint16_t* __restrict__ out, int64_t rows,
                                int64_t cols) {
    extern __shared__ float tab[];
    D::init(tab);
    __syncthreads();
    const uint8_t* w = ptrs[blockIdx.y];
    uint16_t* o = out + (size_t) blockIdx.y * rows * cols;
    const int64_t per_row = cols / 32;
    const int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= rows * per_row) return;
    const int64_t r = u / per_row, c0 = (u % per_row) * 32;
    const uint8_t* blk = w + (size_t) r * (cols / D::VALS) * D::BYTES + (size_t) (c0 / D::VALS) * D::BYTES;
    float v[32];
    D::deq(blk, (int) ((c0 % D::VALS) / 32), tab, v);
    uint16_t* dst = o + (size_t) r * cols + c0;
#pragma unroll
    for (int j = 0; j < 32; ++j) dst[j] = bf16_rne_g(v[j]);
}

template <class D>
void deq_bf16(const uint8_t* const* ptrs, uint16_t* out, int64_t n, int64_t rows, int64_t cols, cudaStream_t st) {
    const int64_t units = rows * (cols / 32);
    deq_bf16_kernel<D><<<dim3((unsigned) ((units + 255) / 256), (unsigned) n), 256, TABW<D>() * sizeof(float), st>>>(
        ptrs, out, rows, cols);
}

int g_sms = 0;

template <class D, int MT, int LPR, bool SH>
void launch(const uint8_t* w1, const uint8_t* const* wp, float s2, const float* s2p, const float* x, int64_t ldx,
            int64_t xs, float* y, int64_t ldy, int64_t ys, const int32_t* idx, const int32_t* xidx, int64_t rows, int64_t cols, int64_t M, int nmat, bool relu2,
            bool acc, cudaStream_t st) {
    const size_t smem = (TABW<D>() + (SH ? (size_t) MT * (cols + cols / D::VALS * 4) : 0)) * sizeof(float);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(gemv_kernel<D, MT, LPR, SH>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr = true;
    }
    if (!g_sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    constexpr int GPW = 32 / LPR;
    int64_t blocks = (rows + WARPS * GPW - 1) / (WARPS * GPW);
    // ~2 resident blocks per SM over all matrices: each block stages its token rows once and then sweeps many rows
    const int64_t cap = ((int64_t) g_sms * 2 + nmat - 1) / (nmat > 0 ? nmat : 1);
    if (blocks > cap) blocks = cap;
    gemv_kernel<D, MT, LPR, SH><<<dim3((unsigned) blocks, (unsigned) nmat), WARPS * 32, smem, st>>>(
        w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, (int) rows, (int) cols, (int) M, relu2 ? 1 : 0, acc ? 1 : 0);
}

template <class D, int MT, bool SH>
void by_lpr2(const uint8_t* w1, const uint8_t* const* wp, float s2, const float* s2p, const float* x, int64_t ldx,
            int64_t xs, float* y, int64_t ldy, int64_t ys, const int32_t* idx, const int32_t* xidx, int64_t rows, int64_t cols, int64_t M, int nmat, bool relu2,
            bool acc, cudaStream_t st) {
    // the widest group whose last step keeps >= 85% of its lanes busy (2688 columns = 42 blocks: 16 lanes, not 32)
    const int64_t nb = cols / D::VALS;
    auto eff = [&](int64_t l) { return (double) nb / (double) (((nb + l - 1) / l) * l); };
    if (nb >= 32 && eff(32) >= 0.85) launch<D, MT, 32, SH>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else if (nb >= 16 && eff(16) >= 0.85) launch<D, MT, 16, SH>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else if (nb >= 8 && eff(8) >= 0.85) launch<D, MT, 8, SH>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else launch<D, MT, 4, SH>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
}

template <class D, int MT>
void by_lpr(const uint8_t* w1, const uint8_t* const* wp, float s2, const float* s2p, const float* x, int64_t ldx,
            int64_t xs, float* y, int64_t ldy, int64_t ys, const int32_t* idx, const int32_t* xidx, int64_t rows, int64_t cols, int64_t M, int nmat, bool relu2,
            bool acc, cudaStream_t st) {
    if ((TABW<D>() + MT * (cols + cols / D::VALS * 4)) * 4 <= 99 * 1024)
        by_lpr2<D, MT, true>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else
        by_lpr2<D, MT, false>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
}

template <class D, int MT>
void launch_wide(const uint8_t* w, float s2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
                 int64_t cols, int64_t M, bool relu2, bool acc, cudaStream_t st) {
    const int64_t blocks = std::min<int64_t>(rows, (int64_t) g_sms * 4);
    gemv_wide_kernel<D, MT><<<(unsigned) blocks, WARPS * 32, TABW<D>() * sizeof(float), st>>>(
        w, s2, x, ldx, y, ldy, (int) rows, (int) cols, (int) M, relu2 ? 1 : 0, acc ? 1 : 0);
}

template <class D>
void dispatch(const uint8_t* w1, const uint8_t* const* wp, float s2, const float* s2p, const float* x, int64_t ldx,
              int64_t xs, float* y, int64_t ldy, int64_t ys, const int32_t* idx, const int32_t* xidx, int64_t rows, int64_t cols, int64_t M, int nmat,
              bool relu2, bool acc, cudaStream_t st) {
    if (!g_sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    if (!wp && rows * WARPS < (int64_t) g_sms * WARPS * 16 && cols / D::VALS >= 64) {
        // one matrix of few rows (fewer than ~16 per SM at a warp a row): a block per row
        if (M <= 1) launch_wide<D, 1>(w1, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
        else if (M <= 2) launch_wide<D, 2>(w1, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
        else if (M <= 4) launch_wide<D, 4>(w1, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
        else launch_wide<D, 8>(w1, s2, x, ldx, y, ldy, rows, cols, M, relu2, acc, st);
        return;
    }
    if (M <= 1) by_lpr<D, 1>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else if (M <= 2) by_lpr<D, 2>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else if (M <= 4) by_lpr<D, 4>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
    else by_lpr<D, 8>(w1, wp, s2, s2p, x, ldx, xs, y, ldy, ys, idx, xidx, rows, cols, M, nmat, relu2, acc, st);
}

}  // namespace

bool nemo_gemv_nvfp4_ok(int64_t cols, int64_t M) {
    return M >= 1 && M <= NEMO_GEMV_MAX_M && cols % 64 == 0 && cols >= 256;
}


uint64_t nemo_tiled_bytes(NemoWType t, int64_t rows, int64_t cols) {
    const int64_t nt = (rows + 7) / 8;
    switch (t) {
    case NemoWType::NVFP4: return (uint64_t) nt * (cols / 64) * 288;
    case NemoWType::Q8_0: return (uint64_t) nt * (cols / 32) * 272;
    case NemoWType::BF16: return (uint64_t) nt * (cols / 16) * 256;
    default: return 0;
    }
}

bool nemo_tiled_repack(NemoWType t, const uint8_t* src, int64_t rows, int64_t cols, uint8_t* dst) {
    const int64_t nt = (rows + 7) / 8;
    const uint64_t rb = nemo_row_bytes(t, cols);
    if (t == NemoWType::NVFP4) {
        const int64_t nb = cols / 64;
        for (int64_t T = 0; T < nt; ++T)
            for (int64_t b = 0; b < nb; ++b) {
                uint8_t* tb = dst + ((size_t) T * nb + b) * 288;
                std::memset(tb, 0, 288);
                for (int g = 0; g < 8; ++g) {
                    const int64_t r = T * 8 + g;
                    if (r >= rows) continue;
                    const uint8_t* blk = src + (size_t) r * rb + (size_t) b * 36;
                    std::memcpy(tb + 256 + g * 4, blk, 4);
                    for (int tt = 0; tt < 4; ++tt)
                        for (int s = 0; s < 4; ++s) {
                            tb[(g * 4 + tt) * 8 + 2 * s] = blk[4 + 8 * s + 2 * tt];
                            tb[(g * 4 + tt) * 8 + 2 * s + 1] = blk[4 + 8 * s + 2 * tt + 1];
                        }
                }
            }
        return true;
    }
    if (t == NemoWType::Q8_0) {
        const int64_t nb = cols / 32;
        for (int64_t T = 0; T < nt; ++T)
            for (int64_t b = 0; b < nb; ++b) {
                uint8_t* tb = dst + ((size_t) T * nb + b) * 272;
                std::memset(tb, 0, 272);
                for (int g = 0; g < 8; ++g) {
                    const int64_t r = T * 8 + g;
                    if (r >= rows) continue;
                    const uint8_t* blk = src + (size_t) r * rb + (size_t) b * 34;
                    std::memcpy(tb + 256 + g * 2, blk, 2);
                    for (int tt = 0; tt < 4; ++tt)
                        for (int h = 0; h < 2; ++h) {
                            uint8_t* o = tb + (g * 4 + tt) * 8 + 4 * h;
                            o[0] = blk[2 + 16 * h + 2 * tt];
                            o[1] = blk[2 + 16 * h + 2 * tt + 1];
                            o[2] = blk[2 + 16 * h + 8 + 2 * tt];
                            o[3] = blk[2 + 16 * h + 8 + 2 * tt + 1];
                        }
                }
            }
        return true;
    }
    if (t == NemoWType::BF16) {
        const int64_t nb = cols / 16;
        for (int64_t T = 0; T < nt; ++T)
            for (int64_t b = 0; b < nb; ++b) {
                uint8_t* tb = dst + ((size_t) T * nb + b) * 256;
                std::memset(tb, 0, 256);
                for (int g = 0; g < 8; ++g) {
                    const int64_t r = T * 8 + g;
                    if (r >= rows) continue;
                    const uint8_t* rowp = src + (size_t) r * rb + (size_t) b * 32;
                    for (int tt = 0; tt < 4; ++tt) {
                        std::memcpy(tb + (g * 4 + tt) * 8, rowp + 4 * tt, 4);
                        std::memcpy(tb + (g * 4 + tt) * 8 + 4, rowp + 16 + 4 * tt, 4);
                    }
                }
            }
        return true;
    }
    return false;
}

void nemo_tiled_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
                     int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream) {
    if (M <= 0 || rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    switch (t) {
    case NemoWType::NVFP4: tiled_launch<LNvfp4>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::Q8_0: tiled_launch<LQ8>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::BF16: tiled_launch<LBf16>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    default: break;
    }
}

void nemo_tiled_dequant_bf16(NemoWType t, const void* w, uint16_t* out, int64_t rows, int64_t cols, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    const int64_t units = (rows + 7) / 8 * (cols / nemo_block_values(t == NemoWType::BF16 ? NemoWType::F32 : t));
    const unsigned blocks = (unsigned) std::min<int64_t>((units + 7) / 8, 8192);
    switch (t) {
    case NemoWType::NVFP4: tiled_deq_kernel<LNvfp4><<<blocks, 256, 0, st>>>((const uint8_t*) w, out, (int) rows, (int) cols); break;
    case NemoWType::Q8_0: tiled_deq_kernel<LQ8><<<blocks, 256, 0, st>>>((const uint8_t*) w, out, (int) rows, (int) cols); break;
    case NemoWType::BF16: {
        const unsigned b16 = (unsigned) std::min<int64_t>(((rows + 7) / 8 * (cols / 16) + 7) / 8, 8192);
        tiled_deq_kernel<LBf16><<<b16, 256, 0, st>>>((const uint8_t*) w, out, (int) rows, (int) cols);
        break;
    }
    default: break;
    }
}

bool nemo_tc_gemv_ok(NemoWType t, int64_t cols, int64_t M) {
    if (M < 1 || M > 8 || cols < 128 || cols % 16) return false;
    return (t == NemoWType::NVFP4 && cols % 64 == 0) || (t == NemoWType::Q8_0 && cols % 32 == 0) ||
           t == NemoWType::BF16;
}

void nemo_tc_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
                  int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream) {
    if (M <= 0 || rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    switch (t) {
    case NemoWType::NVFP4: tc_launch<TNvfp4>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::Q8_0: tc_launch<TQ8>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    case NemoWType::BF16: tc_launch<TBf16>(w, scale2, x, ldx, y, ldy, rows, cols, M, relu2, accumulate, st); break;
    default: break;
    }
}

bool nemo_gemv_ok(NemoWType t, int64_t cols, int64_t M) {
    if (M < 1 || M > NEMO_GEMV_MAX_M || cols < 256) return false;
    return cols % nemo_block_values(t) == 0;
}

void nemo_gemv(NemoWType t, const void* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy,
               int64_t rows, int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream) {
    if (M <= 0 || rows <= 0) return;
    const uint8_t* w8 = (const uint8_t*) w;
    cudaStream_t st = (cudaStream_t) stream;
    switch (t) {
    case NemoWType::NVFP4:
        dispatch<DNvfp4>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::Q8_0:
        dispatch<DQ8>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::BF16:
        dispatch<DBf16>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::F32:
        dispatch<DF32>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::Q2_0:
        dispatch<DQ2_0>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::IQ4_NL:
        dispatch<DIq4nl>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::IQ4_XS:
        dispatch<DIq4xs>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    case NemoWType::IQ3_XXS:
        dispatch<DIq3xxs>(w8, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate, st);
        break;
    }
}

void nemo_gemv_nvfp4(const uint8_t* w, float scale2, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
                     int64_t cols, int64_t M, bool relu2, bool accumulate, void* stream) {
    if (M <= 0 || rows <= 0) return;
    dispatch<DNvfp4>(w, nullptr, scale2, nullptr, x, ldx, 0, y, ldy, 0, nullptr, nullptr, rows, cols, M, 1, relu2, accumulate,
             (cudaStream_t) stream);
}

void nemo_gemv_many(NemoWType t, const uint8_t* const* w, const float* scale2, const float* x, int64_t x_stride,
                    float* y, int64_t y_stride, int64_t rows, int64_t cols, int64_t n, bool relu2, void* stream,
                    const int32_t* idx, const int32_t* xidx) {
    if (n <= 0 || rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
#define NEMO_MANY(D) dispatch<D>(nullptr, w, 1.0f, scale2, x, cols, x_stride, y, rows, y_stride, idx, xidx, rows, cols, 1, \
                                 (int) n, relu2, false, st)
    switch (t) {
    case NemoWType::NVFP4: NEMO_MANY(DNvfp4); break;
    case NemoWType::Q2_0: NEMO_MANY(DQ2_0); break;
    case NemoWType::IQ4_NL: NEMO_MANY(DIq4nl); break;
    case NemoWType::IQ4_XS: NEMO_MANY(DIq4xs); break;
    case NemoWType::IQ3_XXS: NEMO_MANY(DIq3xxs); break;
    default: break;
    }
#undef NEMO_MANY
}

void nemo_dequant_bf16_many_t(NemoWType t, const uint8_t* const* ptr, uint16_t* out, int64_t n, int64_t rows,
                              int64_t cols, void* stream) {
    if (n <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    switch (t) {
    case NemoWType::NVFP4: nemo_dequant_bf16_many(ptr, out, n, rows, cols, stream); break;
    case NemoWType::Q2_0: deq_bf16<DQ2_0>(ptr, out, n, rows, cols, st); break;
    case NemoWType::IQ4_NL: deq_bf16<DIq4nl>(ptr, out, n, rows, cols, st); break;
    case NemoWType::IQ4_XS: deq_bf16<DIq4xs>(ptr, out, n, rows, cols, st); break;
    case NemoWType::IQ3_XXS: deq_bf16<DIq3xxs>(ptr, out, n, rows, cols, st); break;
    default: break;
    }
}

void nemo_gemv_nvfp4_many(const uint8_t* const* w, const float* scale2, const float* x, int64_t x_stride, float* y,
                          int64_t y_stride, int64_t rows, int64_t cols, int64_t n, bool relu2, void* stream,
                          const int32_t* idx) {
    if (n <= 0 || rows <= 0) return;
    dispatch<DNvfp4>(nullptr, w, 1.0f, scale2, x, cols, x_stride, y, rows, y_stride, idx, nullptr, rows, cols, 1, (int) n, relu2, false,
             (cudaStream_t) stream);
}

}  // namespace strata::kernels
