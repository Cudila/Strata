// src/kernels/cuda/nemo_fp4.cu - see include/strata/kernels/nemo_fp4.hpp.  Compiled for sm_120a only (the block-scaled
// FP4 mma is an arch-specific instruction); without it every entry point is a no-op and nemo_fp4_ok() is false.
//
// The GEMM: a block computes 128 rows (tokens) x 128 columns (weight rows) of Y; 8 warps as 2 x 4, a warp 64 x 32 =
// 4 m16 x 4 n8 tiles; K in steps of 64 through a 4-stage cp.async pipeline.  A stage is 128 rows x 32 bytes of each
// operand (two 16-byte chunks a row, swizzled by bit 2 of the row so ldmatrix's 8 rows land on distinct banks) and
// their 4 scale bytes a row.  The fragments are b16 ldmatrix loads: an 8 x 8 b16 matrix is 8 rows x 32 values, and
// thread (g, t) receives row g's bytes 4t .. 4t+3 = values 8t .. 8t+7, which is what the FP4 mma wants (see cute's
// SM120_16x8x64_TN_VS traits: A reg 0/1 rows g / g+8, values 8t.., regs 2/3 the same 32 values on; B reg 0/1 column
// g, values 8t.. and 32 + 8t..).  Scales: thread (g, t) gives A's row g + 8 (t & 1) and B's column g, a byte per 16 K.
#include "strata/kernels/nemo_fp4.hpp"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200)
#define NEMO_FP4_MMA 1
#endif

/// 1 when this device code has the FP4 mma (read by nemo_fp4_ok)
#ifdef NEMO_FP4_MMA
__device__ int g_has_mma = 1;
#else
__device__ int g_has_mma = 0;
#endif

#ifndef NEMO_FP4_KSUB
#define NEMO_FP4_KSUB 2
#endif
#ifndef NEMO_FP4_ST
#define NEMO_FP4_ST 3
#endif
constexpr int BM = 128, BN = 128, NTH = 256;
constexpr int KSUB = NEMO_FP4_KSUB, ST = NEMO_FP4_ST;   ///< 64-value K steps a stage, stages
constexpr int CH = 2 * KSUB;                       ///< 16-byte chunks of a stage row
constexpr int QB = BM * 32 * KSUB, SB = BM * 4 * KSUB;   // a stage's bytes of one operand: values, scales
constexpr int STAGE = 2 * QB + 2 * SB;            // A values, B values, A scales, B scales

__device__ __forceinline__ uint32_t pack_nib(uint64_t v) {   // 8 bytes of one nibble each -> 4 bytes, in order
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FFull;
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFFull;
    v = (v | (v >> 16)) & 0x00000000FFFFFFFFull;
    return (uint32_t) v;
}

/// ggml's 8 bytes of a 16-value sub-block (value j < 8: the low nibble of byte j, 8 + j: its high nibble) -> natural.
__device__ __forceinline__ uint2 natural16(uint64_t x) {
    return make_uint2(pack_nib(x & 0x0F0F0F0F0F0F0F0Full), pack_nib((x >> 4) & 0x0F0F0F0F0F0F0F0Full));
}

/// A thread per (matrix, row, 64-value block).
__global__ void from_ggml_kernel(const uint8_t* const* __restrict__ ptr, int64_t rows, int64_t cols,
                                 uint8_t* __restrict__ q, uint8_t* __restrict__ s) {
    const int64_t nb = cols / 64, units = rows * nb;
    const uint8_t* w = ptr[blockIdx.y];
    q += (size_t) blockIdx.y * rows * cols / 2;
    s += (size_t) blockIdx.y * rows * cols / 16;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = u / nb, b = u % nb;
        const uint32_t* blk = reinterpret_cast<const uint32_t*>(w + (size_t) (r * nb + b) * 36);   // 4-byte aligned
        uint4 o0, o1;
        uint2 v;
        v = natural16((uint64_t) blk[1] | ((uint64_t) blk[2] << 32)); o0.x = v.x; o0.y = v.y;
        v = natural16((uint64_t) blk[3] | ((uint64_t) blk[4] << 32)); o0.z = v.x; o0.w = v.y;
        v = natural16((uint64_t) blk[5] | ((uint64_t) blk[6] << 32)); o1.x = v.x; o1.y = v.y;
        v = natural16((uint64_t) blk[7] | ((uint64_t) blk[8] << 32)); o1.z = v.x; o1.w = v.y;
        uint4* d = reinterpret_cast<uint4*>(q + (size_t) r * (cols / 2) + (size_t) b * 32);
        d[0] = o0;
        d[1] = o1;
        *reinterpret_cast<uint32_t*>(s + (size_t) r * (cols / 16) + (size_t) b * 4) = blk[0];
    }
}

/// The tiled layout: tile T's block b (288 bytes) holds, for lane (g, t), row g's qs bytes 8s + 2t, +1 at lane * 8 + 2s
/// (s = the sub-block), then the 8 rows' 4 scale bytes.  A thread per (tile, block, row).
__global__ void from_tiled_kernel(const uint8_t* __restrict__ w, int rows, int cols, uint8_t* __restrict__ q,
                                  uint8_t* __restrict__ s) {
    const int nb = cols / 64;
    const int64_t units = (int64_t) rows * nb;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int r = (int) (u / nb), b = (int) (u % nb), T = r / 8, g = r % 8;
        const uint8_t* tb = w + ((size_t) T * nb + b) * 288;
        uint32_t nat[8];
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            uint64_t x = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint16_t p = *reinterpret_cast<const uint16_t*>(tb + (g * 4 + t) * 8 + 2 * sb);
                x |= (uint64_t) p << (16 * t);
            }
            const uint2 v = natural16(x);
            nat[2 * sb] = v.x;
            nat[2 * sb + 1] = v.y;
        }
        uint4* d = reinterpret_cast<uint4*>(q + (size_t) r * (cols / 2) + (size_t) b * 32);
        d[0] = make_uint4(nat[0], nat[1], nat[2], nat[3]);
        d[1] = make_uint4(nat[4], nat[5], nat[6], nat[7]);
        *reinterpret_cast<uint32_t*>(s + (size_t) r * (cols / 16) + (size_t) b * 4) =
            *reinterpret_cast<const uint32_t*>(tb + 256 + g * 4);
    }
}

__device__ __forceinline__ float act(const float* x, int64_t i, int relu2) {
    float v = x[i];
    if (relu2) {
        v = fmaxf(v, 0.0f);
        v *= v;
    }
    return v;
}

/// One block per row: amax, the global scale, then per 16 values the e4m3 scale and the e2m1 codes.
__global__ void quant_rows_kernel(const float* __restrict__ x, int64_t ldx, const int32_t* __restrict__ gather,
                                  int64_t K, int relu2, uint8_t* __restrict__ q, uint8_t* __restrict__ s,
                                  float* __restrict__ gs) {
    const int64_t r = blockIdx.x;
    const float* xr = x + (size_t) (gather ? gather[r] : r) * ldx;
    __shared__ float red[32];
    float m = 0.0f;
    for (int64_t c = threadIdx.x; c < K; c += blockDim.x) m = fmaxf(m, fabsf(act(xr, c, relu2)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float g = red[0] > 0.0f ? red[0] / (448.0f * 6.0f) : 1.0f;
    if (threadIdx.x == 0) gs[r] = g;
    for (int64_t b = threadIdx.x; b < K / 16; b += blockDim.x) {
        float v[16], am = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = act(xr, b * 16 + j, relu2);
            am = fmaxf(am, fabsf(v[j]));
        }
        const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(am / (6.0f * g), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 sv;
        sv.__x = s8;
        const float sb = (float) sv * g;
        const float inv = sb > 0.0f ? 1.0f / sb : 0.0f;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t c = (uint32_t) __nv_cvt_float_to_fp4(v[j] * inv, __NV_E2M1, cudaRoundNearest) & 15u;
            if (j < 8) w0 |= c << (4 * j);
            else w1 |= c << (4 * (j - 8));
        }
        *reinterpret_cast<uint2*>(q + (size_t) r * (K / 2) + (size_t) b * 8) = make_uint2(w0, w1);
        s[(size_t) r * (K / 16) + b] = s8;
    }
}

/// The same quantization, one pass: TPR threads a row hold its 16-value blocks in registers (VPT blocks each, rows
/// of K <= 16 * TPR * VPT), amax over them, then the codes from the registers.  256 / TPR rows a block.
template <int TPR, int VPT, class XT>
__global__ void __launch_bounds__(256) quant_rows_reg_kernel(const XT* __restrict__ x, int64_t ldx,
                                                             const int32_t* __restrict__ gather, int64_t M, int64_t K,
                                                             int relu2, uint8_t* __restrict__ q,
                                                             uint8_t* __restrict__ s, float* __restrict__ gs) {
    __shared__ float red[256 / 32];
    const int sub = threadIdx.x % TPR, rib = threadIdx.x / TPR;
    const int64_t r = (int64_t) blockIdx.x * (256 / TPR) + rib;
    const bool live = r < M;
    const XT* xr = x + (size_t) (live ? (gather ? gather[r] : r) : 0) * ldx;
    const int nb = (int) (K / 16);
    float v[VPT][16];
    float am = 0.0f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int b = sub + i * TPR;
        if (live && b < nb) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float4 u;
                if (sizeof(XT) == 4) {
                    u = reinterpret_cast<const float4*>(xr + (size_t) b * 16)[j];
                } else {   // BF16: 4 values a uint2
                    const uint2 h = reinterpret_cast<const uint2*>(xr + (size_t) b * 16)[j];
                    u = make_float4(__uint_as_float(h.x << 16), __uint_as_float(h.x & 0xffff0000u),
                                    __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xffff0000u));
                }
                if (relu2) {
                    u.x = fmaxf(u.x, 0.0f); u.x *= u.x; u.y = fmaxf(u.y, 0.0f); u.y *= u.y;
                    u.z = fmaxf(u.z, 0.0f); u.z *= u.z; u.w = fmaxf(u.w, 0.0f); u.w *= u.w;
                }
                v[i][4 * j] = u.x; v[i][4 * j + 1] = u.y; v[i][4 * j + 2] = u.z; v[i][4 * j + 3] = u.w;
                am = fmaxf(am, fmaxf(fmaxf(fabsf(u.x), fabsf(u.y)), fmaxf(fabsf(u.z), fabsf(u.w))));
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) v[i][j] = 0.0f;
        }
    }
    // the row's amax: a warp's lanes (TPR = 32), or the row's warps through shared memory
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if (TPR > 32) {
        if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = am;
        __syncthreads();
        am = 0.0f;
#pragma unroll
        for (int w = 0; w < TPR / 32; ++w) am = fmaxf(am, red[rib * (TPR / 32) + w]);
    }
    if (!live) return;
    const float g = am > 0.0f ? am / (448.0f * 6.0f) : 1.0f;
    if (sub == 0) gs[r] = g;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int b = sub + i * TPR;
        if (b >= nb) continue;
        float bm = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) bm = fmaxf(bm, fabsf(v[i][j]));
        const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(bm / (6.0f * g), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 sv;
        sv.__x = s8;
        const float sb = (float) sv * g;
        const float inv = sb > 0.0f ? 1.0f / sb : 0.0f;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t c = (uint32_t) __nv_cvt_float_to_fp4(v[i][j] * inv, __NV_E2M1, cudaRoundNearest) & 15u;
            if (j < 8) w0 |= c << (4 * j);
            else w1 |= c << (4 * (j - 8));
        }
        *reinterpret_cast<uint2*>(q + (size_t) r * (K / 2) + (size_t) b * 8) = make_uint2(w0, w1);
        s[(size_t) r * (K / 16) + b] = s8;
    }
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp4(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma_fp4(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
#ifdef NEMO_FP4_MMA
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"((uint16_t) 0),
                   "h"((uint16_t) 0), "r"(sb), "h"((uint16_t) 0), "h"((uint16_t) 0));
#endif
}

/// The swizzled byte offset of (row, 16-byte chunk) in a stage's operand.
/// (8 consecutive rows of one chunk - an ldmatrix - land on the 8 distinct 16-byte bank groups)
__device__ __forceinline__ int swz(int row, int chunk) {
    const int x = CH == 2 ? (row >> 2) & 1 : CH == 4 ? (row >> 1) & 3 : row & 7;
    return row * 32 * KSUB + ((chunk ^ x) << 4);
}

/// Rows [r0, r1) of A (absolute), columns [n0, n0 + 128) of B (b already at the problem's matrix).
/// GG: B straight from ggml block_nvfp4 rows at bg (no N4 copy): each thread takes one (row, 64-value block) of a stage
/// into registers a stage ahead, reorders its nibbles and stores it in the stage's N4 layout.
template <bool GG>
__device__ __forceinline__ void gemm_tile(const uint8_t* __restrict__ bg, const uint8_t* __restrict__ aq,
                                          const uint8_t* __restrict__ as,
                                          const float* __restrict__ ags, const uint8_t* __restrict__ bq,
                                          const uint8_t* __restrict__ bs, float alpha, float* __restrict__ y,
                                          int64_t ldy, int64_t r0, int64_t r1, int64_t n0, int64_t K, int relu2,
                                          int accumulate, uint8_t* sm, uint16_t* __restrict__ yb = nullptr,
                                          const int32_t* __restrict__ yrow = nullptr,
                                          const int32_t* __restrict__ arow = nullptr) {
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / 4, wn = warp % 4;   // warp tile: rows wm * 64, columns wn * 32
    const int KB = (int) (K / 64);
    const int64_t qrow = K / 2, srow = K / 16;
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);

    // arow: A's rows (and their global scales) at arow[r] - the tokens' latent rows quantized once, not per expert;
    // a thread's A rows are the same every stage, so their row indices are read once
    int64_t arw[CH / 2 > 0 ? CH / 2 : 1], ars = 0;
#pragma unroll
    for (int i = 0; i < CH / 2; ++i) {
        const int64_t r = r0 + (tid + i * NTH) / CH;
        arw[i] = r < r1 ? (arow ? arow[r] : r) : (arow ? arow[r0] : r0);
    }
    {
        const int64_t r = r0 + tid % 128;
        ars = r < r1 ? (arow ? arow[r] : r) : (arow ? arow[r0] : r0);
    }
    // loader: a stage is KSUB 64-value steps; per row CH 16-byte chunks of values and KSUB 4-byte scale words
    auto load = [&](int kb0, int stage) {
        const uint32_t st = sbase + stage * STAGE;
#pragma unroll
        for (int i = 0; i < CH / 2; ++i) {
            const int c = tid + i * NTH, r = c / CH, ch = c % CH, kb = kb0 + ch / 2;
            const bool kin = kb < KB;
            const bool ain = kin && r0 + r < r1;
            cp16(st + swz(r, ch), aq + (size_t) arw[i] * qrow + (size_t) (kin ? kb : 0) * 32 + (ch & 1) * 16,
                 ain ? 16 : 0);
            if (!GG)
                cp16(st + QB + swz(r, ch), bq + (size_t) (n0 + r) * qrow + (size_t) (kin ? kb : 0) * 32 + (ch & 1) * 16,
                     kin ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < KSUB; ++i) {
            const int r = tid % 128, kb = kb0 + i;
            const bool kin = kb < KB;
            if (tid < 128) {
                const bool ain = kin && r0 + r < r1;
                cp4(st + 2 * QB + (r * KSUB + i) * 4, as + (size_t) ars * srow + (size_t) (kin ? kb : 0) * 4,
                    ain ? 4 : 0);
            } else if (!GG) {
                cp4(st + 2 * QB + SB + (r * KSUB + i) * 4, bs + (size_t) (n0 + r) * srow + (size_t) (kin ? kb : 0) * 4,
                    kin ? 4 : 0);
            }
        }
    };
    const int NS = (KB + KSUB - 1) / KSUB;   // stages
    // GG: thread -> (row tid / 2, block tid % 2 of the stage); its 36 bytes in registers a stage ahead
    static_assert(!GG || (KSUB == 2 && NTH == 2 * BN), "the ggml B loader is written for 2 blocks a stage");
    uint32_t gw[9];
    const int gr = tid >> 1, gbk = tid & 1;
    auto gload = [&](int ks) {
        const int kb = ks * KSUB + gbk;
        if (kb < KB) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(bg + ((size_t) (n0 + gr) * KB + kb) * 36);
#pragma unroll
            for (int i = 0; i < 9; ++i) gw[i] = src[i];
        } else {
#pragma unroll
            for (int i = 0; i < 9; ++i) gw[i] = 0u;
        }
    };
    auto gstore = [&](int stage) {
        uint8_t* stg = sm + stage * STAGE;
        uint32_t nat[8];
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const uint2 v = natural16((uint64_t) gw[1 + 2 * sb] | ((uint64_t) gw[2 + 2 * sb] << 32));
            nat[2 * sb] = v.x;
            nat[2 * sb + 1] = v.y;
        }
        *reinterpret_cast<uint4*>(stg + QB + swz(gr, 2 * gbk)) = make_uint4(nat[0], nat[1], nat[2], nat[3]);
        *reinterpret_cast<uint4*>(stg + QB + swz(gr, 2 * gbk + 1)) = make_uint4(nat[4], nat[5], nat[6], nat[7]);
        *reinterpret_cast<uint32_t*>(stg + 2 * QB + SB + (gr * KSUB + gbk) * 4) = gw[0];
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll
    for (int p = 0; p < ST - 1; ++p) {
        if (p < NS) load(p * KSUB, p);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    if (GG) {
        gload(0);
        gstore(0);
        if (NS > 1) gload(1);
    }
    for (int ks = 0; ks < NS; ++ks) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
        __syncthreads();
        if (ks + ST - 1 < NS) load((ks + ST - 1) * KSUB, (ks + ST - 1) % ST);
        asm volatile("cp.async.commit_group;\n" ::);
        const int stage = ks % ST;
        const uint32_t st = sbase + stage * STAGE;
        const uint8_t* ssa = sm + stage * STAGE + 2 * QB;
        const uint8_t* ssb = ssa + SB;
#pragma unroll
        for (int u = 0; u < KSUB; ++u) {
            uint32_t af[4][4], bf[4][2], sfa[4], sfb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = wm * 64 + i * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                ldsm4(af[i], st + swz(row, 2 * u + lane / 16));
                sfa[i] = *reinterpret_cast<const uint32_t*>(ssa + ((wm * 64 + i * 16 + g + 8 * (t & 1)) * KSUB + u) * 4);
            }
#pragma unroll
            for (int jp = 0; jp < 2; ++jp) {
                const int row = wn * 32 + jp * 16 + (lane % 8) + (lane / 16) * 8;
                uint32_t r[4];
                ldsm4(r, st + QB + swz(row, 2 * u + ((lane / 8) & 1)));
                bf[2 * jp][0] = r[0];
                bf[2 * jp][1] = r[1];
                bf[2 * jp + 1][0] = r[2];
                bf[2 * jp + 1][1] = r[3];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                sfb[j] = *reinterpret_cast<const uint32_t*>(ssb + ((wn * 32 + j * 8 + g) * KSUB + u) * 4);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_fp4(acc[i][j], af[i], bf[j][0], bf[j][1], sfa[i], sfb[j]);
        }
        if (GG && ks + 1 < NS) {
            // the next stage's B (loaded during this one) into its N4 layout; the one after into the registers
            gstore((ks + 1) % ST);
            if (ks + 2 < NS) gload(ks + 2);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t r = r0 + wm * 64 + i * 16 + g + 8 * h;
            if (r >= r1) continue;
            const float sc = alpha * ags[arow ? arow[r] : r];
            const int64_t ro = yrow ? yrow[r] : r;   // (the output row: remapped for the experts' (t, j) rows)
            float* yr = y + (size_t) ro * ldy + n0 + wn * 32 + 2 * t;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v0 = acc[i][j][2 * h] * sc, v1 = acc[i][j][2 * h + 1] * sc;
                float2* p = reinterpret_cast<float2*>(yr + j * 8);
                if (accumulate) {
                    const float2 o = *p;
                    v0 += o.x;
                    v1 += o.y;
                }
                if (relu2) {
                    v0 = fmaxf(v0, 0.0f); v0 *= v0;
                    v1 = fmaxf(v1, 0.0f); v1 *= v1;
                }
                if (yb) {   // BF16 out (round to nearest even): the next GEMM's input, half the bytes
                    uint32_t u0 = __float_as_uint(v0), u1 = __float_as_uint(v1);
                    u0 = (u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16;
                    u1 = (u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16;
                    *reinterpret_cast<uint32_t*>(yb + (size_t) ro * ldy + n0 + wn * 32 + 2 * t + j * 8) = u0 | (u1 << 16);
                    continue;
                }
                *p = make_float2(v0, v1);
            }
        }
}

__global__ void __launch_bounds__(NTH) gemm_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as,
                                                   const float* __restrict__ ags, const uint8_t* __restrict__ bq,
                                                   const uint8_t* __restrict__ bs, float alpha, float* __restrict__ y,
                                                   int64_t ldy, int64_t M, int64_t K, int relu2, int accumulate) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int64_t r0 = (int64_t) blockIdx.y * BM;
    gemm_tile<false>(nullptr, aq, as, ags, bq, bs, alpha, y, ldy, r0, r0 + BM < M ? r0 + BM : M,
                     (int64_t) blockIdx.x * BN, K, relu2, accumulate, sm);
}

__global__ void __launch_bounds__(NTH) gemm_grouped_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as,
                                                           const float* __restrict__ ags, const uint8_t* __restrict__ bq,
                                                           const uint8_t* __restrict__ bs, const float* __restrict__ alpha,
                                                           const int32_t* __restrict__ tiles, float* __restrict__ y,
                                                           int64_t N, int64_t K, int relu2, uint16_t* __restrict__ yb,
                                                           const int32_t* __restrict__ yrow,
                                                           const uint8_t* const* __restrict__ bptr,
                                                           const int32_t* __restrict__ arow) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int p = tiles[3 * blockIdx.y], r0 = tiles[3 * blockIdx.y + 1], r1 = tiles[3 * blockIdx.y + 2];
    if (bptr)
        gemm_tile<true>(bptr[p], aq, as, ags, nullptr, nullptr, alpha[p], y, N, r0, r1, (int64_t) blockIdx.x * BN, K,
                        relu2, 0, sm, yb, yrow, arow);
    else
        gemm_tile<false>(nullptr, aq, as, ags, bq + (size_t) p * N * K / 2, bs + (size_t) p * N * K / 16, alpha[p], y,
                         N, r0, r1, (int64_t) blockIdx.x * BN, K, relu2, 0, sm, yb, yrow, arow);
}

bool g_attr = false;
void set_attr() {
    if (g_attr) return;
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ST * STAGE);
    cudaFuncSetAttribute(gemm_grouped_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ST * STAGE);
    g_attr = true;
}

}  // namespace

/// The register kernels for rows of up to 12,288 values (false: a longer row).
template <class XT>
bool quant_reg(const XT* x, int64_t ldx, const int32_t* gather, int64_t M, int64_t K, int rl, uint8_t* q, uint8_t* s,
               float* gs, cudaStream_t st) {
    const int nb = (int) (K / 16);
    if (nb <= 64) quant_rows_reg_kernel<32, 2, XT><<<(unsigned) ((M + 7) / 8), 256, 0, st>>>(x, ldx, gather, M, K, rl, q, s, gs);
    else if (nb <= 192) quant_rows_reg_kernel<32, 6, XT><<<(unsigned) ((M + 7) / 8), 256, 0, st>>>(x, ldx, gather, M, K, rl, q, s, gs);
    else if (nb <= 768) quant_rows_reg_kernel<128, 6, XT><<<(unsigned) ((M + 1) / 2), 256, 0, st>>>(x, ldx, gather, M, K, rl, q, s, gs);
    else return false;
    return true;
}

bool nemo_fp4_ok() {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, major = 0, minor = 0, has = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok = major == 12 && minor == 0 && cudaMemcpyFromSymbol(&has, g_has_mma, sizeof(int)) == cudaSuccess && has == 1;
        if (!ok) cudaGetLastError();
    }
    return ok == 1;
}

void nemo_fp4_quant_rows(const float* x, int64_t ldx, const int32_t* gather, int64_t M, int64_t K, bool relu2,
                         uint8_t* q, uint8_t* s, float* gs, void* stream) {
    if (M <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int rl = relu2 ? 1 : 0;
    if (ldx % 4 == 0 && ((uintptr_t) x & 15) == 0 && quant_reg<float>(x, ldx, gather, M, K, rl, q, s, gs, st)) return;
    quant_rows_kernel<<<(unsigned) M, 256, 0, st>>>(x, ldx, gather, K, rl, q, s, gs);
}

bool nemo_fp4_quant_rows_bf16(const uint16_t* x, int64_t ldx, int64_t M, int64_t K, uint8_t* q, uint8_t* s, float* gs,
                              void* stream) {
    if (M <= 0) return true;
    if (ldx % 8 || ((uintptr_t) x & 15)) return false;
    return quant_reg<uint16_t>(x, ldx, nullptr, M, K, 0, q, s, gs, (cudaStream_t) stream);
}

void nemo_fp4_from_ggml_many(const uint8_t* const* ptr, int64_t n, int64_t rows, int64_t cols, uint8_t* q, uint8_t* s,
                             void* stream) {
    if (n <= 0) return;
    const int64_t units = rows * (cols / 64);
    const unsigned bx = (unsigned) ((units + 255) / 256 < 512 ? (units + 255) / 256 : 512);
    from_ggml_kernel<<<dim3(bx, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(ptr, rows, cols, q, s);
}

void nemo_fp4_from_tiled(const uint8_t* w, int64_t rows, int64_t cols, uint8_t* q, uint8_t* s, void* stream) {
    const int64_t units = rows * (cols / 64);
    const unsigned bx = (unsigned) ((units + 255) / 256 < 8192 ? (units + 255) / 256 : 8192);
    from_tiled_kernel<<<bx, 256, 0, (cudaStream_t) stream>>>(w, (int) rows, (int) cols, q, s);
}

void nemo_fp4_gemm(const uint8_t* aq, const uint8_t* as, const float* ags, const uint8_t* bq, const uint8_t* bs,
                   float alpha, float* y, int64_t ldy, int64_t M, int64_t N, int64_t K, bool relu2, bool accumulate,
                   void* stream) {
    if (M <= 0) return;
    set_attr();
    gemm_kernel<<<dim3((unsigned) (N / BN), (unsigned) ((M + BM - 1) / BM)), NTH, ST * STAGE, (cudaStream_t) stream>>>(
        aq, as, ags, bq, bs, alpha, y, ldy, M, K, relu2 ? 1 : 0, accumulate ? 1 : 0);
}

void nemo_fp4_gemm_grouped(const uint8_t* aq, const uint8_t* as, const float* ags, const uint8_t* bq,
                           const uint8_t* bs, const float* alpha, const int32_t* tiles, int ntiles, float* y,
                           int64_t N, int64_t K, bool relu2, void* stream, uint16_t* yb, const int32_t* yrow,
                           const uint8_t* const* bptr, const int32_t* arow) {
    if (ntiles <= 0) return;
    set_attr();
    gemm_grouped_kernel<<<dim3((unsigned) (N / BN), (unsigned) ntiles), NTH, ST * STAGE, (cudaStream_t) stream>>>(
        aq, as, ags, bq, bs, alpha, tiles, y, N, K, relu2 ? 1 : 0, yb, yrow, bptr, arow);
}

}  // namespace strata::kernels
