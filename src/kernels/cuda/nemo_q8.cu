// src/kernels/cuda/nemo_q8.cu - see include/strata/kernels/nemo_q8.hpp.
//
// The GEMM: a block computes 128 rows (tokens) x 128 columns (weight rows) of Y; 8 warps as 2 x 4, a warp 64 x 32 =
// 4 m16 x 4 n8 tiles; K in stages of 64 values (two 32-value blocks) through a 4-stage cp.async pipeline.  A stage is
// 128 rows x 64 bytes of each operand (four 16-byte chunks a row, swizzled by bits 1-2 of the row so ldmatrix's 8 rows
// land on distinct banks) and the rows' two scales.  The s8 m16n8k32 fragments are b16 ldmatrix loads: thread (g, t)
// gets row g's bytes 4t .. 4t+3 of a 16-byte chunk, which is A's reg 0 (row g, k 4t..) / 1 (row g + 8) / 2 / 3 (k 16 + 4t..)
// and B's reg 0 (column g, k 4t..) / 1 (k 16 + 4t..).  Per 32-value block the int32 sum of an mma is exact; it goes
// into the FP32 accumulator times the row's and the column's scales.
#include "strata/kernels/nemo_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

#ifndef NEMO_Q8_KS
#define NEMO_Q8_KS 2
#endif
#ifndef NEMO_Q8_ST
#define NEMO_Q8_ST 2
#endif
constexpr int BM = 128, BN = 128, NTH = 256, KS = NEMO_Q8_KS, ST = NEMO_Q8_ST;   ///< 64-value steps a stage, stages
constexpr int CH = 4 * KS, RB = 64 * KS;             ///< 16-byte chunks, bytes of a stage row
constexpr int QB = BM * RB, SB = BM * 8 * KS;      // a stage's bytes of one operand: values, scales
constexpr int STAGE = 2 * QB + 2 * SB;
constexpr int MAGIC = 0x4B400000;   ///< the bits of 12582912.0f

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp8(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/// The swizzled byte offset of (row, 16-byte chunk) in a stage's operand.
__device__ __forceinline__ int swz(int row, int chunk) {
    return row * RB + ((chunk ^ (CH == 4 ? (row >> 1) & 3 : row & 7)) << 4);
}

__global__ void __launch_bounds__(NTH) gemm_kernel(const int8_t* __restrict__ aq, const float* __restrict__ as,
                                                   const int8_t* __restrict__ bq, const float* __restrict__ bs,
                                                   float alpha, float* __restrict__ y, int64_t ldy, int64_t M,
                                                   int64_t K, int accumulate) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / 4, wn = warp % 4;   // warp tile: rows wm * 64, columns wn * 32
    const int64_t r0 = (int64_t) blockIdx.y * BM, n0 = (int64_t) blockIdx.x * BN;
    const int64_t r1 = r0 + BM < M ? r0 + BM : M;
    const int NS = (int) (K / (64 * KS)), KB = (int) (K / 32);
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);

    auto load = [&](int ks, int stage) {
        const uint32_t st = sbase + stage * STAGE;
#pragma unroll
        for (int i = 0; i < CH / 2; ++i) {
            const int c = tid + i * NTH, r = c / CH, ch = c % CH;
            const bool ain = r0 + r < r1;
            cp16(st + swz(r, ch), aq + (size_t) (ain ? r0 + r : r0) * K + (size_t) ks * RB + ch * 16, ain ? 16 : 0);
            cp16(st + QB + swz(r, ch), bq + (size_t) (n0 + r) * K + (size_t) ks * RB + ch * 16, 16);
        }
        const int r = tid % 128;
#pragma unroll
        for (int i = 0; i < KS; ++i) {
            if (tid < 128) {
                const bool ain = r0 + r < r1;
                cp8(st + 2 * QB + (r * KS + i) * 8, as + (size_t) (ain ? r0 + r : r0) * KB + (ks * KS + i) * 2, ain ? 8 : 0);
            } else {
                cp8(st + 2 * QB + SB + (r * KS + i) * 8, bs + (size_t) (n0 + r) * KB + (ks * KS + i) * 2, 8);
            }
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll
    for (int p = 0; p < ST - 1; ++p) {
        if (p < NS) load(p, p);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ks = 0; ks < NS; ++ks) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
        __syncthreads();
        if (ks + ST - 1 < NS) load(ks + ST - 1, (ks + ST - 1) % ST);
        asm volatile("cp.async.commit_group;\n" ::);
        const int stage = ks % ST;
        const uint32_t st = sbase + stage * STAGE;
        const float* ssa = reinterpret_cast<const float*>(sm + stage * STAGE + 2 * QB);
        const float* ssb = reinterpret_cast<const float*>(sm + stage * STAGE + 2 * QB + SB);
#pragma unroll
        for (int u = 0; u < 2 * KS; ++u) {
            uint32_t af[4][4], bf[4][2];
            float sa[4][2], sb[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = wm * 64 + i * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                ldsm4(af[i], st + swz(row, 2 * u + lane / 16));
                sa[i][0] = ssa[(wm * 64 + i * 16 + g) * 2 * KS + u];
                sa[i][1] = ssa[(wm * 64 + i * 16 + g + 8) * 2 * KS + u];
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
            for (int j = 0; j < 4; ++j) {
                sb[j][0] = ssb[(wn * 32 + j * 8 + 2 * t) * 2 * KS + u];
                sb[j][1] = ssb[(wn * 32 + j * 8 + 2 * t + 1) * 2 * KS + u];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // the sums start at the bits of 2^23 + 2^22: a block's |sum| <= 32 * 127 * 128 < 2^22, so the
                    // result read as a float is 12582912 + sum exactly (no int -> float conversion, a quarter-rate op)
                    int c[4] = {MAGIC, MAGIC, MAGIC, MAGIC};
                    mma_s8(c, af[i], bf[j][0], bf[j][1]);
                    acc[i][j][0] = fmaf((__int_as_float(c[0]) - 12582912.0f) * sb[j][0], sa[i][0], acc[i][j][0]);
                    acc[i][j][1] = fmaf((__int_as_float(c[1]) - 12582912.0f) * sb[j][1], sa[i][0], acc[i][j][1]);
                    acc[i][j][2] = fmaf((__int_as_float(c[2]) - 12582912.0f) * sb[j][0], sa[i][1], acc[i][j][2]);
                    acc[i][j][3] = fmaf((__int_as_float(c[3]) - 12582912.0f) * sb[j][1], sa[i][1], acc[i][j][3]);
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t r = r0 + wm * 64 + i * 16 + g + 8 * h;
            if (r >= r1) continue;
            float* yr = y + (size_t) r * ldy + n0 + wn * 32 + 2 * t;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v0 = acc[i][j][2 * h] * alpha, v1 = acc[i][j][2 * h + 1] * alpha;
                float2* p = reinterpret_cast<float2*>(yr + j * 8);
                if (accumulate) {
                    const float2 o = *p;
                    v0 += o.x;
                    v1 += o.y;
                }
                *p = make_float2(v0, v1);
            }
        }
}

/// A thread per (row, 32-value block).
__global__ void quant_rows_kernel(const float* __restrict__ x, int64_t ldx, int64_t M, int64_t K,
                                  int8_t* __restrict__ q, float* __restrict__ s) {
    const int64_t KB = K / 32, units = M * KB;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = u / KB, b = u % KB;
        const float4* src = reinterpret_cast<const float4*>(x + (size_t) r * ldx + (size_t) b * 32);
        float v[32];
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 f = src[i];
            v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
            a = fmaxf(a, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
        }
        const float d = a / 127.0f, inv = a > 0.0f ? 127.0f / a : 0.0f;
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t p = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) p |= ((uint32_t) (__float2int_rn(v[4 * i + k] * inv) & 0xff)) << (8 * k);
            w[i] = p;
        }
        uint4* dq = reinterpret_cast<uint4*>(q + (size_t) r * K + (size_t) b * 32);
        dq[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dq[1] = make_uint4(w[4], w[5], w[6], w[7]);
        s[(size_t) r * KB + b] = d;
    }
}

/// Tiled Q8_0: tile (T, b) of 272 bytes holds, for lane (g, t), row g's bytes 16h + 2t, +1, 16h + 8 + 2t, +1 at
/// lane * 8 + 4h, then the 8 rows' FP16 scales.  A thread per (row, block).
__global__ void from_tiled_q8_kernel(const uint8_t* __restrict__ w, int rows, int cols, int8_t* __restrict__ q,
                                     float* __restrict__ s) {
    const int nb = cols / 32;
    const int64_t units = (int64_t) rows * nb;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int r = (int) (u / nb), b = (int) (u % nb), T = r / 8, g = r % 8;
        const uint8_t* tb = w + ((size_t) T * nb + b) * 272;
        uint8_t v[32];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint2 p = *reinterpret_cast<const uint2*>(tb + (g * 4 + t) * 8);
            const uint32_t pp[2] = {p.x, p.y};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                v[16 * h + 2 * t] = (uint8_t) (pp[h] & 0xff);
                v[16 * h + 2 * t + 1] = (uint8_t) ((pp[h] >> 8) & 0xff);
                v[16 * h + 8 + 2 * t] = (uint8_t) ((pp[h] >> 16) & 0xff);
                v[16 * h + 8 + 2 * t + 1] = (uint8_t) (pp[h] >> 24);
            }
        }
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            o[i] = (uint32_t) v[4 * i] | ((uint32_t) v[4 * i + 1] << 8) | ((uint32_t) v[4 * i + 2] << 16) |
                   ((uint32_t) v[4 * i + 3] << 24);
        uint4* dq = reinterpret_cast<uint4*>(q + (size_t) r * cols + (size_t) b * 32);
        dq[0] = make_uint4(o[0], o[1], o[2], o[3]);
        dq[1] = make_uint4(o[4], o[5], o[6], o[7]);
        s[(size_t) r * nb + b] = __half2float(*reinterpret_cast<const __half*>(tb + 256 + g * 2));
    }
}

/// Tiled BF16: tile (T, b) of 256 bytes holds, for lane (g, t), row g's values 2t, 2t + 1 (bytes 0..3) and 8 + 2t,
/// 9 + 2t (4..7) of the 16-value block b at lane * 8.  A thread per (row, 32-value block): two tiles' blocks.
__global__ void from_tiled_bf16_kernel(const uint8_t* __restrict__ w, int rows, int cols, int8_t* __restrict__ q,
                                       float* __restrict__ s) {
    const int nb16 = cols / 16, nb = cols / 32;
    const int64_t units = (int64_t) rows * nb;
    for (int64_t u = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (int64_t) gridDim.x * blockDim.x) {
        const int r = (int) (u / nb), b = (int) (u % nb), T = r / 8, g = r % 8;
        float v[32];
        float a = 0.0f;
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const uint8_t* tb = w + ((size_t) T * nb16 + 2 * b + hb) * 256;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint2 p = *reinterpret_cast<const uint2*>(tb + (g * 4 + t) * 8);
                const int k = 16 * hb;
                v[k + 2 * t] = __uint_as_float(p.x << 16);
                v[k + 2 * t + 1] = __uint_as_float(p.x & 0xffff0000u);
                v[k + 8 + 2 * t] = __uint_as_float(p.y << 16);
                v[k + 9 + 2 * t] = __uint_as_float(p.y & 0xffff0000u);
            }
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) a = fmaxf(a, fabsf(v[i]));
        const float d = a / 127.0f, inv = a > 0.0f ? 127.0f / a : 0.0f;
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t p = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) p |= ((uint32_t) (__float2int_rn(v[4 * i + k] * inv) & 0xff)) << (8 * k);
            o[i] = p;
        }
        uint4* dq = reinterpret_cast<uint4*>(q + (size_t) r * cols + (size_t) b * 32);
        dq[0] = make_uint4(o[0], o[1], o[2], o[3]);
        dq[1] = make_uint4(o[4], o[5], o[6], o[7]);
        s[(size_t) r * nb + b] = d;
    }
}

unsigned grid_for(int64_t units) { return (unsigned) ((units + 255) / 256 < 8192 ? (units + 255) / 256 : 8192); }

}  // namespace

void nemo_q8_quant_rows(const float* x, int64_t ldx, int64_t M, int64_t K, int8_t* q, float* s, void* stream) {
    if (M <= 0) return;
    quant_rows_kernel<<<grid_for(M * (K / 32)), 256, 0, (cudaStream_t) stream>>>(x, ldx, M, K, q, s);
}

void nemo_q8_from_tiled_q8(const uint8_t* w, int64_t rows, int64_t cols, int8_t* q, float* s, void* stream) {
    from_tiled_q8_kernel<<<grid_for(rows * (cols / 32)), 256, 0, (cudaStream_t) stream>>>(w, (int) rows, (int) cols, q, s);
}

void nemo_q8_from_tiled_bf16(const uint8_t* w, int64_t rows, int64_t cols, int8_t* q, float* s, void* stream) {
    from_tiled_bf16_kernel<<<grid_for(rows * (cols / 32)), 256, 0, (cudaStream_t) stream>>>(w, (int) rows, (int) cols, q,
                                                                                             s);
}

void nemo_q8_gemm(const int8_t* aq, const float* as, const int8_t* bq, const float* bs, float alpha, float* y,
                  int64_t ldy, int64_t M, int64_t N, int64_t K, bool accumulate, void* stream) {
    if (M <= 0) return;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ST * STAGE);
        attr = true;
    }
    gemm_kernel<<<dim3((unsigned) (N / BN), (unsigned) ((M + BM - 1) / BM)), NTH, ST * STAGE, (cudaStream_t) stream>>>(
        aq, as, bq, bs, alpha, y, ldy, M, K, accumulate ? 1 : 0);
}

}  // namespace strata::kernels
