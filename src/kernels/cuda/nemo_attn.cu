// src/kernels/cuda/nemo_attn.cu - strata-nemotron stage 3: causal NoPE GQA.  See nemo_attn.hpp.
//
// The cache (NemoKV): K int8 with a scale per (cell, KV head), V FP16, in pages.
//
// Decode and the verify window (T <= 8): split-K on CUDA cores.  Block (split, kv head) of 16 warps = the 16 query
// heads of that KV head, every query of the call, so a key row is fetched once and read by all of them.  Lane l
// holds dims 4l .. 4l+3 of q and of the running output; one key is one 4-wide dot and a warp reduction, then the
// online-softmax update.  A second kernel merges a head's splits: o = sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s.
// A token's result does not depend on T (the verify window computes what decode would).
//
// The prompt (T > 8): flash attention on tensor cores.  Warp w of a block = query t0 + w: its 16 MMA rows are that
// query's 16 heads over KV head blockIdx.y (GQA: they share every key, and one causal limit, p0 + t).  Keys go through
// shared memory in tiles of 64, double buffered (cp.async).  S = Q K^T on int8 tensor cores (Q quantized per (query,
// head) in registers, K's scale per key applied to S's accumulator; m16n8k32, exact int32 sums); the online softmax in
// base 2; P (FP16) times V (FP16) with FP16 sums over a tile's 64 keys, added to FP32 accumulators.  Against FP32 math
// the error is ~3e-3 relative (BF16 everything: ~1.3e-3), measured on layer 7 (a NumPy emulation of the kernel and
// attn_parity).
#include "strata/kernels/nemo_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 128;   ///< head dim the kernels are written for (4 per lane)
/// decode and the verify window over at least this many keys: tensor cores (int8 Q too), below it FP32 Q on CUDA cores
const int64_t NEMO_ATTN_TC_DECODE_KEYS =
    std::getenv("STRATA_NEMO_ATTN_TC_KEYS") ? std::atoll(std::getenv("STRATA_NEMO_ATTN_TC_KEYS")) : 16384;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ size_t cell_of(int shift, int64_t j) { return (size_t) (j & ((1ll << shift) - 1)); }

__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

/// The split-K pieces: 256 keys each (doubled past 2,048 pieces), so a split's keys do not depend on T and the
/// verify window sums exactly what decode sums, piece by piece.
int64_t split_len_for(int64_t n_keys) {
    int64_t len = 256;
    while ((n_keys + len - 1) / len > 2048) len *= 2;
    return len;
}

int n_splits_for(const NemoAttnShapes& s, int64_t T, int64_t n_keys) {
    (void) s;
    (void) T;
    const int64_t n = (n_keys + split_len_for(n_keys) - 1) / split_len_for(n_keys);
    return (int) (n < 1 ? 1 : n);
}

/// Block per token, a thread per (head, dim): K's amax per head (4 warps of 32 dims), int8 + scale; V to FP16.
__global__ void kv_append_kernel(NemoKV kv, const float* __restrict__ k, const float* __restrict__ v, int64_t p0,
                                 int n_kv) {
    __shared__ float red[32];
    const int t = blockIdx.x, i = threadIdx.x, h = i / HD, width = n_kv * HD;
    const int64_t j = p0 + t;
    const size_t c = cell_of(kv.shift, j);
    const float kx = k[(size_t) t * width + i];
    float a = fabsf(kx);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if (i % 32 == 0) red[i / 32] = a;
    __syncthreads();
    const float am = fmaxf(fmaxf(red[h * 4], red[h * 4 + 1]), fmaxf(red[h * 4 + 2], red[h * 4 + 3]));
    const float sc = am / 127.0f;
    const int qv = sc > 0.0f ? __float2int_rn(kx / sc) : 0;
    kv.k[j >> kv.shift][c * width + i] = (int8_t) max(-127, min(127, qv));
    if (i % HD == 0) kv.ks[j >> kv.shift][c * n_kv + h] = sc;
    const float vx = fminf(fmaxf(v[(size_t) t * width + i], -65504.0f), 65504.0f);
    kv.v[j >> kv.shift][c * width + i] = __half_as_ushort(__float2half_rn(vx));
}

/// Block (split, kv head) of 16 warps = the 16 query heads of that KV head, every query of the call (T <= 8): a key
/// row is read once and used by all of them.
template <int TM>
__global__ void attn_split_kernel(const float* __restrict__ q, const NemoKV kv, float* __restrict__ pacc,
                                  float* __restrict__ pm, float* __restrict__ pl, int64_t p0, int T, int n_head,
                                  int n_kv, int split_len, int n_splits, float scale, int64_t lo) {
    const int split = blockIdx.x, kvh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = n_head / n_kv;
    const int h = kvh * group + warp;
    const int64_t j0 = lo + (int64_t) split * split_len;
    int64_t j1 = j0 + split_len;
    if (j1 > p0 + T) j1 = p0 + T;                        // query t sees cells 0 .. p0+t

    float4 qv[TM];
    float m[TM], l[TM], a[TM][4];
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        qv[t] = t < T ? reinterpret_cast<const float4*>(q + ((size_t) t * n_head + h) * HD)[lane] : make_float4(0, 0, 0, 0);
        m[t] = -INFINITY;
        l[t] = 0.0f;
        a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
    }
    const int row = n_kv * HD;
    for (int64_t j = j0; j < j1; ++j) {
        const int pg = (int) (j >> kv.shift);
        const size_t c = cell_of(kv.shift, j);
        const uint32_t kr = reinterpret_cast<const uint32_t*>(kv.k[pg] + c * row + (size_t) kvh * HD)[lane];
        const float ksc = kv.ks[pg][c * n_kv + kvh] * scale;
        const uint2 vr = reinterpret_cast<const uint2*>(kv.v[pg] + c * row + (size_t) kvh * HD)[lane];
        const float k0 = (float) (int8_t) (kr & 0xff), k1 = (float) (int8_t) ((kr >> 8) & 0xff);
        const float k2 = (float) (int8_t) ((kr >> 16) & 0xff), k3 = (float) (int8_t) (kr >> 24);
        const float v0 = h2f((uint16_t) vr.x), v1 = h2f((uint16_t) (vr.x >> 16));
        const float v2 = h2f((uint16_t) vr.y), v3 = h2f((uint16_t) (vr.y >> 16));
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            if (t >= T || j > p0 + t) continue;
            const float s = warp_sum(qv[t].x * k0 + qv[t].y * k1 + qv[t].z * k2 + qv[t].w * k3) * ksc;
            const float mn = fmaxf(m[t], s);
            const float corr = expf(m[t] - mn);           // m = -inf on the first key: corr = 0, nothing to rescale
            const float p = expf(s - mn);
            l[t] = l[t] * corr + p;
            a[t][0] = a[t][0] * corr + p * v0;
            a[t][1] = a[t][1] * corr + p * v1;
            a[t][2] = a[t][2] * corr + p * v2;
            a[t][3] = a[t][3] * corr + p * v3;
            m[t] = mn;
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        if (t >= T) continue;
        const size_t slot = ((size_t) t * n_splits + split) * n_head + h;
        reinterpret_cast<float4*>(pacc + slot * HD)[lane] = make_float4(a[t][0], a[t][1], a[t][2], a[t][3]);
        if (lane == 0) {
            pm[slot] = m[t];
            pl[slot] = l[t];
        }
    }
}

__global__ void attn_merge_kernel(const float* __restrict__ pacc, const float* __restrict__ pm,
                                  const float* __restrict__ pl, float* __restrict__ o, int n_head, int n_splits) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    float M = -INFINITY;
    for (int sp = 0; sp < n_splits; ++sp) M = fmaxf(M, pm[((size_t) t * n_splits + sp) * n_head + h]);
    float l = 0.0f;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int sp = 0; sp < n_splits; ++sp) {
        const size_t slot = ((size_t) t * n_splits + sp) * n_head + h;
        const float ms = pm[slot];
        if (ms == -INFINITY) continue;                    // a split past the query's last key
        const float w = expf(ms - M);
        l += w * pl[slot];
        const float4 a = reinterpret_cast<const float4*>(pacc + slot * HD)[lane];
        acc.x += w * a.x; acc.y += w * a.y; acc.z += w * a.z; acc.w += w * a.w;
    }
    const float inv = 1.0f / l;
    reinterpret_cast<float4*>(o + ((size_t) t * n_head + h) * HD)[lane] =
        make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

// ---- the prompt's form (see the top) ----
#ifndef NEMO_ATTN_AKT
#define NEMO_ATTN_AKT 128
#endif
constexpr int AKT = NEMO_ATTN_AKT;
   ///< keys a tile
constexpr int NTL = AKT / 8, KSL = AKT / 16;   ///< a tile's n-tiles of S, k-steps of P V
constexpr int KP = HD;           ///< a shared K row, bytes (16-byte chunks XOR-swizzled by row & 7: ldmatrix's 8 rows
constexpr int VP = HD;           ///< a shared V row, halves      land on distinct banks without padding)
#ifndef NEMO_ATTN_MINB
#define NEMO_ATTN_MINB 1
#endif
#ifndef NEMO_ATTN_NW
#define NEMO_ATTN_NW 8
#endif
__device__ __forceinline__ int sw16(int row, int chunk) { return (chunk ^ (row & 7)) * 16; }

__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(src));
}

__device__ __forceinline__ uint32_t pack_s8(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) d << 24);
}


/// SPLIT = false: block (8 queries, kv head) over all their keys, o normalized.  SPLIT = true (decode and the verify
/// window at long context): block (split, kv head) for every query (T <= NW) over keys [split * split_len, ..),
/// unnormalized partial sums into the split-K scratch (pacc, pm in natural-log units, pl) for attn_merge_kernel.
template <int NW, bool SPLIT>
__global__ void __launch_bounds__(NW * 32, NEMO_ATTN_MINB) attn_tc_kernel(const float* __restrict__ q, const NemoKV kv,
                                                         float* __restrict__ o, int64_t p0, int T, int n_head,
                                                         int n_kv, float qscale, int split_len, int n_splits,
                                                         float* __restrict__ pacc, float* __restrict__ pm,
                                                         float* __restrict__ pl, int64_t lo) {
    extern __shared__ __align__(16) uint8_t sm[];
    int8_t* sk = (int8_t*) sm;                                  // [2][AKT][KP]
    uint16_t* sv = (uint16_t*) (sm + 2 * AKT * KP);             // [2][AKT][VP]
    float* sks = (float*) (sm + 2 * AKT * KP + 2 * AKT * VP * 2);   // [2][AKT]
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, tq = lane % 4;
    const int kvh = blockIdx.y, t0 = SPLIT ? 0 : blockIdx.x * NW, t = t0 + warp;
    const bool live = t < T;
    const int t_last = min(t0 + NW, T) - 1;
    // this block's keys [j0, n_keys); this warp's last key
    const int64_t j0 = lo + (SPLIT ? (int64_t) blockIdx.x * split_len : 0);
    const int64_t n_keys = SPLIT ? min(p0 + t_last + 1, j0 + split_len) : p0 + t_last + 1;
    const int64_t lim = min(p0 + t, n_keys - 1);
    const int ntiles = (int) ((n_keys - j0 + AKT - 1) / AKT);
    const int row = n_kv * HD;

    // Q's int8 A fragments (4 k-steps of 32 dims; regs: row g / g + 8, dims 4tq.. and 16 + 4tq..) and its row scales
    uint32_t qa[4][4];
    float sq0, sq1;
    {
        const float* q0 = q + ((size_t) (live ? t : t_last) * n_head + kvh * 16) * HD;
        float x[2][4][8];
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int d = ks * 32 + hf * 16 + 4 * tq;
                const float4 u = *reinterpret_cast<const float4*>(q0 + (size_t) g * HD + d);
                const float4 w = *reinterpret_cast<const float4*>(q0 + (size_t) (g + 8) * HD + d);
                x[0][ks][hf * 4 + 0] = u.x; x[0][ks][hf * 4 + 1] = u.y; x[0][ks][hf * 4 + 2] = u.z; x[0][ks][hf * 4 + 3] = u.w;
                x[1][ks][hf * 4 + 0] = w.x; x[1][ks][hf * 4 + 1] = w.y; x[1][ks][hf * 4 + 2] = w.z; x[1][ks][hf * 4 + 3] = w.w;
                a0 = fmaxf(a0, fmaxf(fmaxf(fabsf(u.x), fabsf(u.y)), fmaxf(fabsf(u.z), fabsf(u.w))));
                a1 = fmaxf(a1, fmaxf(fmaxf(fabsf(w.x), fabsf(w.y)), fmaxf(fabsf(w.z), fabsf(w.w))));
            }
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1));
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));
        sq0 = a0 / 127.0f;
        sq1 = a1 / 127.0f;
        const float i0 = sq0 > 0.0f ? 1.0f / sq0 : 0.0f, i1 = sq1 > 0.0f ? 1.0f / sq1 : 0.0f;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const float* u = x[0][ks] + hf * 4;
                const float* w = x[1][ks] + hf * 4;
                qa[ks][2 * hf] = pack_s8(__float2int_rn(u[0] * i0), __float2int_rn(u[1] * i0), __float2int_rn(u[2] * i0),
                                         __float2int_rn(u[3] * i0));
                qa[ks][2 * hf + 1] = pack_s8(__float2int_rn(w[0] * i1), __float2int_rn(w[1] * i1),
                                             __float2int_rn(w[2] * i1), __float2int_rn(w[3] * i1));
            }
        sq0 *= qscale;   // S = int dot * sq * ks, already in base 2
        sq1 *= qscale;
    }
    // the loader: a thread's rows and 16-byte pieces are the same every tile (K: rows kr0 + 32 i, piece tid & 7;
    // V: rows vr0 + 16 i, piece tid & 15), so their shared offsets are fixed; a tile inside one page (always, unless a
    // window starts mid-page) takes its rows from one page pointer
    constexpr int NTH = NW * 32;
    static_assert(NTH == 256 && AKT % 32 == 0, "the loader is written for 256 threads");
    const int kr0 = threadIdx.x >> 3, kpc = threadIdx.x & 7, vr0 = threadIdx.x >> 4, vpc = threadIdx.x & 15;
    const uint32_t sk_off = (uint32_t) (kr0 * KP + sw16(kr0, kpc));          // + i * 32 * KP (row & 7 is fixed)
    const uint32_t sv_off = (uint32_t) (vr0 * VP * 2 + sw16(vr0, vpc));      // + i * 16 * VP * 2
    auto load = [&](int buf, int tile) {
        const int64_t jt = j0 + (int64_t) tile * AKT;
        int8_t* skb = sk + (size_t) buf * AKT * KP;
        uint8_t* svb = (uint8_t*) (sv + (size_t) buf * AKT * VP);
        const bool one_page = (jt >> kv.shift) == ((jt + AKT - 1) >> kv.shift);
        const int8_t* kp = kv.k[jt >> kv.shift] + cell_of(kv.shift, jt) * row + (size_t) kvh * HD;
        const uint16_t* vp = kv.v[jt >> kv.shift] + cell_of(kv.shift, jt) * row + (size_t) kvh * HD;
#pragma unroll
        for (int i = 0; i < AKT / 32; ++i) {
            const int r = kr0 + 32 * i;
            const int64_t j = jt + r;
            void* dst = skb + sk_off + i * 32 * KP;
            if (j < n_keys) {
                const int8_t* src = one_page ? kp + (size_t) r * row
                                             : kv.k[j >> kv.shift] + cell_of(kv.shift, j) * row + (size_t) kvh * HD;
                cp16(dst, src + kpc * 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = 0; i < AKT / 16; ++i) {
            const int r = vr0 + 16 * i;
            const int64_t j = jt + r;
            void* dst = svb + sv_off + i * 16 * VP * 2;
            if (j < n_keys) {
                const uint16_t* src = one_page ? vp + (size_t) r * row
                                               : kv.v[j >> kv.shift] + cell_of(kv.shift, j) * row + (size_t) kvh * HD;
                cp16(dst, src + vpc * 8);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        if (threadIdx.x < AKT) {
            const int64_t j = jt + threadIdx.x;
            if (j < n_keys) cp4(sks + buf * AKT + threadIdx.x, kv.ks[j >> kv.shift] + cell_of(kv.shift, j) * n_kv + kvh);
            else sks[buf * AKT + threadIdx.x] = 0.0f;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float oc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) oc[i][0] = oc[i][1] = oc[i][2] = oc[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // rows g, g + 8

    load(0, 0);
    for (int it = 0; it < ntiles; ++it) {
        if (it + 1 < ntiles) {
            load((it + 1) & 1, it + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const int64_t jt = j0 + (int64_t) it * AKT;
        if (live && jt <= lim) {
            const int8_t* K = sk + (size_t) (it & 1) * AKT * KP;
            const uint16_t* V = sv + (size_t) (it & 1) * AKT * VP;
            const float* KS = sks + (it & 1) * AKT;
            // S (16 x 64): 8 n-tiles of 8 keys, int32
            int si[NTL][4];
#pragma unroll
            for (int nt = 0; nt < NTL; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = 0;
#pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
#pragma unroll
                for (int kp = 0; kp < 2; ++kp) {
                    // two k-steps (64 dims): matrices (keys nt*8.., bytes kp*64 + 0/16/32/48)
                    uint32_t b[4];
                    const int kr = nt * 8 + (lane % 8);
                    ldsm_x4(b, K + (size_t) kr * KP + sw16(kr, kp * 4 + lane / 8));
                    mma_s8(si[nt], qa[2 * kp], b[0], b[1]);
                    mma_s8(si[nt], qa[2 * kp + 1], b[2], b[3]);
                }
            }
                float sc[NTL][4];
#pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
                const float k0 = KS[nt * 8 + 2 * tq], k1 = KS[nt * 8 + 2 * tq + 1];
                sc[nt][0] = (float) si[nt][0] * sq0 * k0;
                sc[nt][1] = (float) si[nt][1] * sq0 * k1;
                sc[nt][2] = (float) si[nt][2] * sq1 * k0;
                sc[nt][3] = (float) si[nt][3] * sq1 * k1;
            }
            if (jt + AKT - 1 > lim) {   // the diagonal tile: keys past this query are -inf
#pragma unroll
                for (int nt = 0; nt < NTL; ++nt) {
                    const int64_t j = jt + nt * 8 + 2 * tq;
                    if (j > lim) { sc[nt][0] = -INFINITY; sc[nt][2] = -INFINITY; }
                    if (j + 1 > lim) { sc[nt][1] = -INFINITY; sc[nt][3] = -INFINITY; }
                }
            }
            float x0 = m0, x1 = m1;
#pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
                x0 = fmaxf(x0, fmaxf(sc[nt][0], sc[nt][1]));
                x1 = fmaxf(x1, fmaxf(sc[nt][2], sc[nt][3]));
            }
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
            // lazy maximum: a row's reference moves only when the tile's maximum passes it by more than 2^5, so P <= 32
            // (the FP16 tile sums stay far from overflow) and the accumulators are rescaled on few tiles
            const bool up0 = x0 > m0 + 5.0f, up1 = x1 > m1 + 5.0f;   // (m = -inf before the first tile: always)
            if (__any_sync(0xffffffffu, up0 || up1)) {
                const float c0 = up0 ? exp2f(m0 - x0) : 1.0f, c1 = up1 ? exp2f(m1 - x1) : 1.0f;
                if (up0) m0 = x0;
                if (up1) m1 = x1;
                l0 *= c0;
                l1 *= c1;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    oc[i][0] *= c0; oc[i][1] *= c0;
                    oc[i][2] *= c1; oc[i][3] *= c1;
                }
            }
            uint32_t pa[KSL][4];   // P's FP16 A fragments, k-steps of 16 keys: exp2 in FP16, two at a time
#pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
                const __half2 e01 = h2exp2(__floats2half2_rn(sc[nt][0] - m0, sc[nt][1] - m0));
                const __half2 e23 = h2exp2(__floats2half2_rn(sc[nt][2] - m1, sc[nt][3] - m1));
                pa[nt / 2][(nt & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&e01);
                pa[nt / 2][(nt & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&e23);
            }
            {
                // the rows' sums of P on the tensor cores: P times a column of ones (every column of the result is
                // the row's sum, so each thread gets its rows' whole sums - no quad reduction at the end)
                const uint32_t one = 0x3c003c00u;
                uint32_t ls[2] = {0u, 0u};
#pragma unroll
                for (int ks = 0; ks < KSL; ++ks) mma_f16(ls, pa[ks], one, one);
                l0 += __low2float(*reinterpret_cast<const __half2*>(&ls[0]));
                l1 += __low2float(*reinterpret_cast<const __half2*>(&ls[1]));
            }
                // the tile's P V in FP16 (16 n-tiles of 8 dims, 4 k-steps of 16 keys), then into the FP32 sums
            uint32_t ph[16][2];
#pragma unroll
            for (int i = 0; i < 16; ++i) ph[i][0] = ph[i][1] = 0u;
#pragma unroll
            for (int ks = 0; ks < KSL; ++ks) {
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    // matrices: (keys ks*16 + 0/8, dims np*16 + 0/8), transposed
                    uint32_t b[4];
                    const int mi = lane / 8;
                    const int vr = ks * 16 + (mi & 1) * 8 + (lane % 8);
                    ldsm_x4_t(b, (const uint8_t*) (V + (size_t) vr * VP) + sw16(vr, np * 2 + (mi >> 1)));
                    mma_f16(ph[2 * np], pa[ks], b[0], b[1]);
                    mma_f16(ph[2 * np + 1], pa[ks], b[2], b[3]);
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float2 u = __half22float2(*reinterpret_cast<const __half2*>(&ph[i][0]));
                const float2 w = __half22float2(*reinterpret_cast<const __half2*>(&ph[i][1]));
                oc[i][0] += u.x; oc[i][1] += u.y;
                oc[i][2] += w.x; oc[i][3] += w.y;
            }
            }
        __syncthreads();
    }
    if (!live) return;
    if (SPLIT) {
        // the partial sums as they are (m back to natural-log units: exp2(x) = exp(x ln 2))
        const int split = blockIdx.x;
        const size_t s0 = ((size_t) t * n_splits + split) * n_head + kvh * 16 + g, s1 = s0 + 8;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int d = nt * 8 + 2 * tq;
            *reinterpret_cast<float2*>(pacc + s0 * HD + d) = make_float2(oc[nt][0], oc[nt][1]);
            *reinterpret_cast<float2*>(pacc + s1 * HD + d) = make_float2(oc[nt][2], oc[nt][3]);
        }
        if (tq == 0) {
            pm[s0] = m0 * 0.69314718055994531f;
            pm[s1] = m1 * 0.69314718055994531f;
            pl[s0] = l0;
            pl[s1] = l1;
        }
        return;
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    float* o0 = o + ((size_t) t * n_head + kvh * 16) * HD;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int d = nt * 8 + 2 * tq;
        *reinterpret_cast<float2*>(o0 + (size_t) g * HD + d) = make_float2(oc[nt][0] * i0, oc[nt][1] * i0);
        *reinterpret_cast<float2*>(o0 + (size_t) (g + 8) * HD + d) = make_float2(oc[nt][2] * i1, oc[nt][3] * i1);
    }
}

template <int NW>
void tc_attr(size_t smem) {
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(attn_tc_kernel<NW, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    cudaFuncSetAttribute(attn_tc_kernel<NW, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    done = true;
}

}  // namespace


uint64_t nemo_attn_scratch_bytes(const NemoAttnShapes& s, int64_t T, int64_t n_keys) {
    // the largest any split-K call (T <= 8; a longer chunk runs on tensor cores without scratch) over at most n_keys
    // keys needs: T * n_splits is not monotonic in T, so take the maximum over every T' <= T
    uint64_t most = 0;
    for (int64_t t = 1; t <= T && t <= 8; ++t) {
        const uint64_t slots = (uint64_t) t * n_splits_for(s, t, n_keys);
        if (slots > most) most = slots;
    }
    return most * s.n_head * (HD + 2) * sizeof(float) + 256;
}

void nemo_kv_append(const NemoKV& kv, const float* k_new, const float* v_new, int64_t p0, int64_t T,
                    const NemoAttnShapes& s, void* stream) {
    if (T <= 0 || s.head_dim != HD || s.n_head_kv * HD > 1024) return;
    kv_append_kernel<<<(unsigned) T, (unsigned) (s.n_head_kv * HD), 0, (cudaStream_t) stream>>>(kv, k_new, v_new, p0,
                                                                                               (int) s.n_head_kv);
}

void nemo_attn(const float* q, const NemoKV& kv, float* o, int64_t p0, int64_t T, const NemoAttnShapes& s,
               void* scratch, void* stream, int64_t lo) {
    if (lo < 0) lo = 0;
    if (T <= 0 || s.head_dim != HD || s.n_head % s.n_head_kv || s.n_head / s.n_head_kv > 32) return;
    cudaStream_t st = (cudaStream_t) stream;
    if (T > 8 && s.n_head / s.n_head_kv == 16) {
        // the prompt: tensor cores
        constexpr int NW = NEMO_ATTN_NW;
        const size_t smem = (size_t) 2 * AKT * KP + (size_t) 2 * AKT * VP * 2 + (size_t) 2 * AKT * 4;
        tc_attr<NW>(smem);
        attn_tc_kernel<NW, false><<<dim3((unsigned) ((T + NW - 1) / NW), (unsigned) s.n_head_kv), NW * 32, smem, st>>>(
            q, kv, o, p0, (int) T, (int) s.n_head, (int) s.n_head_kv, 1.4426950408889634f / sqrtf((float) HD), 0, 0,
            nullptr, nullptr, nullptr, lo);
        return;
    }
    const int64_t n_keys = p0 + T - lo;   // (the keys from lo)
    const int n_splits = n_splits_for(s, T, n_keys);
    const int split_len = (int) split_len_for(n_keys);
    const uint64_t slots = (uint64_t) T * n_splits * s.n_head;
    float* pacc = (float*) scratch;
    float* pm = pacc + slots * HD;
    float* pl = pm + slots;
    if (n_keys >= NEMO_ATTN_TC_DECODE_KEYS && s.n_head / s.n_head_kv == 16) {
        // a long context: the window's queries on tensor cores, split-K (each split a multiple of the 64-key tile)
        constexpr int NW = 8;
        const int sl = (split_len + AKT - 1) / AKT * AKT;
        const int ns = (int) ((n_keys + sl - 1) / sl);
        const size_t smem = (size_t) 2 * AKT * KP + (size_t) 2 * AKT * VP * 2 + (size_t) 2 * AKT * 4;
        tc_attr<NW>(smem);
        attn_tc_kernel<NW, true><<<dim3((unsigned) ns, (unsigned) s.n_head_kv), NW * 32, smem, st>>>(
            q, kv, o, p0, (int) T, (int) s.n_head, (int) s.n_head_kv, 1.4426950408889634f / sqrtf((float) HD), sl, ns,
            pacc, pm, pl, lo);
        attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) T), 32, 0, st>>>(pacc, pm, pl, o, (int) s.n_head, ns);
        return;
    }
    const int group = (int) (s.n_head / s.n_head_kv);
    const dim3 grid((unsigned) n_splits, (unsigned) s.n_head_kv);
    const float sc = 1.0f / sqrtf((float) HD);
    if (T == 1)
        attn_split_kernel<1><<<grid, group * 32, 0, st>>>(q, kv, pacc, pm, pl, p0, (int) T, (int) s.n_head,
                                                          (int) s.n_head_kv, split_len, n_splits, sc, lo);
    else if (T <= 4)
        attn_split_kernel<4><<<grid, group * 32, 0, st>>>(q, kv, pacc, pm, pl, p0, (int) T, (int) s.n_head,
                                                          (int) s.n_head_kv, split_len, n_splits, sc, lo);
    else
        attn_split_kernel<8><<<grid, group * 32, 0, st>>>(q, kv, pacc, pm, pl, p0, (int) T, (int) s.n_head,
                                                          (int) s.n_head_kv, split_len, n_splits, sc, lo);
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) T), 32, 0, st>>>(pacc, pm, pl, o, (int) s.n_head,
                                                                              n_splits);
}

}  // namespace strata::kernels
