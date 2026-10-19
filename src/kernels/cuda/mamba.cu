// src/kernels/cuda/mamba.cu - strata-nemotron stage 2: Mamba-2 decode, the parts between the projections.
//
// See include/strata/kernels/mamba.hpp.  Every kernel here is memory-bound on the state or the activations; the
// layout choices are for coalescing: the state is (H, P, N) with n fastest, and one warp owns one (h, p) row, its
// lanes walking n.  4 MiB of state read and written per layer and token is ~5.5 us at 1.5 TB/s.
#include "strata/kernels/mamba.hpp"

#include <cuda_runtime.h>

#include <cstdlib>

namespace strata::kernels {
namespace {

/// ggml_compute_softplus_f32: log1p(exp(x)), with the large-x branch that avoids overflow (as gdn.cu).
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// One thread per channel: the 4-tap dot over [state | x], bias, SiLU, then the slide.
__global__ void mamba_conv_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                  const float* __restrict__ bias, float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // tap 0 reads the OLDEST state entry
    acc += x[c] * w[dc - 1];                                // the new input is the LAST tap
    out[c] = silu_f(acc + bias[c]);
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

constexpr int SSM_WARPS = 8;   ///< (h, p) rows per block

/// One warp per (h, p) row of the state; lane l holds n = l, l+32, ... (N/32 values).
template <int NPL>
__global__ void mamba_ssm_kernel(float* __restrict__ state, const float* __restrict__ xbc,
                                 const float* __restrict__ dt_raw, const float* __restrict__ dt_bias,
                                 const float* __restrict__ A, const float* __restrict__ D, float* __restrict__ y,
                                 int H, int P, int G) {
    constexpr int N = NPL * 32;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * SSM_WARPS + warp;          // h * P + p
    if (row >= H * P) return;
    const int h = row / P;
    const int g = h / (H / G);                              // INTERLEAVE: 16 consecutive heads share a group
    const int d_inner = H * P;
    const float* B = xbc + d_inner + (size_t) g * N;
    const float* Cm = xbc + d_inner + (size_t) G * N + (size_t) g * N;
    const float xv = xbc[row];
    const float dt = softplus_f(dt_raw[h] + dt_bias[h]);
    const float dA = expf(dt * A[h]);
    const float dx = dt * xv;
    float* s = state + (size_t) row * N;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < NPL; ++k) {
        const int n = lane + 32 * k;
        const float v = s[n] * dA + dx * B[n];
        s[n] = v;
        acc += v * Cm[n];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc + D[h] * xv;
}

/// The conv over T tokens: one thread per (t, c), the taps in decode's order (oldest first), history from the state.
/// The state is updated by mamba_conv_state_kernel afterwards (every thread here reads the OLD state).
__global__ void mamba_conv_prefill_kernel(const float* __restrict__ cs, const float* __restrict__ zx, int64_t ld_zx,
                                          int64_t off, const float* __restrict__ kW, const float* __restrict__ bias,
                                          float* __restrict__ out, int C, int dc, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= C || t >= T) return;
    const float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) {
        const int src = t - (dc - 1) + i;   // the input this tap reads: < 0 is the carried history
        const float v = src < 0 ? st[dc - 1 + src] : zx[(size_t) src * ld_zx + off + c];
        acc += v * w[i];
    }
    acc += zx[(size_t) t * ld_zx + off + c] * w[dc - 1];
    out[(size_t) t * C + c] = silu_f(acc + bias[c]);
}

/// The conv history after T tokens: the last d_conv-1 inputs, oldest first (from the state when T < d_conv-1).
__global__ void mamba_conv_state_kernel(float* __restrict__ cs, const float* __restrict__ zx, int64_t ld_zx,
                                        int64_t off, int C, int dc, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    float nv[8];
    for (int i = 0; i < dc - 1; ++i) {
        const int src = T - (dc - 1) + i;
        nv[i] = src < 0 ? st[dc - 1 + src] : zx[(size_t) src * ld_zx + off + c];
    }
    for (int i = 0; i < dc - 1; ++i) st[i] = nv[i];
}

constexpr int SCAN_TT = 32;   ///< tokens staged in shared memory per pass

/// The scan: a block owns SSM_WARPS consecutive rows of ONE head (P % SSM_WARPS == 0), so its rows share B, C and
/// dt; a pass stages SCAN_TT tokens of them in shared memory, then each warp steps its row through them with the
/// state in registers.  The per-token arithmetic is mamba_ssm_kernel's, expression for expression.
template <int NPL>
__global__ void mamba_scan_kernel(float* __restrict__ state, const float* __restrict__ xbc,
                                  const float* __restrict__ zx, int64_t ld_zx, const float* __restrict__ dt_bias,
                                  const float* __restrict__ A, const float* __restrict__ D, float* __restrict__ y,
                                  int H, int P, int G, int T) {
    constexpr int N = NPL * 32;
    __shared__ float Bs[SCAN_TT][N];
    __shared__ float Cs[SCAN_TT][N];
    __shared__ float xs[SCAN_TT][SSM_WARPS];
    __shared__ float dts[SCAN_TT], dAs[SCAN_TT];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * SSM_WARPS;
    const int h = row0 / P;
    const int g = h / (H / G);
    const int d_inner = H * P;
    const int conv_dim = d_inner + 2 * G * N;
    const int row = row0 + warp;
    float s[NPL];
#pragma unroll
    for (int k = 0; k < NPL; ++k) s[k] = state[(size_t) row * N + lane + 32 * k];
    const float Dh = D[h];

    // the next pass's B, C, x and dt are loaded into registers while this pass computes (the loads are independent,
    // so they are all in flight at once; staged one by one they were ~16 dependent round trips a pass)
    constexpr int NTH = SSM_WARPS * 32, RB = SCAN_TT * 2 * N / NTH;
    static_assert(SCAN_TT * 2 * N % NTH == 0 && SCAN_TT * SSM_WARPS == NTH, "scan staging shape");
    float rb[RB], rx = 0.0f, rdt = 0.0f;
    auto fetch = [&](int t0) {
        const int nt = min(SCAN_TT, T - t0);
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int i = threadIdx.x + j * NTH, tt = i / (2 * N), n2 = i % (2 * N);
            const float* xr = xbc + (size_t) (t0 + tt) * conv_dim + d_inner;
            rb[j] = tt < nt ? xr[n2 < N ? g * N + n2 : G * N + g * N + (n2 - N)] : 0.0f;
        }
        {
            const int tt = threadIdx.x / SSM_WARPS, r = threadIdx.x % SSM_WARPS;
            rx = tt < nt ? xbc[(size_t) (t0 + tt) * conv_dim + row0 + r] : 0.0f;
        }
        if (threadIdx.x < nt) rdt = zx[(size_t) (t0 + threadIdx.x) * ld_zx + d_inner + conv_dim + h];
    };
    fetch(0);
    for (int t0 = 0; t0 < T; t0 += SCAN_TT) {
        const int nt = min(SCAN_TT, T - t0);
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int i = threadIdx.x + j * NTH, tt = i / (2 * N), n2 = i % (2 * N);
            if (n2 < N) Bs[tt][n2] = rb[j];
            else Cs[tt][n2 - N] = rb[j];
        }
        xs[threadIdx.x / SSM_WARPS][threadIdx.x % SSM_WARPS] = rx;
        if (threadIdx.x < nt) {
            const float dt = softplus_f(rdt + dt_bias[h]);
            dts[threadIdx.x] = dt;
            dAs[threadIdx.x] = expf(dt * A[h]);
        }
        __syncthreads();
        if (t0 + SCAN_TT < T) fetch(t0 + SCAN_TT);
        // each lane's partial sums for the pass's tokens, then one reduce-scatter (31 shuffles for 32 tokens, not
        // 5 a token): afterwards lane l holds token l's sum
        float part[SCAN_TT];
#pragma unroll
        for (int tt = 0; tt < SCAN_TT; ++tt) {
            float acc = 0.0f;
            if (tt < nt) {
                const float dA = dAs[tt];
                const float dx = dts[tt] * xs[tt][warp];
#pragma unroll
                for (int k = 0; k < NPL; ++k) {
                    const int n = lane + 32 * k;
                    const float v = s[k] * dA + dx * Bs[tt][n];
                    s[k] = v;
                    acc += v * Cs[tt][n];
                }
            }
            part[tt] = acc;
        }
#pragma unroll
        for (int o = SCAN_TT / 2; o > 0; o >>= 1) {
            const bool up = lane & o;
#pragma unroll
            for (int i = 0; i < o; ++i) {
                const float send = up ? part[i] : part[i + o];
                const float keep = up ? part[i + o] : part[i];
                part[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
            }
        }
        if (lane < nt) y[(size_t) (t0 + lane) * d_inner + row] = part[0] + Dh * xs[lane][warp];
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < NPL; ++k) state[(size_t) row * N + lane + 32 * k] = s[k];
}

// ---- the prompt's form: chunked SSD (Mamba-2's state-space dual) ----
//
// The recurrence s_t = e^(dt_t A) s_{t-1} + dt_t x_t B_t, y_t = C_t . s_t + D x_t over a chunk of Q tokens is, with
// a_t = dt_t A and its running sum c_i inside the chunk (c_{-1} = 0, last = c_{Q-1}):
//     y_i  = sum_{j <= i} (C_i . B_j) e^(c_i - c_j) dt_j x_j  +  e^(c_i) C_i . S  +  D x_i
//     S'   = e^(last) S + sum_j e^(last - c_j) dt_j x_j B_j^T
// (S the state before the chunk) - matrix products instead of a token-by-token chain.  A block owns one head and
// SSD_PS of its P columns, keeps that slice of the state (N x SSD_PS) in registers and steps through the chunks;
// everything is FP32 (CUDA cores, register tiles).  Not bit-identical to mamba_scan_kernel (other summation orders):
// the verify windows (T < SSD_MIN) keep the sequential scan.
constexpr int SSD_Q = 32, SSD_NTH = 256, SSD_MIN = 64;
#ifndef NEMO_SSD_PS
#define NEMO_SSD_PS 32
#endif
#ifndef NEMO_SSD_CPAD
#define NEMO_SSD_CPAD 0   // B, C rows unpadded: a warp reads 2 rows of C (2-way at most), 4 spaced columns of B (none) - 0.538 -> 0.500 ms on 2048 tokens
#endif
constexpr int SSD_PS = NEMO_SSD_PS;   ///< state columns a block owns (32 measured faster than 16)

template <int N, int PS>
constexpr size_t ssd_smem() {
    return sizeof(float) * (2 * SSD_Q * (N + NEMO_SSD_CPAD) + (N + SSD_Q) * (PS + 4) + SSD_Q * (SSD_Q + 1) + 3 * SSD_Q);
}

/// C B^T of every chunk and group (the heads of a group share it): cb[(chunk * G + g) * Q * Q + i * Q + j] = C_i . B_j.
template <int N>
__global__ void __launch_bounds__(256) mamba_cb_kernel(const float* __restrict__ xbc, float* __restrict__ cb, int H,
                                                       int P, int G, int T) {
    constexpr int Q = SSD_Q, CP = N + 1;
    __shared__ float Cs[Q * CP], Bs[Q * CP];
    const int chunk = blockIdx.x, g = blockIdx.y, t0 = chunk * Q, nq = min(Q, T - t0);
    const int d_inner = H * P, conv_dim = d_inner + 2 * G * N;
    for (int i = threadIdx.x; i < Q * N; i += 256) {
        const int r = i / N, n = i % N;
        const float* xr = xbc + (size_t) (t0 + r) * conv_dim + d_inner;
        Bs[r * CP + n] = r < nq ? xr[g * N + n] : 0.0f;
        Cs[r * CP + n] = r < nq ? xr[G * N + g * N + n] : 0.0f;
    }
    __syncthreads();
    const int yi = threadIdx.x / 16, yc = threadIdx.x % 16;   // rows 2 yi.., columns 2 yc..
    float a00 = 0, a01 = 0, a10 = 0, a11 = 0;
    for (int n = 0; n < N; ++n) {
        const float c0 = Cs[(2 * yi) * CP + n], c1 = Cs[(2 * yi + 1) * CP + n];
        const float b0 = Bs[(2 * yc) * CP + n], b1 = Bs[(2 * yc + 1) * CP + n];
        a00 += c0 * b0; a01 += c0 * b1; a10 += c1 * b0; a11 += c1 * b1;
    }
    float* o = cb + (size_t) (chunk * G + g) * Q * Q;
    o[(2 * yi) * Q + 2 * yc] = a00;
    o[(2 * yi) * Q + 2 * yc + 1] = a01;
    o[(2 * yi + 1) * Q + 2 * yc] = a10;
    o[(2 * yi + 1) * Q + 2 * yc + 1] = a11;
}

template <int N, int PS>
__global__ void __launch_bounds__(SSD_NTH) mamba_ssd_kernel(float* __restrict__ state, const float* __restrict__ xbc,
                                                            const float* __restrict__ zx, int64_t ld_zx,
                                                            const float* __restrict__ dt_bias,
                                                            const float* __restrict__ A, const float* __restrict__ D,
                                                            float* __restrict__ y, int H, int P, int G, int T,
                                                            const float* __restrict__ cbg) {
    constexpr int Q = SSD_Q, CP = N + NEMO_SSD_CPAD, XP = PS + 4;
    constexpr int YC = PS / 16;          // y columns a thread (rows 2 yi, 2 yi + 1)
    constexpr int SN = N / 32, SC = PS / 8;   // S tile a thread: SN rows x SC columns
    extern __shared__ float sh[];
    float* Cs = sh;                        // [Q][CP]
    float* Bs = Cs + Q * CP;               // [Q][CP]
    float* Ss = Bs + Q * CP;               // [N][XP]
    float* Xs = Ss + N * XP;               // [Q][XP]
    float* Ms = Xs + Q * XP;               // [Q][Q + 1]
    float* cs = Ms + Q * (Q + 1);          // [Q] running sums
    float* dts = cs + Q;                   // [Q]
    float* ws = dts + Q;                   // [Q]: e^(last - c_j) dt_j
    const int tid = threadIdx.x;
    const int h = blockIdx.x / (P / PS), p0 = (blockIdx.x % (P / PS)) * PS;
    const int g = h / (H / G);
    const int d_inner = H * P, conv_dim = d_inner + 2 * G * N;
    const float Ah = A[h], Dh = D[h], bias = dt_bias[h];
    const int yi = tid / 16, yc = tid % 16;
    const int sn = tid / 8, sp = (tid % 8) * SC;
    float S[SN][SC];
#pragma unroll
    for (int a = 0; a < SN; ++a)
#pragma unroll
        for (int b = 0; b < SC; ++b) S[a][b] = state[(size_t) (h * P + p0 + sp + b) * N + sn * SN + a];

    // the next chunk's B, C, x and raw dt are loaded into registers while this one computes
    constexpr int RBC = Q * N / SSD_NTH, RX = Q * PS / SSD_NTH;
    static_assert(Q * N % SSD_NTH == 0 && Q * PS % SSD_NTH == 0, "SSD staging shape");
    float rb[RBC], rc[RBC], rx[RX], rdt = 0.0f;
    auto fetch = [&](int t0) {
        const int nq = min(Q, T - t0);
#pragma unroll
        for (int k = 0; k < RBC; ++k) {
            const int i = tid + k * SSD_NTH, r = i / N, n = i % N;
            const float* xr = xbc + (size_t) (t0 + r) * conv_dim + d_inner;
            rb[k] = r < nq ? xr[g * N + n] : 0.0f;
            rc[k] = r < nq ? xr[G * N + g * N + n] : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < RX; ++k) {
            const int i = tid + k * SSD_NTH, r = i / PS, c = i % PS;
            rx[k] = r < nq ? xbc[(size_t) (t0 + r) * conv_dim + h * P + p0 + c] : 0.0f;
        }
        if (tid < 32) rdt = tid < nq ? zx[(size_t) (t0 + tid) * ld_zx + d_inner + conv_dim + h] : 0.0f;
    };
    fetch(0);
    for (int t0 = 0; t0 < T; t0 += Q) {
        const int nq = min(Q, T - t0);
        __syncthreads();   // the last chunk's reads of the shared tiles are done
#pragma unroll
        for (int k = 0; k < RBC; ++k) {
            const int i = tid + k * SSD_NTH, r = i / N, n = i % N;
            Bs[r * CP + n] = rb[k];
            Cs[r * CP + n] = rc[k];
        }
#pragma unroll
        for (int k = 0; k < RX; ++k) {
            const int i = tid + k * SSD_NTH, r = i / PS, c = i % PS;
            Xs[r * XP + c] = rx[k];
        }
#pragma unroll
        for (int a = 0; a < SN; ++a)
#pragma unroll
            for (int b = 0; b < SC; ++b) Ss[(sn * SN + a) * XP + sp + b] = S[a][b];
        if (tid < 32) {
            // dt and the running sum of dt A (a warp scan); padded rows: dt 0, so they change nothing
            const float dt = tid < nq ? softplus_f(rdt + bias) : 0.0f;
            float c = dt * Ah;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float u = __shfl_up_sync(0xffffffffu, c, o);
                if (tid >= o) c += u;
            }
            cs[tid] = c;
            dts[tid] = dt;
            ws[tid] = expf(__shfl_sync(0xffffffffu, c, 31) - c) * dt;
        }
        __syncthreads();
        if (t0 + Q < T) fetch(t0 + Q);
        // C B^T (the group's, from mamba_cb_kernel) -> M (masked, decayed, times dt_j), 2 x 2 a thread;
        // C . S (the state's part of y)
        float cb[2][2], yo[2][YC];
        {
            const float* cbc = cbg + ((size_t) (t0 / Q) * G + g) * Q * Q;
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int b = 0; b < 2; ++b) cb[a][b] = cbc[(2 * yi + a) * Q + 2 * yc + b];
        }
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < YC; ++b) yo[a][b] = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float c0 = Cs[(2 * yi) * CP + n], c1 = Cs[(2 * yi + 1) * CP + n];
#pragma unroll
            for (int b = 0; b < YC; ++b) {
                const float sv = Ss[n * XP + YC * yc + b];
                yo[0][b] += c0 * sv;
                yo[1][b] += c1 * sv;
            }
        }
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int i = 2 * yi + a, j = 2 * yc + b;
                Ms[i * (Q + 1) + j] = j <= i ? cb[a][b] * expf(cs[i] - cs[j]) * dts[j] : 0.0f;
            }
        __syncthreads();
        // y = M X + e^(c_i) C . S + D x
        float yd[2][YC];
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < YC; ++b) yd[a][b] = 0.0f;
        for (int j = 0; j < Q; ++j) {
            const float m0 = Ms[(2 * yi) * (Q + 1) + j], m1 = Ms[(2 * yi + 1) * (Q + 1) + j];
#pragma unroll
            for (int b = 0; b < YC; ++b) {
                const float xv = Xs[j * XP + YC * yc + b];
                yd[0][b] += m0 * xv;
                yd[1][b] += m1 * xv;
            }
        }
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            const int i = 2 * yi + a;
            if (i >= nq) continue;
            const float e = expf(cs[i]);
            float* yr = y + (size_t) (t0 + i) * d_inner + h * P + p0 + YC * yc;
#pragma unroll
            for (int b = 0; b < YC; ++b) yr[b] = yd[a][b] + e * yo[a][b] + Dh * Xs[i * XP + YC * yc + b];
        }
        // S' = e^(last) S + sum_j B_j^T (w_j x_j)
        const float el = expf(cs[Q - 1]);
#pragma unroll
        for (int a = 0; a < SN; ++a)
#pragma unroll
            for (int b = 0; b < SC; ++b) S[a][b] *= el;
        for (int j = 0; j < Q; ++j) {
            float bv[SN], wv[SC];
            const float wj = ws[j];
#pragma unroll
            for (int a = 0; a < SN; ++a) bv[a] = Bs[j * CP + sn * SN + a];
#pragma unroll
            for (int b = 0; b < SC; ++b) wv[b] = wj * Xs[j * XP + sp + b];
#pragma unroll
            for (int a = 0; a < SN; ++a)
#pragma unroll
                for (int b = 0; b < SC; ++b) S[a][b] += bv[a] * wv[b];
        }
    }
#pragma unroll
    for (int a = 0; a < SN; ++a)
#pragma unroll
        for (int b = 0; b < SC; ++b) state[(size_t) (h * P + p0 + sp + b) * N + sn * SN + a] = S[a][b];
}

/// One block per (group, row): y * silu(z), the group's mean square, the scale, the weight.
__global__ void mamba_gated_norm_kernel(const float* __restrict__ y, int64_t ld_y, const float* __restrict__ z,
                                        int64_t ld_z, const float* __restrict__ w, float* __restrict__ out,
                                        int64_t ld_out, int gsize, float eps) {
    const int g = blockIdx.x;
    y += (size_t) blockIdx.y * ld_y;
    z += (size_t) blockIdx.y * ld_z;
    out += (size_t) blockIdx.y * ld_out;
    const float* yg = y + (size_t) g * gsize;
    const float* zg = z + (size_t) g * gsize;
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < gsize; i += blockDim.x) {
        const float v = yg[i] * silu_f(zg[i]);
        ss += v * v;
    }
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float) gsize + eps);
    // a second pass recomputes y * silu(z) instead of keeping it: `out` may alias `y`, and every thread reads only
    // the elements it writes, so the in-place form is safe
    for (int i = threadIdx.x; i < gsize; i += blockDim.x) {
        const float v = yg[i] * silu_f(zg[i]);
        out[(size_t) g * gsize + i] = v * scale * w[(size_t) g * gsize + i];
    }
}

}  // namespace

void mamba_conv_step(float* conv_state, const float* x, const float* kW, const float* bias, float* out,
                     int64_t channels, int64_t d_conv, void* stream) {
    const int threads = 256;
    const int blocks = (int) ((channels + threads - 1) / threads);
    mamba_conv_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(conv_state, x, kW, bias, out, (int) channels,
                                                                    (int) d_conv);
}

void mamba_ssm_step(float* state, const float* xbc, const float* dt_raw, const float* dt_bias, const float* A,
                    const float* D, float* y, const MambaShapes& s, void* stream) {
    const int rows = (int) (s.H * s.P);
    const int blocks = (rows + SSM_WARPS - 1) / SSM_WARPS;
    cudaStream_t st = (cudaStream_t) stream;
    switch (s.N) {
    case 64:
        mamba_ssm_kernel<2><<<blocks, SSM_WARPS * 32, 0, st>>>(state, xbc, dt_raw, dt_bias, A, D, y, (int) s.H,
                                                               (int) s.P, (int) s.G);
        break;
    case 128:
        mamba_ssm_kernel<4><<<blocks, SSM_WARPS * 32, 0, st>>>(state, xbc, dt_raw, dt_bias, A, D, y, (int) s.H,
                                                               (int) s.P, (int) s.G);
        break;
    case 256:
        mamba_ssm_kernel<8><<<blocks, SSM_WARPS * 32, 0, st>>>(state, xbc, dt_raw, dt_bias, A, D, y, (int) s.H,
                                                               (int) s.P, (int) s.G);
        break;
    default:
        break;   // the caller checks N at load time (nemotron_check); an unsupported N is a wiring error
    }
}

void mamba_gated_norm(const float* y, const float* z, const float* w, float* out, int64_t d_inner, int64_t G,
                      float eps, void* stream) {
    mamba_gated_norm_kernel<<<dim3((unsigned) G, 1), 256, 0, (cudaStream_t) stream>>>(y, 0, z, 0, w, out, 0,
                                                                                   (int) (d_inner / G), eps);
}

void mamba_gated_norm_rows(const float* y, const float* zx, int64_t ld_zx, const float* w, float* out, int64_t T,
                           int64_t d_inner, int64_t G, float eps, void* stream) {
    if (T <= 0) return;
    mamba_gated_norm_kernel<<<dim3((unsigned) G, (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
        y, d_inner, zx, ld_zx, w, out, d_inner, (int) (d_inner / G), eps);
}

void mamba_conv_prefill(float* conv_state, const float* zx, int64_t ld_zx, const float* kW, const float* bias,
                        float* xbc, int64_t T, const MambaShapes& s, void* stream) {
    if (T <= 0) return;
    const int C = (int) s.conv_dim(), threads = 256;
    cudaStream_t st = (cudaStream_t) stream;
    mamba_conv_prefill_kernel<<<dim3((unsigned) ((C + threads - 1) / threads), (unsigned) T), threads, 0, st>>>(
        conv_state, zx, ld_zx, s.d_inner(), kW, bias, xbc, C, (int) s.d_conv, (int) T);
    mamba_conv_state_kernel<<<(C + threads - 1) / threads, threads, 0, st>>>(conv_state, zx, ld_zx, s.d_inner(), C,
                                                                              (int) s.d_conv, (int) T);
}

void mamba_ssm_scan(float* state, const float* xbc, const float* zx, int64_t ld_zx, const float* dt_bias,
                    const float* A, const float* D, float* y, int64_t T, const MambaShapes& s, void* stream) {
    if (T <= 0 || s.P % SSM_WARPS) return;
    const int blocks = (int) (s.H * s.P / SSM_WARPS);
    cudaStream_t st = (cudaStream_t) stream;
    if (T >= SSD_MIN && s.N == 128 && s.P % SSD_PS == 0 && !std::getenv("STRATA_NEMO_SEQ_SCAN")) {
        constexpr int N = 128;
        constexpr size_t smem = ssd_smem<N, SSD_PS>();
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute(mamba_ssd_kernel<N, SSD_PS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            attr = true;
        }
        // the chunks' C B^T: a scratch of chunks x G x Q x Q floats, grown on demand (one per process)
        static float* cbs = nullptr;
        static size_t cbs_n = 0;
        const int64_t chunks = (T + SSD_Q - 1) / SSD_Q;
        const size_t need = (size_t) chunks * s.G * SSD_Q * SSD_Q;
        if (need > cbs_n) {
            if (cbs) cudaFree(cbs);
            if (cudaMalloc(&cbs, need * sizeof(float)) != cudaSuccess) { cbs = nullptr; cbs_n = 0; return; }
            cbs_n = need;
        }
        mamba_cb_kernel<N><<<dim3((unsigned) chunks, (unsigned) s.G), 256, 0, st>>>(xbc, cbs, (int) s.H, (int) s.P,
                                                                                   (int) s.G, (int) T);
        mamba_ssd_kernel<N, SSD_PS><<<(unsigned) (s.H * (s.P / SSD_PS)), SSD_NTH, smem, st>>>(
            state, xbc, zx, ld_zx, dt_bias, A, D, y, (int) s.H, (int) s.P, (int) s.G, (int) T, cbs);
        return;
    }
    switch (s.N) {
    case 64:
        mamba_scan_kernel<2><<<blocks, SSM_WARPS * 32, 0, st>>>(state, xbc, zx, ld_zx, dt_bias, A, D, y, (int) s.H,
                                                                (int) s.P, (int) s.G, (int) T);
        break;
    case 128:
        mamba_scan_kernel<4><<<blocks, SSM_WARPS * 32, 0, st>>>(state, xbc, zx, ld_zx, dt_bias, A, D, y, (int) s.H,
                                                                (int) s.P, (int) s.G, (int) T);
        break;
    default:
        break;   // N = 256 would need 32 KiB of staged B/C per pass: not this model
    }
}

}  // namespace strata::kernels
