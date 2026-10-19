// src/kernels/cuda/nemo_c4.cu - strata-nemotron: the C4 column layout of NVFP4 matrices and its sparse-input GEMV
// (include/strata/kernels/nemo_c4.hpp).
#include "strata/kernels/nemo_c4.hpp"

#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

__constant__ float kC4E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float c4_ue4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float) m * (1.0f / 512.0f);
}

/// One thread per (block column c, row pair): the pair's 16 codes of each column and the two rows' scales.
__global__ void c4_from_ggml_kernel(const uint8_t* __restrict__ g, int rows, int cols, uint8_t* __restrict__ c4) {
    const int nbc = cols / 16, pairs = rows / 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) nbc * pairs) return;
    const int c = (int) (i / pairs), rp = (int) (i % pairs), r0 = 2 * rp;
    const uint64_t rb = (uint64_t) (cols / 64) * 36;
    const int blk = c / 4, s = c % 4;
    uint8_t* codes = c4;
    uint8_t* scales = c4 + (uint64_t) rows * cols / 2;
    const uint8_t* b0 = g + (uint64_t) r0 * rb + (uint64_t) blk * 36;
    const uint8_t* b1 = b0 + rb;
    scales[(uint64_t) c * rows + r0] = b0[s];
    scales[(uint64_t) c * rows + r0 + 1] = b1[s];
    for (int v = 0; v < 16; ++v) {   // ggml sub-block s: byte j holds values j (low nibble) and j + 8 (high)
        const uint32_t q0 = (b0[4 + 8 * s + (v & 7)] >> (v < 8 ? 0 : 4)) & 15u;
        const uint32_t q1 = (b1[4 + 8 * s + (v & 7)] >> (v < 8 ? 0 : 4)) & 15u;
        codes[(uint64_t) (16 * c + v) * (rows / 2) + rp] = (uint8_t) (q0 | (q1 << 4));
    }
}

constexpr int C4_WARPS = 16, C4_RB = 128;   // a block: 128 rows of one matrix, 4 a lane; its 8 warps split the inputs
constexpr int C4_MAXK = 4096;

/// One block (8 warps) an input row: its nonzero columns in increasing order and their values, and how many.  Warp w
/// takes columns [w * span, (w + 1) * span); the slices' counts are summed in order.
__global__ void __launch_bounds__(C4_WARPS * 32) c4_compact_kernel(const float* __restrict__ x, int64_t x_stride,
                                                                   int cols, uint16_t* __restrict__ list,
                                                                   float* __restrict__ vals, int* __restrict__ count) {
    __shared__ int cnt[C4_WARPS + 1];
    const int j = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* xin = x + (size_t) j * x_stride;
    const int span = (cols / 32 + C4_WARPS - 1) / C4_WARPS * 32;
    const int k0 = warp * span, k1 = min(cols, k0 + span);
    float v[C4_MAXK / C4_WARPS / 32];
    int mine = 0;
#pragma unroll
    for (int i = 0; i < C4_MAXK / C4_WARPS / 32; ++i) {
        const int k = k0 + i * 32 + lane;
        v[i] = k < k1 ? xin[k] : 0.0f;
        mine += v[i] != 0.0f;
    }
    for (int o = 16; o > 0; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
    if (lane == 0) cnt[warp + 1] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        cnt[0] = 0;
        for (int w = 1; w <= C4_WARPS; ++w) cnt[w] += cnt[w - 1];
        count[j] = cnt[C4_WARPS];
    }
    __syncthreads();
    int at = cnt[warp];
    uint16_t* l = list + (size_t) j * cols;
    float* vl = vals + (size_t) j * cols;
#pragma unroll
    for (int i = 0; i < C4_MAXK / C4_WARPS / 32; ++i) {
        const unsigned m = __ballot_sync(0xffffffffu, v[i] != 0.0f);
        if (v[i] != 0.0f) {
            const int p = at + __popc(m & ((1u << lane) - 1u));
            l[p] = (uint16_t) (k0 + i * 32 + lane);
            vl[p] = v[i];
        }
        at += __popc(m);
    }
}

constexpr int C4_SEG = 64;    // a matrix's list is split in C4_SEG x C4_TEAMS parts: C4_SEG blocks of C4_TEAMS teams
constexpr int C4_TEAMS = 1;

/// grid (n, C4_SEG), C4_TEAMS x ROWS / 8 threads.  Team e of block (j, s) sums all rows of matrix j over part
/// s * C4_TEAMS + e of its list, lane l of a team owning rows 8l .. 8l + 7 (per column a 32-bit code load and an
/// 8-byte scale load: a team reads a column's codes as one contiguous run).  The teams' sums meet in shared memory in
/// team order, the block's goes to `part` (n x C4_SEG x rows), and c4_reduce_kernel adds the blocks' in order.
template <int ROWS>
__global__ void __launch_bounds__(C4_TEAMS * ROWS / 8) c4_gemv_kernel(const uint8_t* const* __restrict__ wp,
                                                                      const uint16_t* __restrict__ list,
                                                                      const float* __restrict__ vals,
                                                                      const int* __restrict__ count,
                                                                      float* __restrict__ part, int cols) {
    __shared__ float tsum[C4_TEAMS][ROWS];
    const int j = blockIdx.x, team = threadIdx.x / (ROWS / 8), tt = threadIdx.x % (ROWS / 8);
    const int p = blockIdx.y * C4_TEAMS + team, np = C4_SEG * C4_TEAMS;
    const uint8_t* codes = wp[j];
    const uint8_t* scales = codes + (uint64_t) ROWS * cols / 2;
    const uint16_t* nz = list + (size_t) j * cols;
    const float* xv = vals + (size_t) j * cols;
    const int n = count[j];
    const int t0 = (int) ((int64_t) n * p / np), t1 = (int) ((int64_t) n * (p + 1) / np);
    const int r0 = 8 * tt;
    const uint32_t* cb = reinterpret_cast<const uint32_t*>(codes) + tt;   // 8 rows: 4 bytes
    const uint2* sb = reinterpret_cast<const uint2*>(scales) + tt;        // their 8 scale bytes
    float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int t = t0; t < t1; ++t) {
        const int k = __ldg(nz + t);
        const float xk = __ldg(xv + t);
        const uint2 sc = __ldg(sb + (size_t) (k >> 4) * (ROWS / 8));
        const uint32_t q = __ldg(cb + (size_t) k * (ROWS / 8));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t sbyte = ((i < 4 ? sc.x : sc.y) >> (8 * (i & 3))) & 0xffu;
            a[i] += kC4E2M1[(q >> (4 * i)) & 15u] * c4_ue4m3(sbyte) * xk;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) tsum[team][r0 + i] = a[i];
    __syncthreads();
    float* out = part + ((size_t) j * C4_SEG + blockIdx.y) * ROWS;
    for (int r = threadIdx.x; r < ROWS; r += blockDim.x) {
        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < C4_TEAMS; ++e) s += tsum[e][r];
        out[r] = s;
    }
}

__global__ void c4_reduce_kernel(const float* __restrict__ part, const float* __restrict__ s2p, float* __restrict__ y,
                                 int64_t y_stride, int rows, const int32_t* __restrict__ idx) {
    const int j = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < C4_SEG; ++s) sum += part[((size_t) j * C4_SEG + s) * rows + r];
    y[(size_t) (idx ? idx[j] : j) * y_stride + r] = sum * s2p[j];
}

}  // namespace

void nemo_c4_from_ggml(const uint8_t* ggml, int64_t rows, int64_t cols, uint8_t* c4, void* stream) {
    const int64_t n = (cols / 16) * (rows / 2);
    c4_from_ggml_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(ggml, (int) rows, (int) cols,
                                                                                        c4);
}

void nemo_c4_compact(const float* x, int64_t x_stride, int64_t cols, int64_t n, uint16_t* list, float* vals,
                     int* count, void* stream) {
    if (n <= 0 || cols > C4_MAXK) return;
    c4_compact_kernel<<<(unsigned) n, C4_WARPS * 32, 0, (cudaStream_t) stream>>>(x, x_stride, (int) cols, list, vals,
                                                                                 count);
}

void nemo_c4_gemv_many(const uint8_t* const* w, const float* scale2, const uint16_t* list, const float* vals,
                       const int* count, float* part, float* y, int64_t y_stride, int64_t rows, int64_t cols, int64_t n,
                       const int32_t* idx, void* stream) {
    if (n <= 0 || rows != 1024 || cols > C4_MAXK) return;
    cudaStream_t st = (cudaStream_t) stream;
    c4_gemv_kernel<1024><<<dim3((unsigned) n, C4_SEG), C4_TEAMS * 1024 / 8, 0, st>>>(w, list, vals, count, part, (int) cols);
    c4_reduce_kernel<<<dim3((unsigned) ((rows + 255) / 256), (unsigned) n), 256, 0, st>>>(part, scale2, y, y_stride,
                                                                                         (int) rows, idx);
}

int64_t nemo_c4_part_floats(int64_t rows, int64_t n) { return n * C4_SEG * rows; }

}  // namespace strata::kernels
