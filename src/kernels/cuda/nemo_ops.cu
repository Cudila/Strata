// src/kernels/cuda/nemo_ops.cu - see include/strata/kernels/nemo_ops.hpp.
#include "strata/kernels/nemo_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>

#include <cmath>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void rms_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* orow = out + (size_t) blockIdx.x * n;
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float s = rsqrtf(red[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) orow[i] = xr[i] * s * w[i];
}

__global__ void bf16_rows_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}

__global__ void gather_kernel(const float* __restrict__ in, const int32_t* __restrict__ idx, float* __restrict__ out,
                              int n) {
    const float* src = in + (size_t) idx[blockIdx.x] * n;
    float* dst = out + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

__global__ void scatter_kernel(const float* __restrict__ in, const int32_t* __restrict__ idx, float* __restrict__ out,
                               int n) {
    const float* src = in + (size_t) blockIdx.x * n;
    float* dst = out + (size_t) idx[blockIdx.x] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

__global__ void weighted_kernel(const float* __restrict__ parts, const float* __restrict__ w, float* __restrict__ out,
                                int k, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += w[(size_t) t * k + j] * parts[((size_t) t * k + j) * n + i];
    out[(size_t) t * n + i] = acc;
}

/// The same for BF16 parts, 8 values a thread (one 16-byte load a part).
__global__ void weighted_bf16_kernel(const uint16_t* __restrict__ parts, const float* __restrict__ w,
                                     float* __restrict__ out, int k, int n) {
    const int t = blockIdx.y;
    const int i8 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i8 * 8 >= n) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = 0; j < k; ++j) {
        const float wj = w[(size_t) t * k + j];
        const uint4 u = reinterpret_cast<const uint4*>(parts + ((size_t) t * k + j) * n)[i8];
        const uint32_t h[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            acc[2 * q] += wj * __uint_as_float(h[q] << 16);
            acc[2 * q + 1] += wj * __uint_as_float(h[q] & 0xffff0000u);
        }
    }
    float4* o = reinterpret_cast<float4*>(out + (size_t) t * n) + 2 * i8;
    o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

__global__ void accum_kernel(const float* __restrict__ out, const int32_t* __restrict__ seg_start,
                             const int32_t* __restrict__ seg_token, const int32_t* __restrict__ row,
                             const int32_t* __restrict__ wid, const float* __restrict__ w, float* __restrict__ routed,
                             int n) {
    const int sgi = blockIdx.x;
    const int r0 = seg_start[sgi], r1 = seg_start[sgi + 1];
    float* dst = routed + (size_t) seg_token[sgi] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = dst[i];
        for (int r = r0; r < r1; ++r) acc += w[wid[r]] * out[(size_t) row[r] * n + i];
        dst[i] = acc;
    }
}

__global__ void accum_f64_kernel(const float* __restrict__ out, const int32_t* __restrict__ seg_start,
                                 const int32_t* __restrict__ seg_token, const int32_t* __restrict__ row,
                                 const int32_t* __restrict__ wid, const float* __restrict__ w,
                                 double* __restrict__ routed, int n) {
    const int sgi = blockIdx.x;
    const int r0 = seg_start[sgi], r1 = seg_start[sgi + 1];
    double* dst = routed + (size_t) seg_token[sgi] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = dst[i];
        for (int r = r0; r < r1; ++r) acc += (double) w[wid[r]] * (double) out[(size_t) row[r] * n + i];
        dst[i] = acc;
    }
}

__global__ void f64_to_f32_kernel(const double* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = (float) x[i];
}

/// LayerNorm without bias: (x - mean) * rsqrt(var + eps) * w (ggml_norm, then the weight).
__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                  int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* orow = out + (size_t) blockIdx.x * n;
    __shared__ float red[32];
    __shared__ float stat;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    s = warp_sum(s);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) stat = v / (float) n;
    }
    __syncthreads();
    const float mean = stat;
    float q = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    q = warp_sum(q);
    __syncthreads();
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = q;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (int) (blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) stat = rsqrtf(v / (float) n + eps);
    }
    __syncthreads();
    const float r = stat;
    for (int i = threadIdx.x; i < n; i += blockDim.x) orow[i] = (xr[i] - mean) * r * w[i];
}

/// The first index of the largest value, per row (one block a row).
__global__ void argmax_kernel(const float* __restrict__ x, int64_t n, int32_t* __restrict__ out,
                              float* __restrict__ prob) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v > bv) { bv = v; bi = (int) i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    if (threadIdx.x % 32 == 0) { sv[threadIdx.x / 32] = bv; si[threadIdx.x / 32] = bi; }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = (int) (blockDim.x / 32);
        bv = threadIdx.x < nw ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < nw ? si[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (threadIdx.x == 0) { out[blockIdx.x] = bi; sv[0] = bv; }
    }
    if (!prob) return;
    // the top token's softmax probability: 1 / sum exp(x - max)
    __syncthreads();
    const float mx = sv[0];
    float acc = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) acc += __expf(xr[i] - mx);
    acc = warp_sum(acc);
    __shared__ float ps[32];
    if (threadIdx.x % 32 == 0) ps[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int q = 0; q < (int) (blockDim.x / 32); ++q) t += ps[q];
        prob[blockIdx.x] = 1.0f / t;
    }
}

/// Resident experts: entry j's up / down pointers and scale_2 from the routed id ids[j].
__global__ void resident_ptrs_kernel(const int32_t* __restrict__ ids, const uint8_t* base, uint64_t stride,
                                     uint64_t ub, const float* __restrict__ s_up, const float* __restrict__ s_dn,
                                     const uint8_t** __restrict__ ptr, float* __restrict__ s2, int k) {
    const int j = threadIdx.x;
    if (j >= k) return;
    const int e = ids[j];
    ptr[j] = base + (size_t) e * stride;
    ptr[k + j] = base + (size_t) e * stride + ub;
    s2[j] = s_up[e];
    s2[k + j] = s_dn[e];
}

}  // namespace

void nemo_layer_norm_rows(const float* x, const float* w, float* out, int64_t T, int64_t n, float eps, void* stream) {
    if (T > 0) layer_norm_kernel<<<(unsigned) T, 1024, 0, (cudaStream_t) stream>>>(x, w, out, (int) n, eps);
}

void nemo_argmax_rows(const float* x, int64_t T, int64_t n, int32_t* out, void* stream, float* prob) {
    if (T > 0) argmax_kernel<<<(unsigned) T, 1024, 0, (cudaStream_t) stream>>>(x, n, out, prob);
}

void nemo_resident_expert_ptrs(const int32_t* ids, const uint8_t* base, uint64_t stride, uint64_t ub,
                               const float* s_up, const float* s_dn, const uint8_t** ptr, float* s2, int64_t k,
                               void* stream) {
    resident_ptrs_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, base, stride, ub, s_up, s_dn, ptr, s2, (int) k);
}

void nemo_accum_rows(const float* out, const int32_t* seg_start, const int32_t* seg_token, const int32_t* row,
                     const int32_t* wid, const float* w, float* routed, int64_t nseg, int64_t n, void* stream) {
    if (nseg > 0)
        accum_kernel<<<(unsigned) nseg, 256, 0, (cudaStream_t) stream>>>(out, seg_start, seg_token, row, wid, w, routed,
                                                                         (int) n);
}

void nemo_accum_rows_f64(const float* out, const int32_t* seg_start, const int32_t* seg_token, const int32_t* row,
                         const int32_t* wid, const float* w, double* routed, int64_t nseg, int64_t n, void* stream) {
    if (nseg > 0)
        accum_f64_kernel<<<(unsigned) nseg, 256, 0, (cudaStream_t) stream>>>(out, seg_start, seg_token, row, wid, w,
                                                                             routed, (int) n);
}

void nemo_f64_to_f32(const double* x, float* y, int64_t n, void* stream) {
    if (n > 0)
        f64_to_f32_kernel<<<(unsigned) std::min<int64_t>((n + 255) / 256, 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
}

void nemo_rms_norm_rows(const float* x, const float* w, float* out, int64_t T, int64_t n, float eps, void* stream) {
    // a few rows (decode): a 1024-thread block per row, the row's one block is the whole kernel's latency
    if (T > 0) rms_rows_kernel<<<(unsigned) T, T <= 16 ? 1024 : 256, 0, (cudaStream_t) stream>>>(x, w, out, (int) n, eps);
}

void nemo_bf16_rows_to_f32(const uint16_t* in, float* out, int64_t T, int64_t n, void* stream) {
    const int64_t total = T * n;
    if (total > 0) bf16_rows_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, (cudaStream_t) stream>>>(in, out, total);
}

void nemo_gather_rows(const float* in, const int32_t* idx, float* out, int64_t count, int64_t n, void* stream) {
    if (count > 0) gather_kernel<<<(unsigned) count, 256, 0, (cudaStream_t) stream>>>(in, idx, out, (int) n);
}

void nemo_scatter_rows(const float* in, const int32_t* idx, float* out, int64_t count, int64_t n, void* stream) {
    if (count > 0) scatter_kernel<<<(unsigned) count, 256, 0, (cudaStream_t) stream>>>(in, idx, out, (int) n);
}

void nemo_weighted_rows_bf16(const uint16_t* parts, const float* w, float* out, int64_t T, int64_t k, int64_t n,
                             void* stream) {
    if (T > 0)
        weighted_bf16_kernel<<<dim3((unsigned) ((n / 8 + 127) / 128), (unsigned) T), 128, 0, (cudaStream_t) stream>>>(
            parts, w, out, (int) k, (int) n);
}

void nemo_weighted_rows(const float* parts, const float* w, float* out, int64_t T, int64_t k, int64_t n,
                        void* stream) {
    if (T > 0)
        weighted_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
            parts, w, out, (int) k, (int) n);
}

}  // namespace strata::kernels
