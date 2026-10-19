// src/kernels/nemo_q8_parity.cpp - strata-nemotron: the W8A8 path (nemo_q8) against the host.
//
//   quant     nemo_q8_quant_rows = the same rounding on the host (amax / 127, nearest even), byte for byte
//   layout    tiled Q8_0 -> Q8N gives the GGUF rows' bytes and scales; tiled BF16 -> Q8N = the host's Q8_0 rounding
//   gemm      Y against a float64 sum of the decoded operands (the int8 products are exact, only the FP32 sums of
//             the scaled blocks differ); accumulate; M not a multiple of the tile
//   speed     M = 16384 on the attention's q (4096 x 4096) and fc1 (1024 x 4096) shapes, against cuBLAS BF16 (the
//             path it replaces, nemo_gemm_bf16)
//
// Usage: nemo_q8_parity
#include "strata/kernels/nemo_gemm.hpp"
#include "strata/kernels/nemo_gemv.hpp"
#include "strata/kernels/nemo_q8.hpp"
#include "strata/kernels/nemo_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

#define CK(x)                                                                                                \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
            std::exit(2);                                                                                    \
        }                                                                                                    \
    } while (0)

template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void quant_host(const float* x, int64_t K, int8_t* q, float* s) {
    for (int64_t b = 0; b < K / 32; ++b) {
        float a = 0.0f;
        for (int i = 0; i < 32; ++i) a = std::fmax(a, std::fabs(x[b * 32 + i]));
        const float inv = a > 0.0f ? 127.0f / a : 0.0f;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) std::nearbyint(x[b * 32 + i] * inv);
        s[b] = a / 127.0f;
    }
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float f_of_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("%-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
}

}  // namespace

int main() {
    using strata::kernels::NemoWType;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t M = 300, N = 384, K = 1024;

    // activations (a few outliers, as after an RMS norm)
    std::vector<float> x((size_t) M * K);
    for (float& v : x) v = nd(rng);
    for (int i = 0; i < 64; ++i) x[(size_t) (rng() % (M * K))] *= 30.0f;
    float* dx = dev(x);
    int8_t* daq = nullptr;
    float* das = nullptr;
    CK(cudaMalloc(&daq, (size_t) M * K));
    CK(cudaMalloc(&das, (size_t) M * K / 32 * 4));
    strata::kernels::nemo_q8_quant_rows(dx, K, M, K, daq, das, nullptr);
    std::vector<int8_t> hq((size_t) M * K), aq = host(daq, (size_t) M * K);
    std::vector<float> hs((size_t) M * K / 32), as = host(das, (size_t) M * K / 32);
    for (int64_t r = 0; r < M; ++r) quant_host(x.data() + r * K, K, hq.data() + r * K, hs.data() + r * K / 32);
    check(hq == aq && hs == as, "quant: device = host, byte for byte");

    // Q8_0 weights: GGUF rows (FP16 scale + 32 int8 a block) -> tiled -> Q8N
    std::vector<uint8_t> w8((size_t) N * K / 32 * 34);
    std::vector<int8_t> wq((size_t) N * K);
    std::vector<float> ws((size_t) N * K / 32);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t b = 0; b < K / 32; ++b) {
            uint8_t* blk = w8.data() + (size_t) (r * (K / 32) + b) * 34;
            const __half d = __float2half(0.001f + 0.02f * std::fabs(nd(rng)));
            std::memcpy(blk, &d, 2);
            ws[(size_t) r * (K / 32) + b] = __half2float(d);
            for (int i = 0; i < 32; ++i) {
                const int8_t v = (int8_t) ((int) (rng() % 255) - 127);
                blk[2 + i] = (uint8_t) v;
                wq[(size_t) r * K + b * 32 + i] = v;
            }
        }
    std::vector<uint8_t> tl((size_t) strata::kernels::nemo_tiled_bytes(NemoWType::Q8_0, N, K));
    strata::kernels::nemo_tiled_repack(NemoWType::Q8_0, w8.data(), N, K, tl.data());
    uint8_t* dtl = dev(tl);
    int8_t* dbq = nullptr;
    float* dbs = nullptr;
    CK(cudaMalloc(&dbq, (size_t) N * K));
    CK(cudaMalloc(&dbs, (size_t) N * K / 32 * 4));
    strata::kernels::nemo_q8_from_tiled_q8(dtl, N, K, dbq, dbs, nullptr);
    check(host(dbq, (size_t) N * K) == wq && host(dbs, (size_t) N * K / 32) == ws, "layout: tiled Q8_0 -> Q8N exact");

    // BF16 weights -> tiled -> Q8N (the host's Q8_0 rounding of the BF16 values)
    {
        std::vector<uint16_t> wb((size_t) N * K);
        std::vector<float> wf((size_t) N * K);
        for (size_t i = 0; i < wb.size(); ++i) {
            wb[i] = bf16_of(0.05f * nd(rng));
            wf[i] = f_of_bf16(wb[i]);
        }
        std::vector<uint8_t> tb((size_t) strata::kernels::nemo_tiled_bytes(NemoWType::BF16, N, K));
        strata::kernels::nemo_tiled_repack(NemoWType::BF16, (const uint8_t*) wb.data(), N, K, tb.data());
        uint8_t* dtb = dev(tb);
        strata::kernels::nemo_q8_from_tiled_bf16(dtb, N, K, dbq, dbs, nullptr);
        std::vector<int8_t> eq((size_t) N * K);
        std::vector<float> es((size_t) N * K / 32);
        for (int64_t r = 0; r < N; ++r) quant_host(wf.data() + r * K, K, eq.data() + r * K, es.data() + r * K / 32);
        check(host(dbq, (size_t) N * K) == eq && host(dbs, (size_t) N * K / 32) == es, "layout: tiled BF16 -> Q8N = host Q8_0");
        CK(cudaFree(dtb));
        strata::kernels::nemo_q8_from_tiled_q8(dtl, N, K, dbq, dbs, nullptr);   // back to the Q8_0 weights
    }

    // the GEMM against float64 over the decoded operands, alpha, accumulate
    const float alpha = 0.75f;
    std::vector<float> y0((size_t) M * N);
    for (float& v : y0) v = nd(rng);
    float* dy = dev(y0);
    strata::kernels::nemo_q8_gemm(daq, das, dbq, dbs, alpha, dy, N, M, N, K, true, nullptr);
    std::vector<float> y = host(dy, (size_t) M * N);
    double worst = 0, scale = 0;
    for (int64_t r = 0; r < M; ++r)
        for (int64_t n = 0; n < N; ++n) {
            double sum = 0;
            for (int64_t b = 0; b < K / 32; ++b) {
                int64_t dot = 0;
                for (int i = 0; i < 32; ++i) dot += (int64_t) hq[(size_t) r * K + b * 32 + i] * wq[(size_t) n * K + b * 32 + i];
                sum += (double) dot * hs[(size_t) r * K / 32 + b] * ws[(size_t) n * K / 32 + b];
            }
            const double ref = alpha * sum + y0[(size_t) r * N + n];
            worst = std::fmax(worst, std::fabs(ref - y[(size_t) r * N + n]));
            scale = std::fmax(scale, std::fabs(ref));
        }
    std::printf("  gemm: worst |err| %.3g of max |y| %.3g\n", worst, scale);
    check(worst <= 2e-6 * scale + 1e-6, "gemm: = float64 (rel 2e-6), alpha, accumulate, M % 128 != 0");

    // speed: M = 16384 on the q (N 4096, K 4096) and fc1 (N 1024, K 4096) shapes
    {
        const int64_t SM = 16384;
        for (int64_t SN : {4096, 1024}) {
            const int64_t SK = 4096;
            int8_t *a = nullptr, *b = nullptr;
            float *sa = nullptr, *sb = nullptr, *yy = nullptr, *xf = nullptr;
            uint16_t *xb = nullptr, *wbb = nullptr;
            CK(cudaMalloc(&a, (size_t) SM * SK));
            CK(cudaMalloc(&b, (size_t) SN * SK));
            CK(cudaMalloc(&sa, (size_t) SM * SK / 8));
            CK(cudaMalloc(&sb, (size_t) SN * SK / 8));
            CK(cudaMalloc(&yy, (size_t) SM * SN * 4));
            CK(cudaMalloc(&xf, (size_t) SM * SK * 4));
            CK(cudaMalloc(&xb, (size_t) SM * SK * 2));
            CK(cudaMalloc(&wbb, (size_t) SN * SK * 2));
            CK(cudaMemset(a, 1, (size_t) SM * SK));
            CK(cudaMemset(b, 1, (size_t) SN * SK));
            CK(cudaMemset(sa, 0, (size_t) SM * SK / 8));
            CK(cudaMemset(sb, 0, (size_t) SN * SK / 8));
            CK(cudaMemset(xf, 0, (size_t) SM * SK * 4));
            CK(cudaMemset(xb, 0, (size_t) SM * SK * 2));
            CK(cudaMemset(wbb, 0, (size_t) SN * SK * 2));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            float ms = 0, mq = 0, mb = 0;
            for (int rep = 0; rep < 2; ++rep) {
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 10; ++i) strata::kernels::nemo_q8_gemm(a, sa, b, sb, 1.0f, yy, SN, SM, SN, SK, false, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 10; ++i) strata::kernels::nemo_q8_quant_rows(xf, SK, SM, SK, a, sa, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&mq, e0, e1));
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 10; ++i) strata::kernels::nemo_gemm_bf16(xb, wbb, 1.0f, yy, SN, SN, SK, SM, false, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&mb, e0, e1));
            }
            const double fl = 2.0 * SM * SN * SK;
            std::printf("  speed %lld x %lld x %lld: q8 %.3f ms (%.0f TOP/s), quant %.3f ms; cuBLAS BF16 %.3f ms (%.0f TFLOP/s)\n",
                        (long long) SM, (long long) SN, (long long) SK, ms / 10, fl / (ms / 10 * 1e-3) / 1e12, mq / 10,
                        mb / 10, fl / (mb / 10 * 1e-3) / 1e12);
            cudaFree(a); cudaFree(b); cudaFree(sa); cudaFree(sb); cudaFree(yy); cudaFree(xf); cudaFree(xb); cudaFree(wbb);
        }
    }
    std::printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
