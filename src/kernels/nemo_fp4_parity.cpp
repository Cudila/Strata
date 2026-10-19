// src/kernels/nemo_fp4_parity.cpp - strata-nemotron: the W4A4 path (nemo_fp4) against the host.
//
//   quant     nemo_fp4_quant_rows = the same quantization on the host (cuda_fp8.h / cuda_fp4.h's conversions, which
//             are host functions too), byte for byte, with a gather and relu^2
//   layout    ggml rows -> N4 and the tiled layout -> N4 give the same bytes
//   gemm      Y against a float64 sum of the decoded operands (the products are exact, only FP32 summation differs);
//             relu^2 (5e-5: squaring doubles the relative error) and accumulate; the grouped form (B in N4, and B
//             straight from the ggml rows) (three problems, uneven rows) against the plain one
//
// Usage: nemo_fp4_parity
#include "strata/kernels/nemo_fp4.hpp"
#include "strata/kernels/nemo_gemv.hpp"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
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

double e2m1(unsigned c) {
    static const double v[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
double ue4m3(unsigned b) {
    const unsigned e = (b >> 3) & 15, m = b & 7;
    return e ? std::ldexp(1.0 + m / 8.0, (int) e - 7) : std::ldexp(m / 8.0, -6);
}

template <class T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

/// The host twin of quant_rows_kernel.
void quant_host(const std::vector<float>& x, int64_t ldx, const std::vector<int32_t>& gather, int64_t M, int64_t K,
                bool relu2, std::vector<uint8_t>& q, std::vector<uint8_t>& s, std::vector<float>& gs) {
    q.assign((size_t) (M * K / 2), 0);
    s.assign((size_t) (M * K / 16), 0);
    gs.assign((size_t) M, 0);
    for (int64_t r = 0; r < M; ++r) {
        const float* xr = x.data() + (size_t) (gather.empty() ? r : gather[(size_t) r]) * ldx;
        auto act = [&](int64_t c) {
            float v = xr[c];
            if (relu2) { v = std::fmax(v, 0.0f); v *= v; }
            return v;
        };
        float m = 0;
        for (int64_t c = 0; c < K; ++c) m = std::fmax(m, std::fabs(act(c)));
        const float g = m > 0 ? m / (448.0f * 6.0f) : 1.0f;
        gs[(size_t) r] = g;
        for (int64_t b = 0; b < K / 16; ++b) {
            float v[16], am = 0;
            for (int j = 0; j < 16; ++j) { v[j] = act(b * 16 + j); am = std::fmax(am, std::fabs(v[j])); }
            const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(am / (6.0f * g), __NV_SATFINITE, __NV_E4M3);
            __nv_fp8_e4m3 sv;
            sv.__x = s8;
            const float sb = (float) sv * g;
            const float inv = sb > 0.0f ? 1.0f / sb : 0.0f;
            for (int j = 0; j < 16; ++j) {
                const unsigned c = (unsigned) __nv_cvt_float_to_fp4(v[j] * inv, __NV_E2M1, cudaRoundNearest) & 15u;
                q[(size_t) (r * K / 2 + b * 8 + j / 2)] |= (uint8_t) (c << (4 * (j & 1)));
            }
            s[(size_t) (r * K / 16 + b)] = s8;
        }
    }
}

/// value (r, k) of an N4 operand
double n4(const std::vector<uint8_t>& q, const std::vector<uint8_t>& s, int64_t K, int64_t r, int64_t k) {
    const uint8_t b = q[(size_t) (r * K / 2 + k / 2)];
    return e2m1((k & 1) ? b >> 4 : b & 15) * ue4m3(s[(size_t) (r * K / 16 + k / 16)]);
}

/// random ggml block_nvfp4 rows: scales in a moderate e4m3 range
std::vector<uint8_t> rand_nvfp4(int64_t rows, int64_t cols, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) (rows * cols / 64 * 36));
    std::uniform_int_distribution<int> byte(0, 255), sc(3 << 3, (10 << 3) | 7);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < cols / 64; ++b) {
            uint8_t* blk = w.data() + (size_t) (r * cols / 64 + b) * 36;
            for (int i = 0; i < 4; ++i) blk[i] = (uint8_t) sc(rng);
            for (int i = 4; i < 36; ++i) blk[i] = (uint8_t) byte(rng);
        }
    return w;
}

}  // namespace

int main() {
    if (!strata::kernels::nemo_fp4_ok()) {
        std::printf("nemo_fp4_parity: no FP4 mma on this build or device: SKIP\n");
        return 0;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    bool ok = true;
    struct Case { int64_t N, K, M; };
    for (const Case cs : {Case{256, 1024, 37}, Case{384, 2688, 300}, Case{128, 4096, 1}, Case{1024, 1024, 129}}) {
        const int64_t N = cs.N, K = cs.K, M = cs.M;
        // activations: rows of different magnitudes, a gather over 2M source rows
        std::vector<float> x((size_t) (2 * M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * std::exp2((float) ((i / K) % 7) - 3.0f);
        std::vector<int32_t> gat((size_t) M);
        for (int64_t r = 0; r < M; ++r) gat[(size_t) r] = (int32_t) ((r * 7 + 3) % (2 * M));
        float* dx = up(x);
        int32_t* dg = up(gat);
        uint8_t *aq, *as;
        float* ags;
        CK(cudaMalloc(&aq, M * K / 2));
        CK(cudaMalloc(&as, M * K / 16));
        CK(cudaMalloc(&ags, M * sizeof(float)));
        bool q_same = true;
        std::vector<uint8_t> hq, hs;
        std::vector<float> hg;
        for (int relu = 0; relu < 2; ++relu) {
            strata::kernels::nemo_fp4_quant_rows(dx, K, dg, M, K, relu != 0, aq, as, ags, st);
            CK(cudaStreamSynchronize(st));
            quant_host(x, K, gat, M, K, relu != 0, hq, hs, hg);
            q_same &= down(aq, (size_t) (M * K / 2)) == hq && down(as, (size_t) (M * K / 16)) == hs &&
                      down(ags, (size_t) M) == hg;
        }
        // the last quantization (relu^2) is A from here on: hq, hs, hg
        // weights: ggml -> N4, tiled -> N4
        const std::vector<uint8_t> w = rand_nvfp4(N, K, rng);
        uint8_t* dw = up(w);
        const uint8_t** dptr;
        CK(cudaMalloc(&dptr, sizeof(void*)));
        CK(cudaMemcpy(dptr, &dw, sizeof(void*), cudaMemcpyHostToDevice));
        uint8_t *bq, *bs, *tq, *ts;
        CK(cudaMalloc(&bq, N * K / 2));
        CK(cudaMalloc(&bs, N * K / 16));
        CK(cudaMalloc(&tq, N * K / 2));
        CK(cudaMalloc(&ts, N * K / 16));
        strata::kernels::nemo_fp4_from_ggml_many(dptr, 1, N, K, bq, bs, st);
        std::vector<uint8_t> tl((size_t) strata::kernels::nemo_tiled_bytes(strata::kernels::NemoWType::NVFP4, N, K));
        strata::kernels::nemo_tiled_repack(strata::kernels::NemoWType::NVFP4, w.data(), N, K, tl.data());
        uint8_t* dtl = up(tl);
        strata::kernels::nemo_fp4_from_tiled(dtl, N, K, tq, ts, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<uint8_t> wq = down(bq, (size_t) (N * K / 2)), ws = down(bs, (size_t) (N * K / 16));
        const bool l_same = wq == down(tq, wq.size()) && ws == down(ts, ws.size());
        // N4 of the weights against the ggml decode (value j < 8 of a sub-block: low nibble of byte j)
        bool w_dec = true;
        for (int64_t r = 0; r < N && w_dec; ++r)
            for (int64_t k = 0; k < K; ++k) {
                const uint8_t* blk = w.data() + (size_t) (r * K / 64 + k / 64) * 36;
                const int sb = (int) ((k % 64) / 16), j = (int) (k % 16);
                const uint8_t qb = blk[4 + sb * 8 + (j % 8)];
                const double ref = ue4m3(blk[sb]) * e2m1(j < 8 ? (qb & 15) : (qb >> 4));
                if (n4(wq, ws, K, r, k) != ref) { w_dec = false; break; }
            }
        // the GEMM: relu^2 epilogue off, accumulate on (Y starts at 1)
        const float alpha = 3.7e-3f;
        std::vector<float> y0((size_t) (M * N), 1.0f);
        float* dy = up(y0);
        strata::kernels::nemo_fp4_gemm(aq, as, ags, bq, bs, alpha, dy, N, M, N, K, false, true, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<float> y = down(dy, (size_t) (M * N));
        double worst = 0, rms = 0;
        std::vector<double> ref((size_t) (M * N));
        for (int64_t r = 0; r < M; ++r)
            for (int64_t n = 0; n < N; ++n) {
                double sacc = 0;
                for (int64_t k = 0; k < K; ++k) sacc += n4(hq, hs, K, r, k) * n4(wq, ws, K, n, k);
                ref[(size_t) (r * N + n)] = (double) alpha * hg[(size_t) r] * sacc;
                rms += ref[(size_t) (r * N + n)] * ref[(size_t) (r * N + n)];
            }
        rms = std::sqrt(rms / (double) ref.size()) + 1e-30;
        for (size_t i = 0; i < ref.size(); ++i) worst = std::fmax(worst, std::fabs(y[i] - 1.0 - ref[i]) / rms);
        // relu^2, no accumulate
        strata::kernels::nemo_fp4_gemm(aq, as, ags, bq, bs, alpha, dy, N, M, N, K, true, false, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<float> yr = down(dy, (size_t) (M * N));
        double worst_r = 0, rms_r = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double v = std::fmax(ref[i], 0.0) * std::fmax(ref[i], 0.0);
            rms_r += v * v;
        }
        rms_r = std::sqrt(rms_r / (double) ref.size()) + 1e-30;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double v = std::fmax(ref[i], 0.0) * std::fmax(ref[i], 0.0);
            worst_r = std::fmax(worst_r, std::fabs(yr[i] - v) / rms_r);
        }
        // grouped: three problems over the same A rows split unevenly, each with this B and its own alpha
        bool g_same = true;
        if (M >= 3) {
            const int64_t cut[4] = {0, M / 5, M / 5 + (M + 1) / 2, M};
            uint8_t *gq, *gsc;
            CK(cudaMalloc(&gq, 3 * N * K / 2));
            CK(cudaMalloc(&gsc, 3 * N * K / 16));
            std::vector<float> al = {alpha, 2 * alpha, 0.5f * alpha};
            std::vector<int32_t> tiles;
            for (int p = 0; p < 3; ++p) {
                CK(cudaMemcpy(gq + p * N * K / 2, bq, N * K / 2, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(gsc + p * N * K / 16, bs, N * K / 16, cudaMemcpyDeviceToDevice));
                for (int64_t r0 = cut[p]; r0 < cut[p + 1]; r0 += 128) {
                    tiles.push_back(p);
                    tiles.push_back((int32_t) r0);
                    tiles.push_back((int32_t) std::min<int64_t>(r0 + 128, cut[p + 1]));
                }
            }
            float* dal = up(al);
            int32_t* dt = up(tiles);
            strata::kernels::nemo_fp4_gemm_grouped(aq, as, ags, gq, gsc, dal, dt, (int) (tiles.size() / 3), dy, N, K,
                                                   false, st);
            CK(cudaStreamSynchronize(st));
            const std::vector<float> yg = down(dy, (size_t) (M * N));
            // plain GEMM per problem's alpha, rows of that problem
            float* dy2 = up(y0);
            for (int p = 0; p < 3; ++p) {
                const int64_t m = cut[p + 1] - cut[p];
                if (m == 0) continue;
                strata::kernels::nemo_fp4_gemm(aq + cut[p] * K / 2, as + cut[p] * K / 16, ags + cut[p], bq, bs, al[(size_t) p],
                                               dy2 + cut[p] * N, N, m, N, K, false, false, st);
            }
            CK(cudaStreamSynchronize(st));
            g_same = std::memcmp(yg.data(), down(dy2, (size_t) (M * N)).data(), yg.size() * sizeof(float)) == 0;
            // the same problems with B read from the ggml rows themselves (bptr): bit-identical
            std::vector<const uint8_t*> bp(3, dw);
            const uint8_t** dbp;
            CK(cudaMalloc(&dbp, 3 * sizeof(void*)));
            CK(cudaMemcpy(dbp, bp.data(), 3 * sizeof(void*), cudaMemcpyHostToDevice));
            CK(cudaMemset(dy, 0, (size_t) M * N * sizeof(float)));
            strata::kernels::nemo_fp4_gemm_grouped(aq, as, ags, nullptr, nullptr, dal, dt, (int) (tiles.size() / 3), dy, N,
                                                   K, false, st, nullptr, nullptr, dbp);
            CK(cudaStreamSynchronize(st));
            g_same &= std::memcmp(yg.data(), down(dy, (size_t) (M * N)).data(), yg.size() * sizeof(float)) == 0;
            cudaFree(dbp);
            cudaFree(gq); cudaFree(gsc); cudaFree(dal); cudaFree(dt); cudaFree(dy2);
        }
        const bool pass = q_same && l_same && w_dec && worst < 1e-5 && worst_r < 5e-5 && g_same;
        std::printf("  N %5lld K %5lld M %4lld: quant = host %s; ggml/tiled -> N4 same %s, decode %s; gemm worst %.2e, "
                    "relu2 %.2e; grouped = plain %s  %s\n", (long long) N, (long long) K, (long long) M,
                    q_same ? "yes" : "NO", l_same ? "yes" : "NO", w_dec ? "ok" : "BAD", worst, worst_r,
                    g_same ? "yes" : "NO", pass ? "ok" : "FAIL");
        ok &= pass;
        cudaFree(dx); cudaFree(dg); cudaFree(aq); cudaFree(as); cudaFree(ags); cudaFree(dw); cudaFree(dptr);
        cudaFree(bq); cudaFree(bs); cudaFree(tq); cudaFree(ts); cudaFree(dtl); cudaFree(dy);
    }
    // throughput: an 8192 x 18560 x 4096 product (in_proj of a chunk)
    {
        const int64_t M = 8192, N = 18560, K = 4096;
        uint8_t *aq, *as, *bq, *bs;
        float *ags, *y;
        CK(cudaMalloc(&aq, M * K / 2)); CK(cudaMalloc(&as, M * K / 16)); CK(cudaMalloc(&ags, M * 4));
        CK(cudaMalloc(&bq, N * K / 2)); CK(cudaMalloc(&bs, N * K / 16)); CK(cudaMalloc(&y, M * N * 4));
        CK(cudaMemset(aq, 0x11, M * K / 2)); CK(cudaMemset(as, 0x38, M * K / 16));
        CK(cudaMemset(bq, 0x22, N * K / 2)); CK(cudaMemset(bs, 0x38, N * K / 16));
        std::vector<float> one((size_t) M, 1.0f);
        CK(cudaMemcpy(ags, one.data(), M * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        strata::kernels::nemo_fp4_gemm(aq, as, ags, bq, bs, 1.0f, y, N, M, N, K, false, false, st);
        cudaEventRecord(e0, st);
        for (int i = 0; i < 5; ++i) strata::kernels::nemo_fp4_gemm(aq, as, ags, bq, bs, 1.0f, y, N, M, N, K, false, false, st);
        cudaEventRecord(e1, st);
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= 5;
        std::printf("  throughput %lld x %lld x %lld: %.3f ms, %.0f TFLOP/s\n", (long long) M, (long long) N, (long long) K,
                    ms, 2.0 * M * N * K / ms / 1e9);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
