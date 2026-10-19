// src/kernels/nemo_gemm_parity.cpp - strata-nemotron: nemo_gemm against a float64 host decode of the same bytes, for
// NVFP4 / Q8_0 / BF16 weights and several token counts, and the property the verify window relies on: a token's row
// of Y is bit-identical whether it is computed alone (M = 1) or in a pass with others.
//
// The NVFP4 decode here is written independently of the kernels (from ggml-common.h's block_nvfp4 and the converter's
// _nvfp4_pack), so a shared misreading would have to be made twice.
//
// Usage: nemo_gemm_parity
#include "strata/kernels/nemo_gemm.hpp"
#include "strata/kernels/nemo_gemv.hpp"
#include "strata/kernels/nemo_tc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using strata::kernels::NemoWType;

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
double f16(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    const double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
    return s ? -v : v;
}

/// Row r, column c of the matrix in its format.
double weight(NemoWType t, const std::vector<uint8_t>& w, uint64_t rb, int64_t r, int64_t c) {
    const uint8_t* row = w.data() + r * rb;
    switch (t) {
    case NemoWType::NVFP4: {
        const uint8_t* blk = row + (c / 64) * 36;
        const int s = (int) ((c % 64) / 16), j = (int) (c % 16);
        const uint8_t q = blk[4 + s * 8 + (j % 8)];
        return ue4m3(blk[s]) * e2m1(j < 8 ? (q & 15) : (q >> 4));
    }
    case NemoWType::Q8_0: {
        const uint8_t* blk = row + (c / 32) * 34;
        return f16((uint16_t) (blk[0] | (blk[1] << 8))) * (int8_t) blk[2 + c % 32];
    }
    case NemoWType::F32: {
        float f;
        std::memcpy(&f, row + c * 4, 4);
        return f;
    }
    case NemoWType::BF16: {
        uint16_t b;
        std::memcpy(&b, row + c * 2, 2);
        uint32_t u = (uint32_t) b << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    }
    return 0;
}

std::vector<uint8_t> random_matrix(NemoWType t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const uint64_t rb = strata::kernels::nemo_row_bytes(t, cols);
    std::vector<uint8_t> w(rb * rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t* row = w.data() + r * rb;
        if (t == NemoWType::NVFP4)
            for (int64_t b = 0; b < cols / 64; ++b)
                for (int s = 0; s < 4; ++s) row[b * 36 + s] = (uint8_t) (0x28 + byte(rng) % 24);   // scales 2^-2..2^1
        if (t == NemoWType::Q8_0)
            for (int64_t b = 0; b < cols / 32; ++b) {   // d in [2^-8, 2^-6): a finite, normal half
                const uint16_t d = (uint16_t) ((7u << 10) | (unsigned) (byte(rng) * 4));
                row[b * 34] = (uint8_t) (d & 0xff);
                row[b * 34 + 1] = (uint8_t) (d >> 8);
            }
        if (t == NemoWType::F32) {
            std::normal_distribution<float> nd2(0.0f, 1.0f);
            for (int64_t c = 0; c < cols; ++c) {
                const float f = nd2(rng);
                std::memcpy(row + c * 4, &f, 4);
            }
        }
        if (t == NemoWType::BF16)
            for (int64_t c = 0; c < cols; ++c) {   // exponent kept near 1: no inf / nan / denormals
                row[c * 2 + 1] = (uint8_t) ((row[c * 2 + 1] & 0x80) | 0x3c | (byte(rng) & 1));
            }
    }
    return w;
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    bool ok = true;
    struct Case { NemoWType t; const char* name; int64_t rows, cols; float s2; };
    const Case cases[] = {{NemoWType::NVFP4, "NVFP4", 1000, 1024, 0.37f}, {NemoWType::NVFP4, "NVFP4", 517, 2688, 1.0f},
                          {NemoWType::NVFP4, "NVFP4", 301, 8192, 0.5f}, {NemoWType::NVFP4, "NVFP4", 77, 256, 1.0f},
                          {NemoWType::Q8_0, "Q8_0", 1000, 4096, 1.0f}, {NemoWType::Q8_0, "Q8_0", 300, 8192, 1.0f},
                          {NemoWType::BF16, "BF16", 1000, 4096, 1.0f}, {NemoWType::BF16, "BF16", 512, 1024, 1.0f},
                          {NemoWType::F32, "F32", 512, 4096, 1.0f}};
    const int64_t Ms[] = {1, 3, 4, 7, 8, 9, 33, 100};
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    for (const Case& cs : cases) {
        const uint64_t rb = strata::kernels::nemo_row_bytes(cs.t, cs.cols);
        const std::vector<uint8_t> w = random_matrix(cs.t, cs.rows, cs.cols, rng);
        const int64_t Mmax = 100;
        std::vector<float> x((size_t) Mmax * cs.cols);
        for (auto& v : x) v = nd(rng);
        // the float64 reference
        std::vector<double> ref((size_t) Mmax * cs.rows);
        std::vector<double> wr((size_t) cs.cols);
        for (int64_t r = 0; r < cs.rows; ++r) {
            for (int64_t c = 0; c < cs.cols; ++c) wr[(size_t) c] = weight(cs.t, w, rb, r, c);
            for (int64_t m = 0; m < Mmax; ++m) {
                double a = 0;
                for (int64_t c = 0; c < cs.cols; ++c) a += wr[(size_t) c] * x[(size_t) (m * cs.cols + c)];
                ref[(size_t) (m * cs.rows + r)] = a * cs.s2;
            }
        }
        void* dw;
        float *dx, *dy;
        CK(cudaMalloc(&dw, w.size()));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x.size() * sizeof(float)));
        CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dy, (size_t) Mmax * cs.rows * sizeof(float)));
        std::vector<float> alone((size_t) Mmax * cs.rows);
        for (int64_t m = 0; m < Mmax; ++m)   // every token alone: the M = 1 kernel
            strata::kernels::nemo_gemm(cs.t, dw, cs.s2, dx + m * cs.cols, cs.cols, dy + m * cs.rows, cs.rows, cs.rows,
                                       cs.cols, 1, false, false, st);
        CK(cudaMemcpy(alone.data(), dy, alone.size() * sizeof(float), cudaMemcpyDeviceToHost));
        // the worst element's error relative to the RMS of the reference (a random dot can sit near zero, so a
        // per-element relative error says nothing there)
        double worst = 0, ss = 0;
        for (size_t i = 0; i < alone.size(); ++i) ss += ref[i] * ref[i];
        const double rms = std::sqrt(ss / (double) alone.size());
        for (size_t i = 0; i < alone.size(); ++i) worst = std::fmax(worst, std::fabs(alone[i] - ref[i]) / rms);
        bool same = true;
        for (int64_t M : Ms) {
            CK(cudaMemset(dy, 0, (size_t) Mmax * cs.rows * sizeof(float)));
            strata::kernels::nemo_gemm(cs.t, dw, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows, cs.cols, M, false, false, st);
            std::vector<float> got((size_t) M * cs.rows);
            CK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            same &= std::memcmp(got.data(), alone.data(), got.size() * sizeof(float)) == 0;
        }
        // relu2 + accumulate: y = 1 + relu(Wx)^2 on top of ones
        std::vector<float> ones((size_t) cs.rows, 1.0f);
        CK(cudaMemcpy(dy, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
        strata::kernels::nemo_gemm(cs.t, dw, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows, cs.cols, 1, true, true, st);
        std::vector<float> ra((size_t) cs.rows);
        CK(cudaMemcpy(ra.data(), dy, ra.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double worst_ra = 0;
        for (int64_t r = 0; r < cs.rows; ++r) {
            const double v = alone[(size_t) r] > 0 ? (double) alone[(size_t) r] * alone[(size_t) r] : 0.0;
            worst_ra = std::fmax(worst_ra, std::fabs(ra[(size_t) r] - (1.0 + v)) / (1.0 + v));
        }
        // the decode GEMV (nemo_gemv): the same error bound, M = 1..8 bit-identical to its own M = 1, relu2+accumulate
        double worst_v = 0;
        bool same_v = true;
        if (true) {
            std::vector<float> v1((size_t) 8 * cs.rows);
            for (int64_t m = 0; m < 8; ++m)
                strata::kernels::nemo_gemv(cs.t, dw, cs.s2, dx + m * cs.cols, cs.cols,
                                                 dy + m * cs.rows, cs.rows, cs.rows, cs.cols, 1, false, false, st);
            CK(cudaMemcpy(v1.data(), dy, v1.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < v1.size(); ++i) worst_v = std::fmax(worst_v, std::fabs(v1[i] - ref[i]) / rms);
            for (int64_t M = 2; M <= 8; ++M) {
                CK(cudaMemset(dy, 0, (size_t) Mmax * cs.rows * sizeof(float)));
                strata::kernels::nemo_gemv(cs.t, dw, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows,
                                                 cs.cols, M, false, false, st);
                std::vector<float> got((size_t) M * cs.rows);
                CK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                same_v &= std::memcmp(got.data(), v1.data(), got.size() * sizeof(float)) == 0;
            }
            CK(cudaMemcpy(dy, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
            strata::kernels::nemo_gemv(cs.t, dw, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows, cs.cols, 1,
                                             true, true, st);
            CK(cudaMemcpy(ra.data(), dy, ra.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < cs.rows; ++r) {
                const double v = v1[(size_t) r] > 0 ? (double) v1[(size_t) r] * v1[(size_t) r] : 0.0;
                worst_ra = std::fmax(worst_ra, std::fabs(ra[(size_t) r] - (1.0 + v)) / (1.0 + v));
            }
            std::printf("  gemv  %4lld x %4lld: worst |err|/rms %.2e vs float64; M = 1..8 bit-identical: %s\n",
                        (long long) cs.rows, (long long) cs.cols, worst_v, same_v ? "yes" : "NO");
        }
        // the verify window's tensor-core GEMV: BF16 activations (worst error ~2^-8 of the RMS), M = 1..8
        // bit-identical to its own M = 1
        double worst_t = 0;
        bool same_t = true;
        if (strata::kernels::nemo_tc_gemv_ok(cs.t, cs.cols, 1)) {
            std::vector<float> t1((size_t) 8 * cs.rows);
            for (int64_t m = 0; m < 8; ++m)
                strata::kernels::nemo_tc_gemv(cs.t, dw, cs.s2, dx + m * cs.cols, cs.cols, dy + m * cs.rows, cs.rows,
                                              cs.rows, cs.cols, 1, false, false, st);
            CK(cudaMemcpy(t1.data(), dy, t1.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < t1.size(); ++i) worst_t = std::fmax(worst_t, std::fabs(t1[i] - ref[i]) / rms);
            for (int64_t M = 2; M <= 8; ++M) {
                CK(cudaMemset(dy, 0, (size_t) Mmax * cs.rows * sizeof(float)));
                strata::kernels::nemo_tc_gemv(cs.t, dw, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows, cs.cols, M, false,
                                              false, st);
                std::vector<float> got((size_t) M * cs.rows);
                CK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                same_t &= std::memcmp(got.data(), t1.data(), got.size() * sizeof(float)) == 0;
            }
            std::printf("  tc    %4lld x %4lld: worst |err|/rms %.2e vs float64 (BF16 activations); M = 1..8 "
                        "bit-identical: %s\n", (long long) cs.rows, (long long) cs.cols, worst_t, same_t ? "yes" : "NO");
        }
        // the tiled layout (repacked on the host): tensor-core GEMV M = 1..8, and its BF16 dequantization
        double worst_l = 0, worst_d = 0;
        bool same_l = true;
        if (strata::kernels::nemo_tc_gemv_ok(cs.t, cs.cols, 1)) {
            std::vector<uint8_t> tl((size_t) strata::kernels::nemo_tiled_bytes(cs.t, cs.rows, cs.cols));
            strata::kernels::nemo_tiled_repack(cs.t, w.data(), cs.rows, cs.cols, tl.data());
            uint8_t* dt;
            CK(cudaMalloc(&dt, tl.size()));
            CK(cudaMemcpy(dt, tl.data(), tl.size(), cudaMemcpyHostToDevice));
            std::vector<float> l1((size_t) 8 * cs.rows);
            for (int64_t m = 0; m < 8; ++m)
                strata::kernels::nemo_tiled_gemv(cs.t, dt, cs.s2, dx + m * cs.cols, cs.cols, dy + m * cs.rows, cs.rows,
                                                 cs.rows, cs.cols, 1, false, false, st);
            CK(cudaMemcpy(l1.data(), dy, l1.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < l1.size(); ++i) worst_l = std::fmax(worst_l, std::fabs(l1[i] - ref[i]) / rms);
            for (int64_t M = 2; M <= 8; ++M) {
                CK(cudaMemset(dy, 0, (size_t) Mmax * cs.rows * sizeof(float)));
                strata::kernels::nemo_tiled_gemv(cs.t, dt, cs.s2, dx, cs.cols, dy, cs.rows, cs.rows, cs.cols, M, false,
                                                 false, st);
                std::vector<float> got((size_t) M * cs.rows);
                CK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                same_l &= std::memcmp(got.data(), l1.data(), got.size() * sizeof(float)) == 0;
            }
            uint16_t* db;
            CK(cudaMalloc(&db, (size_t) cs.rows * cs.cols * 2));
            strata::kernels::nemo_tiled_dequant_bf16(cs.t, dt, db, cs.rows, cs.cols, st);
            std::vector<uint16_t> hb((size_t) (cs.rows * cs.cols));
            CK(cudaMemcpy(hb.data(), db, hb.size() * 2, cudaMemcpyDeviceToHost));
            double wmax = 0;
            for (int64_t r = 0; r < cs.rows; ++r)
                for (int64_t c = 0; c < cs.cols; ++c) wmax = std::fmax(wmax, std::fabs(weight(cs.t, w, rb, r, c)));
            for (int64_t r = 0; r < cs.rows; ++r)
                for (int64_t c = 0; c < cs.cols; ++c) {
                    const double ref_w = weight(cs.t, w, rb, r, c);
                    uint32_t u = (uint32_t) hb[(size_t) (r * cs.cols + c)] << 16;
                    float f;
                    std::memcpy(&f, &u, 4);
                    worst_d = std::fmax(worst_d, std::fabs(f - ref_w) / (std::fabs(ref_w) + 1e-3 * wmax));
                }
            cudaFree(dt);
            cudaFree(db);
            std::printf("  tiled %4lld x %4lld: worst |err|/rms %.2e; M = 1..8 bit-identical: %s; BF16 dequant worst rel "
                        "%.1e\n", (long long) cs.rows, (long long) cs.cols, worst_l, same_l ? "yes" : "NO", worst_d);
        }
        // the row-major dequantization of the prompt path (nemo_dequant_bf16): NVFP4 is exact in BF16
        double worst_r = 0;
        if (cs.t == NemoWType::NVFP4) {
            uint16_t* db;
            CK(cudaMalloc(&db, (size_t) cs.rows * cs.cols * 2));
            strata::kernels::nemo_dequant_bf16(cs.t, dw, db, cs.rows, cs.cols, st);
            std::vector<uint16_t> hb((size_t) (cs.rows * cs.cols));
            CK(cudaMemcpy(hb.data(), db, hb.size() * 2, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < cs.rows; ++r)
                for (int64_t c = 0; c < cs.cols; ++c) {
                    uint32_t u = (uint32_t) hb[(size_t) (r * cs.cols + c)] << 16;
                    float f;
                    std::memcpy(&f, &u, 4);
                    worst_r = std::fmax(worst_r, std::fabs(f - weight(cs.t, w, rb, r, c)));
                }
            cudaFree(db);
            std::printf("  rowdq %4lld x %4lld: worst |err| %.1e (exact expected)\n", (long long) cs.rows,
                        (long long) cs.cols, worst_r);
        }
        const bool pass = worst_r == 0 && worst < 1e-5 && same && worst_ra < 1e-6 && worst_v < 1e-5 && same_v && worst_t < 2e-2 &&
                          same_t && worst_l < 2e-2 && same_l && worst_d < 4e-3;
        std::printf("  %-5s %4lld x %4lld: worst |err|/rms %.2e vs float64; M = 1..100 bit-identical to M = 1: %s; "
                    "relu2+accumulate %.1e  %s\n", cs.name, (long long) cs.rows, (long long) cs.cols, worst,
                    same ? "yes" : "NO", worst_ra, pass ? "ok" : "FAIL");
        ok &= pass;
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
