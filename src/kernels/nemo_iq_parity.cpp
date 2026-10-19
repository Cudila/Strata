// src/kernels/nemo_iq_parity.cpp - strata-nemotron: the expert-file formats (Q2_0, IQ4_NL, IQ4_XS, IQ3_XXS) in
// nemo_gemv / nemo_gemv_many / the prompt's BF16 dequantization against llama.cpp's dequantize_row_* (ggml-quants.c,
// ported below on llama.cpp's tables) in float64, on random blocks.
//
// Checks per format: one matrix, M = 1..8 (error vs float64, M-invariance), 22 matrices at once through
// nemo_gemv_many with a permuted idx (each output row equals its matrix alone), and the dequantized BF16 matrix
// against the float64 values (BF16's rounding, 2^-9 relative).
#include "strata/kernels/nemo_gemm.hpp"
#include "strata/kernels/nemo_gemv.hpp"
#include "strata/kernels/nemo_tc.hpp"

#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::kernels::NemoWType;

#define CK(x)                                                                                 \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                                         \
        }                                                                                     \
    } while (0)

namespace {

double f16(const uint8_t* p) {
    const uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
    return s ? -v : v;
}

uint16_t to_f16(float f) {   // positive normal values only (the test's scales)
    int e;
    const float m = std::frexp(f, &e);   // f = m * 2^e, m in [0.5, 1)
    const int E = e - 1 + 15;
    const int M = (int) std::lround((m * 2.0f - 1.0f) * 1024.0f);
    return (uint16_t) ((E << 10) | (M & 1023));
}

/// llama.cpp's dequantize_row_* for one block, in double.
void deq_block(NemoWType t, const uint8_t* b, double* y) {
    switch (t) {
    case NemoWType::Q2_0: {
        const double d = f16(b);
        for (int j = 0; j < 64; ++j) y[j] = ((int) ((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        break;
    }
    case NemoWType::IQ4_NL: {
        const double d = f16(b);
        for (int j = 0; j < 16; ++j) {
            y[j] = d * kvalues_iq4nl[b[2 + j] & 0xf];
            y[j + 16] = d * kvalues_iq4nl[b[2 + j] >> 4];
        }
        break;
    }
    case NemoWType::IQ4_XS: {
        const double d = f16(b);
        const uint16_t sh = (uint16_t) (b[2] | (b[3] << 8));
        const uint8_t* sl = b + 4;
        const uint8_t* qs = b + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[ib * 32 + j] = dl * kvalues_iq4nl[qs[ib * 16 + j] & 0xf];
                y[ib * 32 + j + 16] = dl * kvalues_iq4nl[qs[ib * 16 + j] >> 4];
            }
        }
        break;
    }
    case NemoWType::IQ3_XXS: {
        const double d = f16(b);
        const uint8_t* qs = b + 2;
        const uint8_t* ss = b + 2 + 64;
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux;
            std::memcpy(&aux, ss + 4 * ib, 4);
            const double db = d * (0.5 + (aux >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux >> 7 * l) & 127];
                const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + qs[ib * 8 + 2 * l]);
                const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + qs[ib * 8 + 2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[ib * 32 + l * 8 + j] = db * g1[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                    y[ib * 32 + l * 8 + j + 4] = db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1 : 1);
                }
            }
        }
        break;
    }
    default: break;
    }
}

int block_bytes(NemoWType t) { return (int) strata::kernels::nemo_row_bytes(t, strata::kernels::nemo_block_values(t)); }

/// A random matrix: random code bytes, the block scales set to small positive f16 values.
std::vector<uint8_t> random_matrix(NemoWType t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const int64_t bv = strata::kernels::nemo_block_values(t), bb = block_bytes(t);
    std::vector<uint8_t> w((size_t) (rows * cols / bv * bb));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.01f, 0.05f);
    for (auto& v : w) v = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += (size_t) bb) {
        const uint16_t h = to_f16(sc(rng));
        w[o] = (uint8_t) (h & 0xff);
        w[o + 1] = (uint8_t) (h >> 8);
    }
    return w;
}

void dequant_matrix(NemoWType t, const std::vector<uint8_t>& w, int64_t rows, int64_t cols, std::vector<double>& out) {
    const int64_t bv = strata::kernels::nemo_block_values(t), bb = block_bytes(t);
    out.assign((size_t) (rows * cols), 0.0);
    for (int64_t b = 0; b < rows * cols / bv; ++b) deq_block(t, w.data() + b * bb, out.data() + b * bv);
}

}  // namespace

int main() {
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct Case { NemoWType t; const char* name; int64_t rows, cols; };
    const Case cases[] = {{NemoWType::IQ3_XXS, "IQ3_XXS", 2688, 1024}, {NemoWType::IQ4_XS, "IQ4_XS", 2688, 1024},
                          {NemoWType::IQ4_NL, "IQ4_NL", 1024, 2688}, {NemoWType::Q2_0, "Q2_0", 1024, 2688},
                          {NemoWType::IQ3_XXS, "IQ3_XXS", 301, 4096}, {NemoWType::Q2_0, "Q2_0", 77, 8192}};
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    bool ok = true;
    for (const Case& cs : cases) {
        const int64_t R = cs.rows, Cc = cs.cols, NM = 22;
        const size_t mb = (size_t) strata::kernels::nemo_row_bytes(cs.t, Cc) * R;
        // NM matrices; matrix 0 is the single-matrix case
        std::vector<std::vector<uint8_t>> ws;
        for (int64_t i = 0; i < NM; ++i) ws.push_back(random_matrix(cs.t, R, Cc, rng));
        std::vector<float> x((size_t) (NM * Cc));
        for (auto& v : x) v = nd(rng);
        std::vector<double> wd;
        dequant_matrix(cs.t, ws[0], R, Cc, wd);
        // float64 reference of matrix 0 against the first 8 rows of x
        std::vector<double> ref((size_t) (8 * R));
        double ss = 0;
        for (int64_t m = 0; m < 8; ++m)
            for (int64_t r = 0; r < R; ++r) {
                double a = 0;
                for (int64_t c = 0; c < Cc; ++c) a += wd[(size_t) (r * Cc + c)] * x[(size_t) (m * Cc + c)];
                ref[(size_t) (m * R + r)] = a;
                ss += a * a;
            }
        const double rms = std::sqrt(ss / (double) ref.size());
        uint8_t* dw;
        float *dx, *dy;
        CK(cudaMalloc(&dw, mb * NM));
        for (int64_t i = 0; i < NM; ++i) CK(cudaMemcpy(dw + i * mb, ws[(size_t) i].data(), mb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x.size() * sizeof(float)));
        CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dy, (size_t) NM * R * 8 * sizeof(float)));
        // one matrix, each token alone
        std::vector<float> one((size_t) (8 * R));
        for (int64_t m = 0; m < 8; ++m)
            strata::kernels::nemo_gemv(cs.t, dw, 1.0f, dx + m * Cc, Cc, dy + m * R, R, R, Cc, 1, false, false, st);
        CK(cudaMemcpy(one.data(), dy, one.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < one.size(); ++i) worst = std::fmax(worst, std::fabs(one[i] - ref[i]) / rms);
        bool same = true;
        for (int64_t M = 2; M <= 8; ++M) {
            CK(cudaMemset(dy, 0, (size_t) 8 * R * sizeof(float)));
            strata::kernels::nemo_gemv(cs.t, dw, 1.0f, dx, Cc, dy, R, R, Cc, M, false, false, st);
            std::vector<float> got((size_t) (M * R));
            CK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            same &= std::memcmp(got.data(), one.data(), got.size() * sizeof(float)) == 0;
        }
        // NM matrices through nemo_gemv_many, entry i = matrix i, input / output row idx[i] (a permutation)
        std::vector<const uint8_t*> hp((size_t) NM);
        std::vector<int32_t> hidx((size_t) NM);
        std::vector<float> hs2((size_t) NM, 1.0f);
        for (int64_t i = 0; i < NM; ++i) {
            hp[(size_t) i] = dw + i * mb;
            hidx[(size_t) i] = (int32_t) ((i * 7 + 3) % NM);
        }
        const uint8_t** dp;
        int32_t* didx;
        float* ds2;
        CK(cudaMalloc(&dp, NM * sizeof(void*)));
        CK(cudaMalloc(&didx, NM * sizeof(int32_t)));
        CK(cudaMalloc(&ds2, NM * sizeof(float)));
        CK(cudaMemcpy(dp, hp.data(), NM * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(didx, hidx.data(), NM * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds2, hs2.data(), NM * sizeof(float), cudaMemcpyHostToDevice));
        strata::kernels::nemo_gemv_many(cs.t, dp, ds2, dx, Cc, dy, R, R, Cc, NM, false, st, didx);
        std::vector<float> many((size_t) (NM * R));
        CK(cudaMemcpy(many.data(), dy, many.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool many_ok = true;
        for (int64_t i = 0; i < NM && many_ok; ++i) {
            const int64_t o = hidx[(size_t) i];
            // (the same matrix as a batch of one: the batched kernel's sum order; a lone nemo_gemv of few rows may
            // take the block-per-row kernel, whose order differs)
            strata::kernels::nemo_gemv_many(cs.t, dp + i, ds2 + i, dx + o * Cc, 0, dy, R, R, Cc, 1, false, st);
            std::vector<float> alone((size_t) R);
            CK(cudaMemcpy(alone.data(), dy, R * sizeof(float), cudaMemcpyDeviceToHost));
            many_ok &= std::memcmp(alone.data(), many.data() + o * R, R * sizeof(float)) == 0;
        }
        // the prompt path's BF16 dequantization of matrix 0
        uint16_t* db;
        CK(cudaMalloc(&db, (size_t) R * Cc * 2));
        const uint8_t** dp0 = dp;
        strata::kernels::nemo_dequant_bf16_many_t(cs.t, dp0, db, 1, R, Cc, st);
        std::vector<uint16_t> hb((size_t) (R * Cc));
        CK(cudaMemcpy(hb.data(), db, hb.size() * 2, cudaMemcpyDeviceToHost));
        double worst_b = 0, wmax = 0;
        for (size_t i = 0; i < hb.size(); ++i) wmax = std::fmax(wmax, std::fabs(wd[i]));
        for (size_t i = 0; i < hb.size(); ++i) {
            uint32_t u = (uint32_t) hb[i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            worst_b = std::fmax(worst_b, std::fabs(f - wd[i]) / (std::fabs(wd[i]) + 1e-3 * wmax));
        }
        CK(cudaGetLastError());
        const bool pass = worst < 1e-5 && same && many_ok && worst_b < 4e-3;
        std::printf("  %-7s %4lld x %4lld: worst |err|/rms %.2e vs float64; M = 1..8 bit-identical: %s; 22 at once "
                    "(permuted rows) = alone: %s; BF16 dequant worst rel %.1e  %s\n", cs.name, (long long) R,
                    (long long) Cc, worst, same ? "yes" : "NO", many_ok ? "yes" : "NO", worst_b, pass ? "ok" : "FAIL");
        ok &= pass;
        cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dp); cudaFree(didx); cudaFree(ds2); cudaFree(db);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
