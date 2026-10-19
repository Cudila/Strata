// src/kernels/nemo_c4_bench.cpp - strata-nemotron: the experts' down_proj in decode (1024 x 2688 NVFP4, 22 experts a
// token) with real relu^2 inputs, four ways:
//   ggml   - today's NVFP4 GEMV over block_nvfp4 rows (reads every byte);
//   c4     - the C4 column layout, reading the nonzero inputs' codes and their blocks' scales only;
//   cublas - a dense BF16 GEMV of the dequantized matrix (cublasGemmBatchedEx), for scale;
//   cusparse - the input as a 1 x K CSR row times the dequantized BF16 matrix (cusparseSpMM, one call an expert).
// Correctness: ggml and c4 against a float64 host sum.  The weights are random NVFP4 (speed does not depend on them).
//
//   nemo_c4_bench <hidden rows: float32, 2688 a row (STRATA_NEMO_ZERO_DUMP)> [tokens]
#include "strata/kernels/nemo_c4.hpp"
#include "strata/kernels/nemo_gemv.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace sk = strata::kernels;

namespace {

constexpr int R = 1024, K = 2688, NE = 64;
int KT = 22;   // matrices a call: 22 a token, 22 x window rows in a verify window

float e2m1(int q) {
    static const float t[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    return t[q & 15];
}
float ue4m3(int b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e ? std::ldexp(1.0f + m / 8.0f, e - 7) : m / 512.0f;
}
/// W[r][k] of a block_nvfp4 row matrix
float deq(const uint8_t* g, int r, int k) {
    const uint8_t* b = g + (size_t) r * (K / 64) * 36 + (size_t) (k / 64) * 36;
    const int s = (k % 64) / 16, v = k % 16;
    const int q = (b[4 + 8 * s + (v & 7)] >> (v < 8 ? 0 : 4)) & 15;
    return e2m1(q) * ue4m3(b[s]);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: nemo_c4_bench <hidden rows> [tokens]\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 1; }
    std::vector<float> hid;
    {
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f) / (long) (K * sizeof(float));
        std::fseek(f, 0, SEEK_SET);
        hid.resize((size_t) n * K);
        if (std::fread(hid.data(), sizeof(float), hid.size(), f) != hid.size()) return 1;
        std::fclose(f);
    }
    const int nrows = (int) (hid.size() / K);
    if (argc > 3) KT = std::atoi(argv[3]);
    const int tokens = std::min(argc > 2 ? std::atoi(argv[2]) : 300, nrows / KT);
    size_t nz = 0;
    for (float v : hid) nz += v != 0.0f;
    std::printf("%d hidden rows, %.2f%% zeros; %d tokens of %d experts\n", nrows, 100.0 - 100.0 * nz / hid.size(),
                tokens, KT);

    // random NVFP4 experts (scales 2^-4 .. 2^1), their C4 copies and BF16 dequantizations
    const size_t gb = (size_t) R * (K / 64) * 36;
    std::vector<uint8_t> g((size_t) NE * gb);
    std::mt19937 rng(7);
    for (int e = 0; e < NE; ++e)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < K / 64; ++b) {
                uint8_t* p = g.data() + (size_t) e * gb + (size_t) r * (K / 64) * 36 + (size_t) b * 36;
                for (int s = 0; s < 4; ++s) p[s] = (uint8_t) (0x18 + rng() % 0x28);
                for (int i = 4; i < 36; ++i) p[i] = (uint8_t) rng();
            }
    std::vector<float> s2(NE);
    for (int e = 0; e < NE; ++e) s2[e] = 0.5f + (rng() % 1000) / 1000.0f;
    uint8_t *dg, *dc;
    CK(cudaMalloc(&dg, g.size()));
    CK(cudaMalloc(&dc, (size_t) NE * sk::nemo_c4_bytes(R, K)));
    CK(cudaMemcpy(dg, g.data(), g.size(), cudaMemcpyHostToDevice));
    for (int e = 0; e < NE; ++e) sk::nemo_c4_from_ggml(dg + (size_t) e * gb, R, K, dc + (size_t) e * sk::nemo_c4_bytes(R, K), nullptr);
    std::vector<__nv_bfloat16> wb((size_t) NE * R * K);
    for (int e = 0; e < NE; ++e)
        for (int r = 0; r < R; ++r)
            for (int k = 0; k < K; ++k)
                wb[((size_t) e * R + r) * K + k] = __float2bfloat16(deq(g.data() + (size_t) e * gb, r, k) * s2[e]);
    __nv_bfloat16* dwb;
    CK(cudaMalloc(&dwb, wb.size() * 2));
    CK(cudaMemcpy(dwb, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));

    // a token: experts (rng) and inputs (consecutive hidden rows)
    std::vector<int> ex((size_t) tokens * KT);
    for (int& e : ex) e = (int) (rng() % NE);
    float *dx, *dy, *ds2, *dvals;
    uint16_t* dlist;
    int* dcount;
    float* dpart;
    CK(cudaMalloc(&dpart, (size_t) sk::nemo_c4_part_floats(R, KT) * 4));
    CK(cudaMalloc(&dlist, (size_t) tokens * KT * K * 2));
    CK(cudaMalloc(&dvals, (size_t) tokens * KT * K * 4));
    CK(cudaMalloc(&dcount, (size_t) tokens * KT * 4));
    CK(cudaMalloc(&dx, (size_t) tokens * KT * K * 4));
    CK(cudaMemcpy(dx, hid.data(), (size_t) tokens * KT * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t) KT * R * 4 * 2));
    CK(cudaMalloc(&ds2, (size_t) tokens * KT * 4));
    std::vector<float> hs2((size_t) tokens * KT);
    for (size_t i = 0; i < hs2.size(); ++i) hs2[i] = s2[(size_t) ex[i]];
    CK(cudaMemcpy(ds2, hs2.data(), hs2.size() * 4, cudaMemcpyHostToDevice));
    std::vector<const uint8_t*> pg(ex.size()), pc(ex.size());
    for (size_t i = 0; i < ex.size(); ++i) {
        pg[i] = dg + (size_t) ex[i] * gb;
        pc[i] = dc + (size_t) ex[i] * sk::nemo_c4_bytes(R, K);
    }
    const uint8_t **dpg, **dpc;
    CK(cudaMalloc(&dpg, pg.size() * sizeof(void*)));
    CK(cudaMalloc(&dpc, pc.size() * sizeof(void*)));
    CK(cudaMemcpy(dpg, pg.data(), pg.size() * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpc, pc.data(), pc.size() * sizeof(void*), cudaMemcpyHostToDevice));

    // correctness on token 0
    {
        sk::nemo_gemv_many(sk::NemoWType::NVFP4, dpg, ds2, dx, K, dy, R, R, K, KT, false, nullptr);
        sk::nemo_c4_compact(dx, K, K, KT, dlist, dvals, dcount, nullptr);
        sk::nemo_c4_gemv_many(dpc, ds2, dlist, dvals, dcount, dpart, dy + (size_t) KT * R, R, R, K, KT, nullptr, nullptr);
        std::vector<float> y((size_t) KT * R * 2);
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double eg = 0, ec = 0, nn = 0;
        for (int j = 0; j < KT; ++j)
            for (int r = 0; r < R; ++r) {
                double ref = 0;
                for (int k = 0; k < K; ++k) {
                    const float xv = hid[(size_t) j * K + k];
                    if (xv != 0.0f) ref += (double) deq(g.data() + (size_t) ex[(size_t) j] * gb, r, k) * xv;
                }
                ref *= s2[(size_t) ex[(size_t) j]];
                eg += (y[(size_t) j * R + r] - ref) * (y[(size_t) j * R + r] - ref);
                ec += (y[(size_t) (KT + j) * R + r] - ref) * (y[(size_t) (KT + j) * R + r] - ref);
                nn += ref * ref;
            }
        std::printf("against float64: ggml rel err %.2e, c4 rel err %.2e %s\n", std::sqrt(eg / nn), std::sqrt(ec / nn),
                    std::sqrt(ec / nn) < 1e-5 ? "PASS" : "FAIL");
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    auto timed = [&](const char* name, auto&& fn) {
        for (int w = 0; w < 3; ++w) fn(w);
        cudaEventRecord(t0);
        for (int t = 0; t < tokens; ++t) fn(t);
        cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1));
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        std::printf("  %-9s %7.2f us a call (%d matrices)\n", name, 1000.0 * ms / tokens, KT);
    };
    std::printf("down_proj, %d tokens:\n", tokens);
    timed("ggml", [&](int t) {
        sk::nemo_gemv_many(sk::NemoWType::NVFP4, dpg + (size_t) t * KT, ds2 + (size_t) t * KT, dx + (size_t) t * KT * K, K,
                           dy, R, R, K, KT, false, nullptr);
    });
    timed("c4", [&](int t) {   // the lists and the product
        sk::nemo_c4_compact(dx + (size_t) t * KT * K, K, K, KT, dlist, dvals, dcount, nullptr);
        sk::nemo_c4_gemv_many(dpc + (size_t) t * KT, ds2 + (size_t) t * KT, dlist, dvals, dcount, dpart, dy, R, R, K, KT,
                              nullptr, nullptr);
    });
    timed("c4 lists", [&](int t) { sk::nemo_c4_compact(dx + (size_t) t * KT * K, K, K, KT, dlist, dvals, dcount, nullptr); });
    sk::nemo_c4_compact(dx, K, K, KT, dlist, dvals, dcount, nullptr);
    timed("c4 gemv", [&](int t) {
        sk::nemo_c4_gemv_many(dpc + (size_t) t * KT, ds2 + (size_t) t * KT, dlist, dvals, dcount, dpart, dy, R, R, K, KT,
                              nullptr, nullptr);
    });

    // cuBLAS: dense BF16, the inputs in BF16
    {
        std::vector<__nv_bfloat16> xb((size_t) tokens * KT * K);
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = __float2bfloat16(hid[i]);
        __nv_bfloat16* dxb;
        CK(cudaMalloc(&dxb, xb.size() * 2));
        CK(cudaMemcpy(dxb, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        std::vector<const void*> A(ex.size()), B(ex.size());
        std::vector<void*> C(ex.size());
        for (size_t i = 0; i < ex.size(); ++i) {
            A[i] = dwb + (size_t) ex[i] * R * K;
            B[i] = dxb + i * K;
            C[i] = dy + (i % KT) * R;
        }
        const void **dA, **dB;
        void** dC;
        CK(cudaMalloc(&dA, A.size() * sizeof(void*)));
        CK(cudaMalloc(&dB, B.size() * sizeof(void*)));
        CK(cudaMalloc(&dC, C.size() * sizeof(void*)));
        CK(cudaMemcpy(dA, A.data(), A.size() * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B.data(), B.size() * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dC, C.data(), C.size() * sizeof(void*), cudaMemcpyHostToDevice));
        cublasHandle_t h;
        cublasCreate(&h);
        const float one = 1.0f, zero = 0.0f;
        timed("cublas", [&](int t) {
            // column-major: W (R x K row-major) is a K x R matrix, op T gives R x K
            cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, R, 1, K, &one, dA + (size_t) t * KT, CUDA_R_16BF, K,
                                dB + (size_t) t * KT, CUDA_R_16BF, K, &zero, dC + (size_t) t * KT, CUDA_R_32F, R, KT,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        });
        cublasDestroy(h);
    }

    // cuSPARSE: the input as a 1 x K CSR row (BF16 values), the matrix dense BF16 (K x R, column-major = W's rows)
    {
        cusparseHandle_t h;
        if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS) { std::printf("  cusparse: no handle\n"); return 0; }
        std::vector<int> off((size_t) tokens * KT * 2), col;
        std::vector<__nv_bfloat16> val;
        for (int i = 0; i < tokens * KT; ++i) {
            off[(size_t) i * 2] = (int) col.size();
            for (int k = 0; k < K; ++k)
                if (hid[(size_t) i * K + k] != 0.0f) { col.push_back(k); val.push_back(__float2bfloat16(hid[(size_t) i * K + k])); }
            off[(size_t) i * 2 + 1] = (int) col.size();
        }
        int *doff, *dcol;
        __nv_bfloat16* dval;
        CK(cudaMalloc(&doff, off.size() * 4));
        CK(cudaMalloc(&dcol, col.size() * 4));
        CK(cudaMalloc(&dval, val.size() * 2));
        CK(cudaMemcpy(dcol, col.data(), col.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dval, val.data(), val.size() * 2, cudaMemcpyHostToDevice));
        // per row a [0, nnz] offsets pair, the column / value arrays shifted to the row's start
        std::vector<int> off2;
        for (int i = 0; i < tokens * KT; ++i) { off2.push_back(0); off2.push_back(off[(size_t) i * 2 + 1] - off[(size_t) i * 2]); }
        CK(cudaMemcpy(doff, off2.data(), off2.size() * 4, cudaMemcpyHostToDevice));
        const float one = 1.0f, zero = 0.0f;
        void* buf = nullptr;
        size_t bufsz = 0;
        cusparseStatus_t st = CUSPARSE_STATUS_SUCCESS;
        std::vector<cusparseSpMatDescr_t> As((size_t) tokens * KT);
        std::vector<cusparseDnMatDescr_t> Bs((size_t) tokens * KT), Cs((size_t) tokens * KT);
        for (int i = 0; i < tokens * KT && st == CUSPARSE_STATUS_SUCCESS; ++i) {
            const int nnzi = off2[(size_t) i * 2 + 1];
            st = cusparseCreateCsr(&As[(size_t) i], 1, K, nnzi, doff + (size_t) i * 2, dcol + off[(size_t) i * 2],
                                   dval + off[(size_t) i * 2], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF);
            if (st == CUSPARSE_STATUS_SUCCESS)
                st = cusparseCreateDnMat(&Bs[(size_t) i], K, R, K, dwb + (size_t) ex[(size_t) i] * R * K, CUDA_R_16BF,
                                         CUSPARSE_ORDER_COL);
            if (st == CUSPARSE_STATUS_SUCCESS)
                st = cusparseCreateDnMat(&Cs[(size_t) i], 1, R, 1, dy + (size_t) (i % KT) * R, CUDA_R_32F,
                                         CUSPARSE_ORDER_COL);
        }
        if (st == CUSPARSE_STATUS_SUCCESS)
            st = cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                         As[0], Bs[0], &zero, Cs[0], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufsz);
        if (st != CUSPARSE_STATUS_SUCCESS) {
            std::printf("  cusparse: SpMM with BF16 A/B and FP32 C is not supported here (%d)\n", (int) st);
        } else {
            CK(cudaMalloc(&buf, bufsz + 1));
            timed("cusparse", [&](int t) {
                for (int j = 0; j < KT; ++j) {
                    const size_t i = (size_t) t * KT + j;
                    cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, As[i], Bs[i],
                                 &zero, Cs[i], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
                }
            });
        }
        cusparseDestroy(h);
    }
    std::printf("bytes a token: ggml %.2f MB; c4 ~%.2f MB at this sparsity; BF16 dense %.2f MB\n", KT * gb / 1e6,
                KT * (0.08 * K * R / 2 + 0.69 * (K / 16) * R) / 1e6, KT * (double) R * K * 2 / 1e6);
    return 0;
}
