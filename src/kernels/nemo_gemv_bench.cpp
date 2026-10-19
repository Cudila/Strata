// src/kernels/nemo_gemv_bench.cpp - strata-nemotron: the decode GEMVs' bandwidth on the model's shapes (random NVFP4
// weights; numbers only, the parity is nemo_gemm_parity's).
#include "strata/kernels/nemo_gemm.hpp"
#include "strata/kernels/nemo_gemv.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    struct Shape { const char* name; int64_t rows, cols, n; };
    const Shape shapes[] = {{"mamba in_proj", 18560, 4096, 1}, {"mamba out_proj", 4096, 8192, 1},
                            {"shared up", 5376, 4096, 1},      {"shared down", 4096, 5376, 1},
                            {"22 experts up", 2688, 1024, 22}, {"22 experts down", 1024, 2688, 22}};
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const Shape& s : shapes) {
        const size_t mb = (size_t) s.rows * s.cols / 64 * 36;
        // distinct matrices so the cache does not serve the weights; 64 copies cycled
        const int copies = 64;
        uint8_t* w;
        cudaMalloc(&w, mb * s.n * copies);
        cudaMemset(w, 0x11, mb * s.n * copies);
        float *x, *y, *s2;
        cudaMalloc(&x, 8 * 8192 * 4 * 32);
        cudaMemset(x, 0, 8 * 8192 * 4 * 32);
        cudaMalloc(&y, (size_t) 8 * 32 * s.rows * 4);
        cudaMalloc(&s2, 64 * 4);
        const uint8_t** ptr;
        cudaMalloc(&ptr, sizeof(void*) * s.n * copies);
        std::vector<const uint8_t*> hp((size_t) s.n * copies);
        for (int i = 0; i < s.n * copies; ++i) hp[(size_t) i] = w + (size_t) i * mb;
        cudaMemcpy(ptr, hp.data(), hp.size() * sizeof(void*), cudaMemcpyHostToDevice);
        for (int tc = 0; tc < 2; ++tc)
        for (int M : {1, 4, 8}) {
            if (s.n > 1 && tc) break;
            if (s.n > 1 && M > 1) break;
            const int iters = 200;
            for (int it = -10; it < iters; ++it) {
                if (it == 0) cudaEventRecord(e0, st);
                const int c = (it + 10) % copies;
                if (s.n == 1 && tc)
                    strata::kernels::nemo_tiled_gemv(strata::kernels::NemoWType::NVFP4, w + (size_t) c * mb, 1.0f, x,
                                                     s.cols, y, s.rows, s.rows, s.cols, M, false, false, st);
                else if (s.n == 1)
                    strata::kernels::nemo_gemv_nvfp4(w + (size_t) c * mb, 1.0f, x, s.cols, y, s.rows, s.rows, s.cols, M,
                                                     false, false, st);
                else
                    strata::kernels::nemo_gemv_nvfp4_many(ptr + (size_t) c * s.n, s2, x, 0, y, s.rows, s.rows, s.cols,
                                                          s.n, false, st);
            }
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000.0 / iters;
            std::printf("%-16s %s M=%d: %7.1f us, %6.0f GB/s\n", s.name, tc ? "tc  " : "cuda", M, us, mb * s.n / us / 1000.0);
        }
        cudaFree(w); cudaFree(x); cudaFree(y); cudaFree(s2); cudaFree(ptr);
    }
    std::printf("err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
