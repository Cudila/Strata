// src/kernels/nemo_attn_bench.cpp - strata-nemotron: nemo_attn's speed on a long context (random K/V and q): a prompt
// chunk of T queries after p0 cached cells, and a decode / verify window there.  No correctness check (attn_parity).
//
// Usage: nemo_attn_bench [p0 (131072)] [T (8192)]
#include "strata/kernels/nemo_attn.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int64_t p0 = argc > 1 ? std::atoll(argv[1]) : 131072, T = argc > 2 ? std::atoll(argv[2]) : 8192;
    strata::kernels::NemoAttnShapes s;
    const int64_t KD = s.n_head_kv * s.head_dim, QD = s.n_head * s.head_dim, cells = p0 + T;
    strata::kernels::NemoKV kv;
    kv.shift = 16;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> kh((size_t) (T * KD)), qh((size_t) (T * QD));
    for (float& x : kh) x = nd(rng);
    for (float& x : qh) x = nd(rng);
    float *dk, *dq, *dout;
    cudaMalloc(&dk, kh.size() * 4);
    cudaMalloc(&dq, qh.size() * 4);
    cudaMalloc(&dout, qh.size() * 4);
    cudaMemcpy(dk, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice);
    for (int64_t pg = 0; pg <= (cells >> kv.shift); ++pg) {
        cudaMalloc(&kv.k[pg], (size_t) KD << kv.shift);
        cudaMalloc(&kv.ks[pg], (size_t) s.n_head_kv * 4 << kv.shift);
        cudaMalloc(&kv.v[pg], (size_t) KD * 2 << kv.shift);
    }
    // fill every cell from the T random rows (cycled)
    for (int64_t c = 0; c < cells; c += T)
        strata::kernels::nemo_kv_append(kv, dk, dk, c, std::min<int64_t>(T, cells - c), s, nullptr);
    void* scratch;
    cudaMalloc(&scratch, strata::kernels::nemo_attn_scratch_bytes(s, 8, cells));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](int64_t pp, int64_t n, int reps) {
        strata::kernels::nemo_attn(dq, kv, dout, pp, n, s, scratch, nullptr);
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) strata::kernels::nemo_attn(dq, kv, dout, pp, n, s, scratch, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    const double ms = time(p0, T, 3);
    const double flop = 4.0 * s.n_head * s.head_dim * (double) T * ((double) p0 + (T + 1) / 2.0);
    std::printf("prompt  T %lld after %lld cells: %.2f ms, %.0f TFLOP/s\n", (long long) T, (long long) p0, ms,
                flop / ms / 1e9);
    for (int64_t n : {1, 6})
        std::printf("window  T %lld after %lld cells: %.3f ms\n", (long long) n, (long long) cells - 8,
                    time(cells - 8, n, 20));
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
