// src/kernels/attn_parity.cpp - strata-nemotron stage 3 test: nemo_attn (causal NoPE GQA over a BF16 K/V cache)
// against the float64 reference `tools/nemotron/attn_naive.py --dump DIR`, on a real attention block's weights and
// the real input that block sees.
//
// The projections run through bf16_gemv_fp32_mmvf (FP32 activations), K/V are appended to the cache as BF16, and the
// same T tokens are read three ways, each with its own cache:
//   decode   T = 1 at a time            (split-K over the cache, as the decode window)
//   prompt   two chunks, 37 + the rest  (the causal mask inside a chunk and the cache before it; tensor cores)
// The cache is int8 K + FP16 V (NemoKV), so against the BF16-K/V reference the tolerance is the int8 rounding's.
//   verify   windows of 4               (the verify window's shape)
// Each path's attention output is checked against the reference with BF16 K/V (`o_bf16kv`, tight: only FP32 summation
// orders differ), the block output against the exact reference (`out`, which includes what the BF16 cache costs).
//
// The cache is paged (NemoKV) in pages of 8 or 64 cells, so every path crosses page boundaries.
//
// Usage: attn_parity <dump dir>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/nemo_attn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
std::vector<T> load(const std::string& dir, const char* name, size_t n) {
    std::vector<T> v(n);
    const std::string p = dir + "/" + name;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f || std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "cannot read %zu values from %s\n", n, p.c_str());
        std::exit(2);
    }
    std::fclose(f);
    return v;
}

long long meta_int(const std::string& json, const char* key) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t at = json.find(k);
    if (at == std::string::npos) {
        std::fprintf(stderr, "meta.json has no %s\n", key);
        std::exit(2);
    }
    return std::atoll(json.c_str() + at + k.size());
}

template <class T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <class T = float>
T* dev(size_t n) {
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}

std::vector<float> down(const float* d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

double rel(const float* got, const float* want, size_t n) {
    double e = 0, w = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) got[i] - (double) want[i];
        e += d * d;
        w += (double) want[i] * want[i];
    }
    return w > 0 ? std::sqrt(e / w) : std::sqrt(e);
}

/// y (T, n_out) = W x for T tokens, one token at a time (bf16_gemv_fp32_mmvf_multi takes at most 8)
void gemv_rows(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int T, cudaStream_t st) {
    for (int t = 0; t < T; ++t) strata::kernels::bf16_gemv_fp32_mmvf(x + (size_t) t * n_in, w, y + (size_t) t * n_out,
                                                                     n_in, n_out, st);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dump dir from tools/nemotron/attn_naive.py --dump>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    std::string json;
    {
        FILE* f = std::fopen((dir + "/meta.json").c_str(), "rb");
        if (!f) { std::fprintf(stderr, "no %s/meta.json\n", dir.c_str()); return 2; }
        char buf[4096];
        json.assign(buf, std::fread(buf, 1, sizeof(buf), f));
        std::fclose(f);
    }
    strata::kernels::NemoAttnShapes s;
    s.n_head = meta_int(json, "n_head"); s.n_head_kv = meta_int(json, "n_head_kv"); s.head_dim = meta_int(json, "head_dim");
    const int T = (int) meta_int(json, "T");
    const int64_t E = meta_int(json, "n_embd"), QD = s.n_head * s.head_dim, KD = s.n_head_kv * s.head_dim;
    std::printf("attn_parity: layer %lld, %d tokens, %lld q heads / %lld kv heads x %lld\n", meta_int(json, "layer"), T,
                (long long) s.n_head, (long long) s.n_head_kv, (long long) s.head_dim);

    auto wq = load<uint16_t>(dir, "w_q_proj.bf16", (size_t) QD * E);
    auto wk = load<uint16_t>(dir, "w_k_proj.bf16", (size_t) KD * E);
    auto wv = load<uint16_t>(dir, "w_v_proj.bf16", (size_t) KD * E);
    auto wo = load<uint16_t>(dir, "w_o_proj.bf16", (size_t) E * QD);
    auto x = load<float>(dir, "x.f32", (size_t) T * E);
    auto q_ref = load<float>(dir, "q.f32", (size_t) T * QD);
    auto k_ref = load<float>(dir, "k.f32", (size_t) T * KD);
    auto v_ref = load<float>(dir, "v.f32", (size_t) T * KD);
    auto o16_ref = load<float>(dir, "o_bf16kv.f32", (size_t) T * QD);
    auto out_ref = load<float>(dir, "out.f32", (size_t) T * E);

    uint16_t *dwq = up(wq), *dwk = up(wk), *dwv = up(wv), *dwo = up(wo);
    float* dx = up(x);
    float *dq = dev((size_t) T * QD), *dk = dev((size_t) T * KD), *dv = dev((size_t) T * KD);
    float *dout = dev((size_t) T * E);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    // the projections for all T tokens
    gemv_rows(dx, dwq, dq, E, QD, T, st);
    gemv_rows(dx, dwk, dk, E, KD, T, st);
    gemv_rows(dx, dwv, dv, E, KD, T, st);
    CK(cudaStreamSynchronize(st));
    const double e_q = rel(down(dq, (size_t) T * QD).data(), q_ref.data(), (size_t) T * QD);

    void* scratch = nullptr;
    CK(cudaMalloc(&scratch, strata::kernels::nemo_attn_scratch_bytes(s, T, T)));
    // the kernel's own check reads the reference's q/k/v: with our GEMV's k a value that sits on a BF16 rounding
    // boundary can round the other way than the float64 one, a 2^-8 step that is not the kernel's error
    float *rq = up(q_ref), *rk = up(k_ref), *rv = up(v_ref);

    struct Path { const char* name; std::vector<int> chunks; };
    std::vector<Path> paths = {{"decode", {}}, {"prompt", {}}, {"verify", {}}};
    for (int t = 0; t < T; ++t) paths[0].chunks.push_back(1);
    paths[1].chunks = {T > 37 ? 37 : T / 2, T - (T > 37 ? 37 : T / 2)};
    for (int t = 0; t < T; t += 4) paths[2].chunks.push_back(T - t < 4 ? T - t : 4);

    bool ok = e_q < 1e-6;
    std::printf("  q projection                  rel %.3e\n", e_q);
    for (auto& p : paths) {
        // the cache in pages of 2^shift cells (each its own allocation), so the paths cross page boundaries
        strata::kernels::NemoKV kv;
        kv.shift = T > 1024 ? 6 : 3;
        const int npages = (int) ((T + (1 << kv.shift) - 1) >> kv.shift);
        if (npages > strata::kernels::NEMO_KV_MAX_PAGES) { std::fprintf(stderr, "too many pages\n"); return 2; }
        for (int pg = 0; pg < npages; ++pg) {
            kv.k[pg] = dev<int8_t>((size_t) KD << kv.shift);
            kv.ks[pg] = dev<float>((size_t) s.n_head_kv << kv.shift);
            kv.v[pg] = dev<uint16_t>((size_t) KD << kv.shift);
        }
        float* o = dev((size_t) T * QD);
        // the kernel on the reference's projections
        int64_t p0 = 0;
        for (int n : p.chunks) {
            strata::kernels::nemo_kv_append(kv, rk + p0 * KD, rv + p0 * KD, p0, n, s, st);
            strata::kernels::nemo_attn(rq + p0 * QD, kv, o + p0 * QD, p0, n, s, scratch, st);
            p0 += n;
        }
        CK(cudaStreamSynchronize(st));
        const std::vector<float> oh = down(o, (size_t) T * QD);
        // the block on our own projections, against the exact reference
        p0 = 0;
        for (int n : p.chunks) {
            strata::kernels::nemo_kv_append(kv, dk + p0 * KD, dv + p0 * KD, p0, n, s, st);
            strata::kernels::nemo_attn(dq + p0 * QD, kv, o + p0 * QD, p0, n, s, scratch, st);
            p0 += n;
        }
        gemv_rows(o, dwo, dout, QD, E, T, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<float> outh = down(dout, (size_t) T * E);
        double worst = 0;
        int at = 0;
        for (int t = 0; t < T; ++t) {
            const double r = rel(oh.data() + (size_t) t * QD, o16_ref.data() + (size_t) t * QD, QD);
            if (r > worst) { worst = r; at = t; }
        }
        const double e_out = rel(outh.data(), out_ref.data(), (size_t) T * E);
        std::printf("  %-7s (%2zu chunks)  attention vs BF16-K/V ref: worst rel %.3e (token %d); block output vs exact "
                    "ref: rel %.3e\n", p.name, p.chunks.size(), worst, at, e_out);
        // the cache holds K as int8 (a scale per cell and head) and V as FP16, and the prompt's chunks (T >= 16)
        // quantize q to int8 too (tensor cores): against the BF16-K/V reference ~2e-2 worst per token (3e-3 overall)
        const bool tc = std::strcmp(p.name, "prompt") == 0;
        ok &= worst < (tc ? 4e-2 : 3e-2) && e_out < 1e-2;
        for (int pg = 0; pg < npages; ++pg) { cudaFree(kv.k[pg]); cudaFree(kv.ks[pg]); cudaFree(kv.v[pg]); }
        cudaFree(o);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
