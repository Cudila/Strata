// src/kernels/moe_parity.cpp - strata-nemotron stage 4 test: the LatentMoE block (nemo_moe.hpp + the BF16 GEMVs)
// against the float64 reference `tools/nemotron/moe_naive.py <ref> <NVFP4 dir> <layer> --dump DIR`: real weights in
// the GGUF's own NVFP4 bytes, the real input that block sees.
//
// Per token, as the decode path will run it: router logits (BF16 GEMV, FP32 activations) -> nemo_router -> fc1
// latent -> the 22 experts through nemo_experts (pointers into the expert table, as the VRAM cache will hand them) ->
// fc2 latent + the shared expert (nvfp4_gemv).  Checked: the selected set equals the reference's (a different set is
// reported with its margin: the reference's 22nd-23rd gap), the routing weights, the routed latent sum, the block's
// MoE output.  Tolerances: FP32 against FP64 over 1024-5376-term dots, 1e-5 relative.
//
// Usage: moe_parity <dump dir>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/nemo_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

double meta_num(const std::string& json, const char* key) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t at = json.find(k);
    if (at == std::string::npos) {
        std::fprintf(stderr, "meta.json has no %s\n", key);
        std::exit(2);
    }
    return std::atof(json.c_str() + at + k.size());
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

template <class T = float>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dump dir from tools/nemotron/moe_naive.py --dump>\n", argv[0]);
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
    const int T = (int) meta_num(json, "T");
    const int64_t E = (int64_t) meta_num(json, "n_embd"), NE = (int64_t) meta_num(json, "n_expert");
    const int64_t K = (int64_t) meta_num(json, "k"), L = (int64_t) meta_num(json, "latent");
    const int64_t FF = (int64_t) meta_num(json, "ff"), SH = (int64_t) meta_num(json, "shared_ff");
    const float scale = (float) meta_num(json, "scale");
    std::printf("moe_parity: layer %d, %d tokens, %lld experts top-%lld, latent %lld, ff %lld, shared %lld\n",
                (int) meta_num(json, "layer"), T, (long long) NE, (long long) K, (long long) L, (long long) FF,
                (long long) SH);
    const size_t up_bytes = (size_t) FF * (L / 64) * 36, dn_bytes = (size_t) L * (FF / 64) * 36;

    auto w_gate = load<uint16_t>(dir, "w_gate.bf16", (size_t) NE * E);
    auto w_fc1 = load<uint16_t>(dir, "w_fc1.bf16", (size_t) L * E);
    auto w_fc2 = load<uint16_t>(dir, "w_fc2.bf16", (size_t) E * L);
    auto bias = load<float>(dir, "bias.f32", NE);
    auto ex_up = load<uint8_t>(dir, "experts_up.nvfp4", up_bytes * NE);
    auto ex_dn = load<uint8_t>(dir, "experts_down.nvfp4", dn_bytes * NE);
    auto ex_s2 = load<float>(dir, "experts_scale2.f32", (size_t) NE * 2);
    auto sh_up = load<uint8_t>(dir, "shared_up.nvfp4", (size_t) SH * (E / 64) * 36);
    auto sh_dn = load<uint8_t>(dir, "shared_down.nvfp4", (size_t) E * (SH / 64) * 36);
    const float sh_up_s2 = load<float>(dir, "shared_up_scale2.f32", 1)[0];
    const float sh_dn_s2 = load<float>(dir, "shared_down_scale2.f32", 1)[0];
    auto x = load<float>(dir, "x.f32", (size_t) T * E);
    auto logits_ref = load<float>(dir, "logits.f32", (size_t) T * NE);
    auto w_ref = load<float>(dir, "w.f32", (size_t) T * K);
    auto ids_ref = load<int32_t>(dir, "ids.i32", (size_t) T * K);
    auto routed_ref = load<float>(dir, "routed.f32", (size_t) T * L);
    auto moe_ref = load<float>(dir, "moe.f32", (size_t) T * E);

    uint16_t *dgate = up(w_gate), *dfc1 = up(w_fc1), *dfc2 = up(w_fc2);
    float* dbias = up(bias);
    uint8_t *dup = up(ex_up), *ddn = up(ex_dn), *dshu = up(sh_up), *dshd = up(sh_dn);
    float *dx = dev(E), *dlog = dev(NE), *dw = dev(K), *dlat = dev(L), *dh = dev((size_t) K * (FF + L));
    float *drouted = dev(L), *dmoe = dev(E), *dshh = dev(SH), *dsh = dev(E);
    int32_t* dids = dev<int32_t>(K);
    const uint8_t** dptr = nullptr;
    CK(cudaMalloc(&dptr, 2 * K * sizeof(void*)));
    float* ds2 = dev(2 * K);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    int flips = 0;
    double w_logit = 0, w_w = 0, w_routed = 0, w_moe = 0;
    for (int t = 0; t < T; ++t) {
        CK(cudaMemcpyAsync(dx, x.data() + (size_t) t * E, E * sizeof(float), cudaMemcpyHostToDevice, st));
        strata::kernels::bf16_gemv_fp32_mmvf(dx, dgate, dlog, E, NE, st);
        strata::kernels::nemo_router(dlog, dbias, dids, dw, 1, NE, K, scale, true, st);
        strata::kernels::bf16_gemv_fp32_mmvf(dx, dfc1, dlat, E, L, st);
        CK(cudaStreamSynchronize(st));
        w_logit = std::max(w_logit, rel(down(dlog, NE).data(), logits_ref.data() + (size_t) t * NE, NE));
        const std::vector<int32_t> ids = down(dids, K);
        const std::vector<float> wv = down(dw, K);
        std::vector<int32_t> a(ids), b(ids_ref.begin() + (size_t) t * K, ids_ref.begin() + (size_t) (t + 1) * K);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            ++flips;
            continue;   // a different set is a different computation: counted, not compared
        }
        // weights in the reference's order
        std::vector<float> wo(K);
        for (int64_t j = 0; j < K; ++j)
            for (int64_t i = 0; i < K; ++i)
                if (ids[i] == ids_ref[(size_t) t * K + j]) wo[j] = wv[i];
        w_w = std::max(w_w, rel(wo.data(), w_ref.data() + (size_t) t * K, K));

        // the expert table as the cache hands it: one pointer and one scale_2 per selected expert
        std::vector<const uint8_t*> ptr(2 * K);
        std::vector<float> s2(2 * K);
        for (int64_t j = 0; j < K; ++j) {
            ptr[j] = dup + (size_t) ids[j] * up_bytes;
            ptr[K + j] = ddn + (size_t) ids[j] * dn_bytes;
            s2[j] = ex_s2[(size_t) ids[j] * 2];
            s2[K + j] = ex_s2[(size_t) ids[j] * 2 + 1];
        }
        CK(cudaMemcpyAsync(dptr, ptr.data(), 2 * K * sizeof(void*), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(ds2, s2.data(), 2 * K * sizeof(float), cudaMemcpyHostToDevice, st));
        strata::kernels::nemo_experts(dptr, dptr + K, ds2, ds2 + K, dw, dlat, dh, drouted, K, L, FF, st);
        strata::kernels::bf16_gemv_fp32_mmvf(drouted, dfc2, dmoe, L, E, st);
        strata::kernels::nvfp4_gemv(dshu, sh_up_s2, dx, dshh, SH, E, true, st);
        strata::kernels::nvfp4_gemv(dshd, sh_dn_s2, dshh, dsh, E, SH, false, st);
        CK(cudaStreamSynchronize(st));
        w_routed = std::max(w_routed, rel(down(drouted, L).data(), routed_ref.data() + (size_t) t * L, L));
        std::vector<float> moe = down(dmoe, E), sh = down(dsh, E);
        for (int64_t i = 0; i < E; ++i) moe[i] += sh[i];
        w_moe = std::max(w_moe, rel(moe.data(), moe_ref.data() + (size_t) t * E, E));
    }
    std::printf("  router logits        worst rel %.3e\n", w_logit);
    std::printf("  selected sets        %d of %d tokens differ from the reference\n", flips, T);
    std::printf("  routing weights      worst rel %.3e\n", w_w);
    std::printf("  routed latent sum    worst rel %.3e\n", w_routed);
    std::printf("  MoE output           worst rel %.3e\n", w_moe);
    const bool ok = w_logit < 1e-5 && flips == 0 && w_w < 1e-5 && w_routed < 1e-5 && w_moe < 1e-5;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
