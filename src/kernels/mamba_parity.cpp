// src/kernels/mamba_parity.cpp - strata-nemotron stage 2 test: the Mamba-2 decode kernels (mamba.hpp) against the
// float64 reference `tools/nemotron/mamba_naive.py --dump DIR`, on a real layer's weights and real activations.
//
// Two paths over the same T tokens, each with its own state:
//   core  the reference's own zxbcdt -> conv -> ssm -> gated norm, checked against the reference's y_core: the
//         kernels between the projections, isolated from any GEMV
//   full  the reference's mixer input -> in_proj (BF16, bf16_gemv_fp32_mmvf: FP32 activations, exact products) ->
//         the same kernels -> out_proj, checked against the reference's mixer output
// and at the end both paths' SSM and conv states against the reference's final ones.
//
// The error measure is ||got - want|| / ||want|| per vector (the worst token is reported).  Tolerances: the kernels
// are FP32 against FP64, and the state is carried over T steps, so 2e-5 for the core; the full path adds two FP32
// GEMV summation orders over 4096 / 8192 terms, so 1e-4.  A wrong head-group mapping, gate order or conv tap gives
// errors of order 1 (checked by --selftest-negative, which breaks each on purpose).
//
// Usage: mamba_parity <dump dir> [--negative]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/mamba.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::kernels::MambaShapes;

namespace {

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                        \
        }                                                                                        \
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
    size_t at = json.find(k);
    if (at == std::string::npos) {
        std::fprintf(stderr, "meta.json has no %s\n", key);
        std::exit(2);
    }
    return std::atoll(json.c_str() + at + k.size());
}

double meta_double(const std::string& json, const char* key) {
    const std::string k = std::string("\"") + key + "\":";
    size_t at = json.find(k);
    return at == std::string::npos ? 0.0 : std::atof(json.c_str() + at + k.size());
}

template <class T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

float* dev(size_t n) {
    float* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemset(d, 0, n * sizeof(float)));
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

struct Worst {
    double v = 0;
    int t = -1;
    void take(double x, int at) {
        if (x > v || t < 0) { v = x; t = at; }
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dump dir from tools/nemotron/mamba_naive.py --dump> [--negative]\n", argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "--bench") == 0) {
        // the prefill kernels on one 8192-token chunk of Nemotron-H's geometry (random data): ms per layer
        MambaShapes b;
        b.H = 128; b.P = 64; b.N = 128; b.G = 8; b.d_conv = 4;
        const int64_t T = argc > 2 ? std::atoll(argv[2]) : 8192, DI = b.d_inner(), CD = b.conv_dim();
        const int64_t IN = DI + CD + b.H;
        std::vector<float> h((size_t) T * IN);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (float) ((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
        float* zxd = up(h);
        std::vector<float> small(CD * 4, 0.1f), hv(b.H, -1.0f);
        float *cw = up(small), *cb = up(small), *dtb = up(hv), *Ad = up(hv), *Dd = up(hv), *nw = up(std::vector<float>(DI, 1));
        float *st = dev((size_t) b.H * b.P * b.N), *cs = dev((size_t) CD * 3);
        float *xbc = dev((size_t) T * CD), *y = dev((size_t) T * DI);
        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
        for (int rep = 0; rep < 3; ++rep) {
            cudaEventRecord(e0);
            strata::kernels::mamba_conv_prefill(cs, zxd, IN, cw, cb, xbc, T, b, nullptr);
            cudaEventRecord(e1);
            strata::kernels::mamba_ssm_scan(st, xbc, zxd, IN, dtb, Ad, Dd, y, T, b, nullptr);
            cudaEventRecord(e2);
            strata::kernels::mamba_gated_norm_rows(y, zxd, IN, nw, y, T, DI, b.G, 1e-5f, nullptr);
            cudaEventRecord(e3);
            CK(cudaEventSynchronize(e3));
            float a, c, d;
            cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&c, e1, e2); cudaEventElapsedTime(&d, e2, e3);
            std::printf("T %lld: conv %.3f ms, scan %.3f ms, gated norm %.3f ms per layer; x40 layers %.1f ms "
                        "(%.0f tokens/s for these parts alone)\n", (long long) T, a, c, d, 40 * (a + c + d),
                        T / (40 * (a + c + d) / 1000.0));
        }
        return 0;
    }
    const std::string dir = argv[1];
    const bool negative = argc > 2 && std::strcmp(argv[2], "--negative") == 0;
    std::string json;
    {
        FILE* f = std::fopen((dir + "/meta.json").c_str(), "rb");
        if (!f) { std::fprintf(stderr, "no %s/meta.json\n", dir.c_str()); return 2; }
        char buf[4096];
        json.assign(buf, std::fread(buf, 1, sizeof(buf), f));
        std::fclose(f);
    }
    MambaShapes s;
    s.H = meta_int(json, "H"); s.P = meta_int(json, "P"); s.N = meta_int(json, "N"); s.G = meta_int(json, "G");
    s.d_conv = meta_int(json, "d_conv");
    const int T = (int) meta_int(json, "T");
    const int64_t E = meta_int(json, "n_embd");
    const float eps = (float) meta_double(json, "eps");
    const int64_t DI = s.d_inner(), CD = s.conv_dim(), H = s.H, IN = DI + CD + H;
    std::printf("mamba_parity: layer %lld, %d tokens, H %lld P %lld N %lld G %lld, in_proj %lld\n",
                meta_int(json, "layer"), T, (long long) s.H, (long long) s.P, (long long) s.N, (long long) s.G,
                (long long) IN);

    // weights and the reference
    auto w_in = load<uint16_t>(dir, "w_in_proj.bf16", (size_t) IN * E);
    auto w_out = load<uint16_t>(dir, "w_out_proj.bf16", (size_t) E * DI);
    auto conv_w = load<float>(dir, "conv_w.f32", (size_t) CD * s.d_conv);
    auto conv_b = load<float>(dir, "conv_b.f32", CD);
    auto dt_bias = load<float>(dir, "dt_bias.f32", H);
    auto A = load<float>(dir, "A.f32", H);
    auto D = load<float>(dir, "D.f32", H);
    auto norm_w = load<float>(dir, "norm_w.f32", DI);
    auto x = load<float>(dir, "x.f32", (size_t) T * E);
    auto zx = load<float>(dir, "zxbcdt.f32", (size_t) T * IN);
    auto y_ref = load<float>(dir, "y_core.f32", (size_t) T * DI);
    auto out_ref = load<float>(dir, "out.f32", (size_t) T * E);
    auto st_ref = load<float>(dir, "state.f32", (size_t) H * s.P * s.N);
    auto cs_ref = load<float>(dir, "conv_state.f32", (size_t) CD * (s.d_conv - 1));

    if (negative) {
        // break the one thing a wrong transcription would break, and require the test to see it: the conv's newest tap
        for (int64_t c = 0; c < CD; ++c) std::swap(conv_w[c * s.d_conv], conv_w[c * s.d_conv + s.d_conv - 1]);
    }

    uint16_t *d_win = up(w_in), *d_wout = up(w_out);
    float *d_cw = up(conv_w), *d_cb = up(conv_b), *d_dtb = up(dt_bias), *d_A = up(A), *d_D = up(D), *d_nw = up(norm_w);
    const size_t state_n = (size_t) H * s.P * s.N, cs_n = (size_t) CD * (s.d_conv - 1);
    float *st1 = dev(state_n), *st2 = dev(state_n), *cs1 = dev(cs_n), *cs2 = dev(cs_n);
    float *d_zx = dev(IN), *d_x = dev(E), *d_xbc = dev(CD), *d_y = dev(DI), *d_out = dev(E);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    Worst w_core, w_gemv, w_full_y, w_full;
    std::vector<float> y_decode((size_t) T * DI);   // the core path's outputs, for the prefill's bit-identity check
    for (int t = 0; t < T; ++t) {
        // core: the reference's zxbcdt
        CK(cudaMemcpyAsync(d_zx, zx.data() + (size_t) t * IN, IN * sizeof(float), cudaMemcpyHostToDevice, stream));
        strata::kernels::mamba_conv_step(cs1, d_zx + DI, d_cw, d_cb, d_xbc, CD, s.d_conv, stream);
        strata::kernels::mamba_ssm_step(st1, d_xbc, d_zx + DI + CD, d_dtb, d_A, d_D, d_y, s, stream);
        strata::kernels::mamba_gated_norm(d_y, d_zx, d_nw, d_y, DI, s.G, eps, stream);
        CK(cudaStreamSynchronize(stream));
        {
            const std::vector<float> yt = down(d_y, DI);
            std::memcpy(y_decode.data() + (size_t) t * DI, yt.data(), DI * sizeof(float));
            w_core.take(rel(yt.data(), y_ref.data() + (size_t) t * DI, DI), t);
        }

        // full: the mixer input through the BF16 projections
        CK(cudaMemcpyAsync(d_x, x.data() + (size_t) t * E, E * sizeof(float), cudaMemcpyHostToDevice, stream));
        strata::kernels::bf16_gemv_fp32_mmvf(d_x, d_win, d_zx, E, IN, stream);
        CK(cudaStreamSynchronize(stream));
        w_gemv.take(rel(down(d_zx, IN).data(), zx.data() + (size_t) t * IN, IN), t);
        strata::kernels::mamba_conv_step(cs2, d_zx + DI, d_cw, d_cb, d_xbc, CD, s.d_conv, stream);
        strata::kernels::mamba_ssm_step(st2, d_xbc, d_zx + DI + CD, d_dtb, d_A, d_D, d_y, s, stream);
        strata::kernels::mamba_gated_norm(d_y, d_zx, d_nw, d_y, DI, s.G, eps, stream);
        CK(cudaStreamSynchronize(stream));
        w_full_y.take(rel(down(d_y, DI).data(), y_ref.data() + (size_t) t * DI, DI), t);
        strata::kernels::bf16_gemv_fp32_mmvf(d_y, d_wout, d_out, DI, E, stream);
        CK(cudaStreamSynchronize(stream));
        w_full.take(rel(down(d_out, E).data(), out_ref.data() + (size_t) t * E, E), t);
    }
    // prefill: the reference's zxbcdt for all T tokens, read in two chunks (37 + the rest: a chunk that is not a
    // multiple of the scan's 16-token tile, and the state and conv history carried across)
    float *st3 = dev(state_n), *cs3 = dev(cs_n);
    float* d_zxT = up(zx);
    float *d_xbcT = dev((size_t) T * CD), *d_yT = dev((size_t) T * DI);
    const int T1 = T > 37 ? 37 : T / 2;
    for (int part = 0; part < 2; ++part) {
        const int t0 = part ? T1 : 0, n = part ? T - T1 : T1;
        const float* zx0 = d_zxT + (size_t) t0 * IN;
        strata::kernels::mamba_conv_prefill(cs3, zx0, IN, d_cw, d_cb, d_xbcT + (size_t) t0 * CD, n, s, stream);
        strata::kernels::mamba_ssm_scan(st3, d_xbcT + (size_t) t0 * CD, zx0, IN, d_dtb, d_A, d_D,
                                        d_yT + (size_t) t0 * DI, n, s, stream);
        strata::kernels::mamba_gated_norm_rows(d_yT + (size_t) t0 * DI, zx0, IN, d_nw, d_yT + (size_t) t0 * DI, n, DI,
                                               s.G, eps, stream);
    }
    CK(cudaStreamSynchronize(stream));
    const std::vector<float> yT = down(d_yT, (size_t) T * DI);
    Worst w_pre;
    for (int t = 0; t < T; ++t) w_pre.take(rel(yT.data() + (size_t) t * DI, y_ref.data() + (size_t) t * DI, DI), t);
    const std::vector<float> st1h = down(st1, state_n), st3h = down(st3, state_n);
    const std::vector<float> cs1h = down(cs1, cs_n), cs3h = down(cs3, cs_n);
    const bool same_y = std::memcmp(yT.data(), y_decode.data(), yT.size() * sizeof(float)) == 0;
    const bool same_st = std::memcmp(st1h.data(), st3h.data(), state_n * sizeof(float)) == 0;
    const bool same_cs = std::memcmp(cs1h.data(), cs3h.data(), cs_n * sizeof(float)) == 0;

    // the chunked SSD form (a prompt chunk of >= 64 rows): all T tokens in one call, against the reference; then a
    // long synthetic sequence (real B, C, x statistics do not matter for the algebra) against the sequential scan
    Worst w_ssd;
    double e_st_ssd = 0, e_syn_y = 0, e_syn_st = 0;
    {
        float *st4 = dev(state_n), *cs4 = dev(cs_n);
        float *d_xbc4 = dev((size_t) T * CD), *d_y4 = dev((size_t) T * DI);
        strata::kernels::mamba_conv_prefill(cs4, d_zxT, IN, d_cw, d_cb, d_xbc4, T, s, stream);
        strata::kernels::mamba_ssm_scan(st4, d_xbc4, d_zxT, IN, d_dtb, d_A, d_D, d_y4, T, s, stream);
        strata::kernels::mamba_gated_norm_rows(d_y4, d_zxT, IN, d_nw, d_y4, T, DI, s.G, eps, stream);
        CK(cudaStreamSynchronize(stream));
        const std::vector<float> y4 = down(d_y4, (size_t) T * DI);
        for (int t = 0; t < T; ++t) w_ssd.take(rel(y4.data() + (size_t) t * DI, y_ref.data() + (size_t) t * DI, DI), t);
        e_st_ssd = rel(down(st4, state_n).data(), st_ref.data(), state_n);
        // synthetic: 2,000 rows from the real ones (cycled), dt from the real raw dt, two calls (1,000 + 1,000)
        const int TS = 2000;
        std::vector<float> zs((size_t) TS * IN), xs((size_t) TS * CD);
        const std::vector<float> xbc_real = down(d_xbc4, (size_t) T * CD);
        for (int t = 0; t < TS; ++t) {
            std::memcpy(zs.data() + (size_t) t * IN, zx.data() + (size_t) (t % T) * IN, IN * sizeof(float));
            std::memcpy(xs.data() + (size_t) t * CD, xbc_real.data() + (size_t) ((t * 7) % T) * CD, CD * sizeof(float));
        }
        float *dzs = up(zs), *dxs = up(xs), *sa = dev(state_n), *sb = dev(state_n);
        float *ya = dev((size_t) TS * DI), *yb = dev((size_t) TS * DI);
        for (int part = 0; part < 2; ++part) {
            const int t0 = part * (TS / 2);
            strata::kernels::mamba_ssm_scan(sa, dxs + (size_t) t0 * CD, dzs + (size_t) t0 * IN, IN, d_dtb, d_A, d_D,
                                            ya + (size_t) t0 * DI, TS / 2, s, stream);
        }
        setenv("STRATA_NEMO_SEQ_SCAN", "1", 1);
        strata::kernels::mamba_ssm_scan(sb, dxs, dzs, IN, d_dtb, d_A, d_D, yb, TS, s, stream);
        unsetenv("STRATA_NEMO_SEQ_SCAN");
        CK(cudaStreamSynchronize(stream));
        const std::vector<float> yah = down(ya, (size_t) TS * DI), ybh = down(yb, (size_t) TS * DI);
        for (int t = 0; t < TS; ++t)
            e_syn_y = std::fmax(e_syn_y, rel(yah.data() + (size_t) t * DI, ybh.data() + (size_t) t * DI, DI));
        e_syn_st = rel(down(sa, state_n).data(), down(sb, state_n).data(), state_n);
    }

    const double e_st1 = rel(down(st1, state_n).data(), st_ref.data(), state_n);
    const double e_st2 = rel(down(st2, state_n).data(), st_ref.data(), state_n);
    const double e_cs1 = rel(down(cs1, cs_n).data(), cs_ref.data(), cs_n);

    std::printf("  core  y (gated norm out)   worst rel %.3e (token %d)\n", w_core.v, w_core.t);
    std::printf("  core  final SSM state      rel %.3e, conv state rel %.3e\n", e_st1, e_cs1);
    std::printf("  full  in_proj GEMV         worst rel %.3e (token %d)\n", w_gemv.v, w_gemv.t);
    std::printf("  full  y                    worst rel %.3e (token %d)\n", w_full_y.v, w_full_y.t);
    std::printf("  full  mixer output         worst rel %.3e (token %d), final state rel %.3e\n", w_full.v, w_full.t,
                e_st2);
    std::printf("  prefill (%d + %d tokens)   worst rel %.3e (token %d); bit-identical to decode: y %s, state %s, "
                "conv %s\n", T1, T - T1, w_pre.v, w_pre.t, same_y ? "yes" : "NO", same_st ? "yes" : "NO",
                same_cs ? "yes" : "NO");
    std::printf("  ssd   (%d tokens, one call) worst rel %.3e (token %d), final state rel %.3e; 2,000 synthetic rows "
                "vs the sequential scan: y %.3e, state %.3e\n", T, w_ssd.v, w_ssd.t, e_st_ssd, e_syn_y, e_syn_st);
    const bool ok = w_core.v < 2e-5 && e_st1 < 2e-5 && e_cs1 < 1e-6 && w_full.v < 1e-4 && e_st2 < 1e-4 &&
                    w_pre.v < 2e-5 && (negative || (same_y && same_st && same_cs)) && w_ssd.v < 2e-5 &&
                    e_st_ssd < 2e-5 && e_syn_y < 1e-4 && e_syn_st < 1e-4;
    if (negative) {
        std::printf(ok ? "FAIL: the broken conv tap was not detected\n" : "PASS (negative): broken conv tap detected\n");
        return ok ? 1 : 0;
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
