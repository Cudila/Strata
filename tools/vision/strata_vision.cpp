// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// ---- Nemotron 3.5 Super VL (projector nemotron_omni): NVIDIA's image processor picks the tile itself, and mtmd's
// generic dynamic-size rule picks another one (it rounds 488 px down to 480 where NVIDIA rounds up to 512).  So the
// picture is brought to NVIDIA's size here, with NVIDIA's resampling (torch bicubic, antialias), and mtmd gets a tile
// it keeps as it is.  From image_processing_nemotron_h_omni.py: patch 16, pixel shuffle 2, 1024..13312 patches.

// _compute_target_patches: (w, h) in patches.  Python's round() is round-half-to-even, as nearbyint here.
void nvidia_tile(int w, int h, int& wp, int& hp) {
    const int ps = 16, merge = 2, min_p = 1024;
    const double available = std::max(std::min(std::max((16384.0 - 4) * merge * merge, (double) min_p), 13312.0),
                                      (double) min_p);
    const double ph0 = std::nearbyint(h / (double) ps + 0.5), pw0 = std::nearbyint(w / (double) ps + 0.5);
    const double f = std::min(std::sqrt(available / (ph0 * pw0)), 1.0);
    long th = (long) std::floor(f * ph0), tw = (long) std::floor(f * pw0);
    if (available > min_p && th * tw < min_p) {
        const double up = std::sqrt(min_p / (double) (th * tw));
        th = (long) std::ceil(up * th);
        tw = (long) std::ceil(up * tw);
    }
    long r = th % merge;
    if (r) th = (th + merge - r) * tw <= available ? th + merge - r : std::max<long>(merge, th - r);
    r = tw % merge;
    if (r) tw = th * (tw + merge - r) <= available ? tw + merge - r : std::max<long>(merge, tw - r);
    wp = (int) tw;
    hp = (int) th;
}

// torch's antialiased bicubic (PIL's filter: a = -0.5, support 2 x the downscale), along one axis of a float image.
// The arithmetic is unchanged from the single-threaded version (double sums, taps added in ascending order), so the
// bytes are identical; only the work is split over threads.  An axis whose length does not change is the identity
// (the weights come out exactly 0, 1, 0, 0) and the caller skips it.
void parallel_for(int n, int n_threads, const std::function<void(int, int)>& fn) {
    n_threads = std::max(1, std::min(n_threads, n));
    if (n_threads == 1) { fn(0, n); return; }
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        const int lo = (int) ((long long) n * t / n_threads), hi = (int) ((long long) n * (t + 1) / n_threads);
        pool.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    for (auto& th : pool) th.join();
}

void aa_axis(const std::vector<float>& in, int w, int h, int c, int out_len, bool horizontal, std::vector<float>& out) {
    const int in_len = horizontal ? w : h;
    const double scale = (double) in_len / out_len, fs = std::max(scale, 1.0), support = 2.0 * fs;
    const auto cubic = [](double x) {
        const double a = -0.5;
        x = std::fabs(x);
        return x < 1.0 ? ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                       : x < 2.0 ? (((x - 5.0) * x + 8.0) * x - 4.0) * a : 0.0;
    };
    const int ow = horizontal ? out_len : w, oh = horizontal ? h : out_len;
    out.assign((size_t) ow * oh * c, 0.0f);
    // the taps of every output position, computed once
    std::vector<int> first(out_len), count(out_len);
    std::vector<size_t> at(out_len);
    std::vector<double> wts;
    for (int i = 0; i < out_len; ++i) {
        const double center = scale * (i + 0.5);
        const int x0 = std::max((int) (center - support + 0.5), 0);
        const int x1 = std::min((int) (center + support + 0.5), in_len);
        first[i] = x0;
        count[i] = x1 - x0;
        at[i] = wts.size();
        double total = 0.0;
        for (int j = x0; j < x1; ++j) {
            const double v = cubic((j - center + 0.5) / fs);
            total += v;
            wts.push_back(v);
        }
        for (int j = x0; j < x1; ++j) {
            double& v = wts[at[i] + (size_t) (j - x0)];
            v = total != 0.0 ? v / total : 0.0;
        }
    }
    const int n_threads = (int) std::max(4u, std::thread::hardware_concurrency() / 2);
    if (horizontal) {   // split by rows: each row is read and written contiguously
        parallel_for(h, n_threads, [&](int r0, int r1) {
            for (int o = r0; o < r1; ++o)
                for (int i = 0; i < out_len; ++i) {
                    const double* wt = &wts[at[i]];
                    for (int ch = 0; ch < c; ++ch) {
                        double s = 0.0;
                        for (int k = 0; k < count[i]; ++k) s += wt[k] * in[((size_t) o * w + first[i] + k) * c + ch];
                        out[((size_t) o * ow + i) * c + ch] = (float) s;
                    }
                }
        });
    } else {            // split by output rows
        parallel_for(out_len, n_threads, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) {
                const double* wt = &wts[at[i]];
                for (int o = 0; o < w; ++o)
                    for (int ch = 0; ch < c; ++ch) {
                        double s = 0.0;
                        for (int k = 0; k < count[i]; ++k) s += wt[k] * in[((size_t) (first[i] + k) * w + o) * c + ch];
                        out[((size_t) i * ow + o) * c + ch] = (float) s;
                    }
            }
        });
    }
}

// RGB bytes (w x h) -> NVIDIA's tile size, rounded back to bytes; empty when it already has that size
std::vector<unsigned char> nvidia_resize(const unsigned char* rgb, int w, int h, int& ow, int& oh) {
    int wp = 0, hp = 0;
    nvidia_tile(w, h, wp, hp);
    ow = wp * 16;
    oh = hp * 16;
    if (ow == w && oh == h) return {};
    std::vector<float> a((size_t) w * h * 3), b, d;
    for (size_t i = 0; i < a.size(); ++i) a[i] = rgb[i];
    // torch: width first, then height; an axis that keeps its length is passed through untouched
    if (ow != w) aa_axis(a, w, h, 3, ow, true, b); else b = std::move(a);
    if (oh != h) aa_axis(b, ow, h, 3, oh, false, d); else d = std::move(b);
    std::vector<unsigned char> out(d.size());
    for (size_t i = 0; i < d.size(); ++i) out[i] = (unsigned char) std::clamp(std::lround(d[i]), 0L, 255L);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model;
    bool gpu = false;
    int threads = 0, max_tokens = 0, min_tokens = 0;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") min_tokens = std::atoi(next().c_str());   // mtmd image_min_tokens (#767)
        else if (a == "--flash-attn") {   // FA keeps K and V in FP16; off = the attention in FP32
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
                             "[--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]\n");
        return 2;
    }
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();

    const auto load_t0 = std::chrono::steady_clock::now();
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no posix_fadvise
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* text = llama_model_load_from_file(model.c_str(), mp);
    if (!text) { std::printf("ERR cannot open the text model %s\n", model.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    // On the CPU "auto" turns flash attention on.  ggml's fast (tiled) CPU kernel for it needs the head size to be a
    // multiple of the vector width: 8 floats with AVX2 (the release builds), 16 with AVX-512 (a build from source on a
    // CPU that has it), where ggml falls back to a kernel that is several times slower and accumulates in FP16.  A
    // 1024x1024 picture on a Ryzen 7 7700X, 8 threads: AVX2 8-10 s with it, 14-15 s without; AVX-512 44 s with it
    // (26% off the FP32 attention's output), 13 s without.  Those numbers were taken with the Qwen vision tower (head
    // size 72, not a multiple of 16).  RADIO-H's head is 80 (1280/16), a multiple of 16, so the explanation does not
    // carry over to this encoder; the AVX-512 default stays until it is measured here.
    if (!gpu && !fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    // on the CPU without --threads: one per core (mtmd's own default is 4 threads)
    if (threads <= 0 && !gpu) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::printf("ERR cannot load the vision encoder %s\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "strata-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // a vocab-only model reports no hparams, so the width comes from the projector: its output is the model's input
    int n_embd = 0;
    bool nemotron = false;   // projector nemotron_omni: NVIDIA's tile size and resampling (nvidia_resize)
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int) gguf_get_val_u32(gg, k);
        const int64_t pk = gg ? gguf_find_key(gg, "clip.projector_type") : -1;
        if (pk >= 0) nemotron = std::string(gguf_get_val_str(gg, pk)) == "nemotron_omni";
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) { std::printf("ERR the vision encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left; allocating ~1 GB later, on a GPU the engine has filled, made Windows page GPU memory and
    // the engine crawl to a standstill.  (A square image well above any cap; mtmd scales it to the token limit.)
    // On the CPU there is no VRAM to reserve, and the warm-up would only delay the engine's start by one encode
    // (~6 s at 1,024 tokens).
    if (!gpu) std::fprintf(stderr, "strata-vision: on the CPU, %d threads, no warm-up\n", threads);
    else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const uint32_t side = 2048;
        std::vector<unsigned char> rgb((size_t) side * side * 3, 128);
        mtmd_bitmap* bm = mtmd_bitmap_init(side, side, rgb.data());
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bms[1] = {bm};
        int warm_tokens = 0;
        if (bm && mtmd_tokenize(ctx, chunks, &txt, bms, 1) == 0) {
            for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && mtmd_encode_chunk(ctx, ch) == 0)
                    warm_tokens = (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s\n", warm_tokens,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
        mtmd_input_chunks_free(chunks);
        if (bm) mtmd_bitmap_free(bm);
    }
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        auto tms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        double dec_ms = 0, rs_ms = 0, tok_ms = 0, enc_ms = 0;
        int src_w = 0, src_h = 0, out_w = 0, out_h = 0;
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false,
                                                                            mtmd_helper_init_opt_default());
        const auto t_dec = std::chrono::steady_clock::now();
        dec_ms = tms(t0, t_dec);
        if (!bw.bitmap) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        src_w = (int) mtmd_bitmap_get_nx(bw.bitmap); src_h = (int) mtmd_bitmap_get_ny(bw.bitmap);
        out_w = src_w; out_h = src_h;
        if (nemotron) {   // NVIDIA's tile (see nvidia_resize); mtmd keeps a tile of that size as it is
            int ow = 0, oh = 0;
            const std::vector<unsigned char> px = nvidia_resize(mtmd_bitmap_get_data(bw.bitmap),
                                                                (int) mtmd_bitmap_get_nx(bw.bitmap),
                                                                (int) mtmd_bitmap_get_ny(bw.bitmap), ow, oh);
            if (!px.empty()) {
                if (const char* dump = std::getenv("STRATA_VISION_DUMP_RGB")) {   // debug: the resized RGB bytes
                    if (FILE* df = std::fopen(dump, "wb")) { std::fwrite(px.data(), 1, px.size(), df); std::fclose(df); }
                }
                mtmd_bitmap_free(bw.bitmap);
                bw.bitmap = mtmd_bitmap_init((uint32_t) ow, (uint32_t) oh, px.data());
                out_w = ow; out_h = oh;
            }
        }
        const auto t_rs = std::chrono::steady_clock::now();
        rs_ms = tms(t_dec, t_rs);
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the image could not be preprocessed";
        const auto t_tok = std::chrono::steady_clock::now();
        tok_ms = tms(t_rs, t_tok);
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no image chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the vision encoder failed";
        const auto t_enc = std::chrono::steady_clock::now();
        enc_ms = tms(t_tok, t_enc);
        if (err.empty()) {
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ichunk);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n && !mtmd_decode_use_mrope(ctx)) {   // no 2-D positions (nemotron_omni): one row
                nx = n;
                ny = 1;
            }
            if (nx * ny != n) err = "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = err.empty() ? std::fopen(out.c_str(), "wb") : nullptr;
            if (err.empty() && !f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                const auto t_wr = std::chrono::steady_clock::now();
                std::fprintf(stderr, "strata-vision: decode %.1f ms, resize %.1f ms, tokenize %.1f ms, encode %.1f ms, write %.1f ms, total %.1f ms (%d tokens, in %dx%d -> out %dx%d)\n",
                             dec_ms, rs_ms, tok_ms, enc_ms, tms(t_enc, t_wr), tms(t0, t_wr), n, src_w, src_h, out_w, out_h);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
