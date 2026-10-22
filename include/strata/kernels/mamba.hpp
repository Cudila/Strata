// include/strata/kernels/mamba.hpp - strata-nemotron stage 2: the Mamba-2 mixer's non-projection parts, decode.
//
// Transcribed from `tools/nemotron/mamba_naive.py` (float64, token by token), which is the specification; the
// projections in_proj / out_proj are GEMVs of whatever format the pack holds.  One token:
//
//     zxbcdt = in_proj @ x                    z (d_inner) | xBC (conv_dim = d_inner + 2*G*N) | dt (H)
//     xBC    = silu(conv_k(xBC) + bias)       depthwise causal conv over the last d_conv inputs   mamba_conv_step
//     x, B, C = split(xBC)                    x (H, P);  B, C (G, N)
//     dt     = softplus(dt + dt_bias)         no clamp (Nemotron: time_step_limit 0..inf)
//     h[h,p,n] = exp(dt*A) * h + dt * x[h,p] * B[g,n]            g = h / (H/G)            mamba_ssm_step
//     y[h,p] = sum_n h[h,p,n] * C[g,n] + D[h] * x[h,p]
//     y      = rms_norm_per_group(y * silu(z)) * w      groups of d_inner/G               mamba_gated_norm
//     out    = out_proj @ y
//
// Real geometry (Nemotron 3.5 Super): H = 128 heads, P = 64, N = 128, G = 8 (16 heads share B and C), d_conv = 4,
// d_inner 8192, conv_dim 10240.  The SSM state is FP32, (H, P, N) with n fastest: 4 MiB per layer per sequence.
//
// THE THREE THINGS THAT ARE EASY TO GET WRONG, each giving a plausible number:
//   * the group of a head is h / (H/G) (INTERLEAVE: heads 0..15 use group 0), not h % G - Mamba-2's ngroups;
//   * SiLU(z) multiplies y BEFORE the norm (norm_before_gate = False), and the norm is over groups of 1024, not
//     over all 8192;
//   * the conv's newest input is tap d_conv-1 (`kW[c*d_conv + d_conv-1]`), as in ggml_ssm_conv and gdn_conv_step.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct MambaShapes {
    int64_t H = 0;        ///< heads
    int64_t P = 0;        ///< head dim
    int64_t N = 0;        ///< state size (a multiple of 32)
    int64_t G = 0;        ///< groups of B and C
    int64_t d_conv = 0;
    int64_t d_inner() const { return H * P; }
    int64_t conv_dim() const { return H * P + 2 * G * N; }
};

/// The conv step with bias and SiLU: `out[c] = silu(sum_i inp[i][c] * kW[c*d_conv + i] + bias[c])`,
/// `inp = [conv_state | x]`, and the state slides.  `conv_state` is (channels, d_conv-1), per channel OLDEST FIRST
/// (gdn_conv_step's layout).  `x` and `out` may not alias.
void mamba_conv_step(float* conv_state, const float* x, const float* kW, const float* bias, float* out,
                     int64_t channels, int64_t d_conv, void* stream);

/// One recurrence step for every (h, p), IN PLACE on `state` (H, P, N).  `xbc` is the conv output (x | B | C),
/// `dt_raw` the in_proj's last H values; `dt_bias`, `A` (= -exp(A_log)) and `D` are (H,).  Writes `y` (H*P),
/// before the gate and norm.
void mamba_ssm_step(float* state, const float* xbc, const float* dt_raw, const float* dt_bias, const float* A,
                    const float* D, float* y, const MambaShapes& s, void* stream);

/// `out = rms_norm over groups of d_inner/G (y * silu(z)) * w`.  `out` may alias `y`.
void mamba_gated_norm(const float* y, const float* z, const float* w, float* out, int64_t d_inner, int64_t G,
                      float eps, void* stream);

// ---- prefill: T tokens at once ----
//
// THE SCAN IS SEQUENTIAL IN t AND PARALLEL OVER THE H*P ROWS, with each row's state in registers for the whole
// chunk, instead of Mamba-2's chunked SSD.  Per token it performs the decode step's arithmetic in the decode step's
// order, so a prompt read through these kernels leaves BIT-IDENTICAL state, conv history and outputs to the same
// tokens decoded one by one (mamba_parity checks it) - the property the verify window relies on.  It costs O(T)
// latency per row, which an 8K chunk spends in a few ms per layer; the chunked form is the next step only if the
// prefill profile says the scan is what limits it.
//
// Layouts: `zx` is the in_proj output for T tokens, row-major with row stride `ld_zx` (= the in_proj width), so the
// conv input of token t is `zx + t*ld_zx + d_inner` and its dt is `zx + t*ld_zx + d_inner + conv_dim`; z is the
// row's first d_inner.  `xbc` is (T, conv_dim), `y` is (T, d_inner).

/// The conv over T tokens with the carried history: out[t][c] = silu(sum_i in[t-(d_conv-1)+i][c] * kW[c*d_conv+i] +
/// bias[c]), in[] before t = 0 coming from `conv_state`, which is then the last d_conv-1 inputs.  With
/// `conv_state_out` the history is read from `conv_state` (left untouched) and the new one written there.
void mamba_conv_prefill(float* conv_state, const float* zx, int64_t ld_zx, const float* kW, const float* bias,
                        float* xbc, int64_t T, const MambaShapes& s, void* stream, float* conv_state_out = nullptr);

/// The recurrence over T tokens from `state`, which is left after the last one; `y` before the gate and norm.
/// With `state_out` (T < 64, the sequential scan only) `state` is left untouched and the new state goes there.
void mamba_ssm_scan(float* state, const float* xbc, const float* zx, int64_t ld_zx, const float* dt_bias,
                    const float* A, const float* D, float* y, int64_t T, const MambaShapes& s, void* stream,
                    float* state_out = nullptr);

/// mamba_gated_norm for T tokens: y (T, d_inner), z = zx row t's first d_inner.
void mamba_gated_norm_rows(const float* y, const float* zx, int64_t ld_zx, const float* w, float* out,
                           int64_t T, int64_t d_inner, int64_t G, float eps, void* stream);

}  // namespace strata::kernels
