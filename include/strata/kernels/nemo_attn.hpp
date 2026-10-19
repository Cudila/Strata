// include/strata/kernels/nemo_attn.hpp - strata-nemotron stage 3: Nemotron-H's attention between the projections.
//
// The model's 8 attention blocks (and the MTP's) are plain causal GQA with NO positional encoding at all: q/k/v come
// straight from the projections (no RoPE, no q/k norm, no bias), 32 query heads over 2 KV heads (query head h reads KV
// head h / 16), head dim 128, scale 1/sqrt(128).  So the K/V cache stores the projections as they are.
//
//     o[t, h] = sum_{j <= p0+t} softmax_j(q[t,h] . k[j, h/16] * scale) v[j, h/16]
//
// ONE kernel serves decode (T = 1), the verify window (T = the drafts + 1) and the prompt (T = a chunk): the T queries
// sit at positions p0 .. p0+T-1 and their K/V rows have been appended at those cells before the call.  The keys are
// split across blocks (split-K), each block keeps an online softmax per head, and a second kernel merges the splits.
//
// K/V cache layout: see NemoKV (int8 K with a scale per cell and head, FP16 V, in pages).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct NemoAttnShapes {
    int64_t n_head = 32, n_head_kv = 2, head_dim = 128;
};

/// The K/V cache, in pages (the engine grows it as the context does, see nemo_model.cpp): cell j is row
/// j & (2^shift - 1) of page j >> shift.  A cell holds, per KV head, K as int8 with one FP32 scale (amax / 127; the
/// prompt's Q.K runs on int8 tensor cores, and a cell is 3/4 the size of a BF16 one) and V as FP16:
///   k[page]  (cells, n_head_kv, head_dim) int8      ks[page]  (cells, n_head_kv) f32      v[page]  (cells, n_head_kv,
///   head_dim) FP16 bits
/// A contiguous cache is one page with a large shift (nemo_kv_flat).
constexpr int NEMO_KV_MAX_PAGES = 64;
struct NemoKV {
    int8_t* k[NEMO_KV_MAX_PAGES] = {};
    float* ks[NEMO_KV_MAX_PAGES] = {};
    uint16_t* v[NEMO_KV_MAX_PAGES] = {};
    int shift = 30;
};
inline NemoKV nemo_kv_flat(int8_t* k, float* ks, uint16_t* v) {
    NemoKV kv;
    kv.k[0] = k;
    kv.ks[0] = ks;
    kv.v[0] = v;
    return kv;
}

/// Bytes of scratch nemo_attn needs for ANY call with at most T queries over at most n_keys keys.
uint64_t nemo_attn_scratch_bytes(const NemoAttnShapes& s, int64_t T, int64_t n_keys);

/// Writes `k_new`/`v_new` (T, n_head_kv * head_dim) f32 into cells p0 .. p0+T-1: K to int8 with its scale per (cell,
/// head), V to FP16 (saturating).
void nemo_kv_append(const NemoKV& kv, const float* k_new, const float* v_new, int64_t p0, int64_t T,
                    const NemoAttnShapes& s, void* stream);

/// q (T, n_head * head_dim) f32 -> o (T, n_head * head_dim) f32, causal over cells 0 .. p0+t for query t.
/// `scratch` is at least nemo_attn_scratch_bytes(s, T_max, keys_max) of device memory for the largest call.
/// `lo` > 0: only cells lo .. (a sliding window for the MTP draft block, whose drafts are verified anyway).
void nemo_attn(const float* q, const NemoKV& kv, float* o, int64_t p0, int64_t T, const NemoAttnShapes& s,
               void* scratch, void* stream, int64_t lo = 0);

}  // namespace strata::kernels
