// include/strata/core/nemo_model.hpp - strata-nemotron: the Nemotron-H model runner (v1).
//
// What it holds, from the GGUF of tools/nemotron/convert.py (checked by nemotron_check at load):
//   VRAM  every dense tensor in its own format (NVFP4 / Q8_0 / BF16 matrices, F32 vectors); per sequence the Mamba
//         SSM states (40 x 4 MiB FP32) and conv histories, the attention K/V cache (BF16, 8 KiB per token); an
//         expert cache - the rest of the card, an equal number of slots per MoE layer, least recently used out.
//   RAM   all 20,480 experts in place in the mapped GGUF, page-locked (cudaHostRegister), copied to a slot on a miss.
//         The token embeddings stay in the mapping (one 8 KiB row per token).
//
// forward(tokens, T) appends T tokens at position pos(): each of the 88 blocks runs over the T rows at once (the
// prompt) or one (decode) - the dense projections through nemo_gemm (weights read once per pass), Mamba through the
// prefill scan or the decode step (bit-identical to each other), attention over the cache, the MoE through the
// router, then per expert its tokens (prompt) or the token's 22 experts in one launch (decode).  It returns the last
// row's logits.  No drafts, no CPU expert lane, no profile yet: v1 is the correct whole model the next steps speed up.
//
// A prompt's state can be set aside (snapshot) and restored, so a conversation's next turn reads only what is new:
// the Mamba state cannot be rolled back, so reuse is "continue exactly from the end of a previous prompt".
#pragma once

#include "strata/core/nemotron_layout.hpp"
#include "strata/kernels/nemo_gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct NemoOptions {
    std::string gguf;
    std::string experts_gguf;         ///< the routed experts from another GGUF (e.g. an i-quant file); "" = gguf
    int64_t max_context = 262144;
    int64_t chunk = 2048;             ///< prompt tokens per forward pass
    int64_t vram_reserve_mib = 1536;  ///< left free after the expert cache is sized
    int64_t expert_slots = 0;         ///< per MoE layer; 0 = fill the card
    /// K/V cells a sequence starts with; past them the cache grows in pages of 65,536 cells, each taking its VRAM
    /// from the expert cache (and giving it back at the next reset), up to max_context
    int64_t kv_base = 262144;
    bool mtp = false;                 ///< load the MTP block (resident) for drafting
};

struct NemoStats {
    uint64_t expert_hits = 0, expert_misses = 0, expert_skips = 0;
    uint64_t prefetch_issued = 0, prefetch_used = 0, prefetch_wasted = 0;
    uint64_t spec_windows = 0, spec_drafted = 0, spec_accepted = 0;
    int64_t slots_per_layer = 0;
    int64_t kv_cells = 0;             ///< K/V cells in VRAM now (it grows with the context, see kv_base)
    double expert_cache_gib = 0, dense_gib = 0;
    bool pinned = false;
};

class NemoModel {
public:
    NemoModel();
    ~NemoModel();
    NemoModel(const NemoModel&) = delete;
    NemoModel& operator=(const NemoModel&) = delete;

    bool load(const NemoOptions& o, std::string& err);
    const NemotronGeometry& geometry() const { return g_; }
    int64_t max_context() const { return max_ctx_; }
    int64_t pos() const { return pos_; }

    /// A fresh sequence: position 0, every state zero.
    void reset();
    /// Appends `n` tokens (any n: chunked internally) and writes the last one's logits (n_vocab f32) to `logits`.
    bool forward(const int32_t* tokens, int64_t n, float* logits, std::string& err);
    /// forward() that writes every row's logits: `all` is n * n_vocab (perplexity, comparisons).
    bool forward_all(const int32_t* tokens, int64_t n, float* all, std::string& err);
    /// Images: rows of n_embd floats that take the place of the token rows at runs of absolute positions (start,
    /// count), the runs' rows one after another in `rows`; in the main model and in the MTP block's input alike.
    /// They hold until the next call (an empty `runs` clears them).
    bool set_image_rows(const float* rows, const std::vector<std::pair<int64_t, int64_t>>& runs, std::string& err);

    /// The current state (Mamba states and conv histories, position, the last logits) set aside / put back.  The K/V
    /// cells below the snapshot's position are untouched by later tokens only if they were not overwritten: restore is
    /// valid when every token fed since the snapshot came after it (pos() >= the snapshot's) - always true for the
    /// "next turn continues the previous prompt" case.
    void snapshot();
    bool restore();
    int64_t snapshot_pos() const { return snap_pos_; }
    const std::vector<float>& snapshot_logits() const { return snap_logits_; }

    /// The MTP block: whether it is loaded, and a draft chain - n greedy drafts for the tokens after `next` (the token
    /// about to be fed: the last forward's sampled token).  It writes the MTP's K/V at the positions it drafts for;
    /// the next forward() keeps the first (it is the true one when `next` is fed).
    bool has_mtp() const;
    /// A draft chain stops after a draft whose MTP probability is below p (0 = always the full chain).
    void set_spec_min_p(float p);
    bool mtp_draft(int32_t next, int n, int32_t* drafts, std::string& err);
    /// One greedy speculative step: n MTP drafts after `next`, the window [next, drafts] verified at once, the longest
    /// prefix the main model agrees with kept (the Mamba state rolled back and the kept rows replayed when a draft
    /// is refused).  `out` gets the kept drafts and the main model's token after them (1 .. n + 1 tokens).
    /// `pick` (optional) draws a row's token from its logits (host, n_vocab): sampling; without it, greedy.
    bool spec_step(int32_t next, int n, std::vector<int32_t>& out, std::string& err,
                   const std::function<int32_t(const float*)>* pick = nullptr);

    NemoStats stats() const { return stats_; }
    /// STRATA_NEMO_PROFILE=1: where the time went since the last report (empty when profiling is off).
    std::string profile_report(bool clear);

private:
    bool mtp_rows(const int32_t* toks, const float* h, int64_t n, int64_t pos0, bool full, std::string& err);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    NemotronGeometry g_;
    int64_t max_ctx_ = 0, pos_ = 0, snap_pos_ = -1;
    std::vector<float> snap_logits_;
    NemoStats stats_;
};

}  // namespace strata::core
