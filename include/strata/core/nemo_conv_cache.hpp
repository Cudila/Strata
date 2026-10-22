// include/strata/core/nemo_conv_cache.hpp - strata-nemotron: checkpoints inside a conversation and parked conversations.
//
// The Mamba state cannot be rolled back, so a request resumes exactly at a position whose state was set aside:
//   live checkpoints   up to N (--prompt-cache): Mamba state + conv + MTP row + head logits at a position of the
//                      current sequence, in page-locked host memory (~165 MB each, allocated once).  They are taken at
//                      the message boundary, at every --prompt-cache-every tokens of a long read and after generation.
//                      The K/V pages are positional, so a resume needs no K/V copy while NemoModel::kv_gen() is the
//                      checkpoint's (no reset() since): the cells past the position are simply overwritten.
//   parked conversations   a budget of page-locked host memory (--conversation-cache-mib): when another conversation
//                      arrives, the current one is parked (its deepest checkpoint, its deepest message-boundary
//                      checkpoint, and the K/V cells [0, deepest pos) of every owner); a later request that extends its
//                      ids restores the cells and the checkpoint.  Oldest parked conversation out when over budget.
// A resume is exact (copies only): a checkpoint is used only for a prompt whose ids (and picture keys) it is a prefix of.
#pragma once

#include "strata/core/nemo_model.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The pictures of a request: runs of <image> tokens (start, count) and each picture's key ("?" = never matches).
struct ConvImages {
    const std::vector<std::pair<int64_t, int64_t>>* runs = nullptr;
    const std::vector<std::string>* keys = nullptr;
    std::string key(size_t i) const { return keys && i < keys->size() ? (*keys)[i] : std::string("?"); }
};

class ConversationCache {
public:
    enum Kind { Chunk = 0, Message = 1, Answer = 2 };
    struct Choice {
        enum Src { None, Live, Park } src = None;
        int idx = -1;       ///< live slot
        uint64_t id = 0;    ///< parked conversation (stable across evictions)
        int ck = -1;        ///< parked: which of its checkpoints
        int64_t len = 0;    ///< tokens the checkpoint covers
    };

    ConversationCache() = default;
    ~ConversationCache();
    ConversationCache(const ConversationCache&) = delete;
    ConversationCache& operator=(const ConversationCache&) = delete;

    /// Allocates the page-locked pools: n_ckpt checkpoints, park_mib MiB for parked conversations (0 = no parking).
    bool init(NemoModel& m, int n_ckpt, int64_t park_mib, std::string& err);
    bool enabled() const { return n_ > 0; }
    bool parking() const { return arena_cap_ > 0; }

    /// Forgets the live checkpoints of an older K/V generation (a reset() happened).
    void sync(const NemoModel& m);
    void clear_live();

    /// The deepest checkpoint (live or parked) whose ids are a prefix of the prompt, its pictures the request's.
    Choice choose(const std::vector<int32_t>& prompt, const ConvImages& img) const;
    int64_t deepest_live() const;

    /// Takes a checkpoint of the model's current state (at m.pos(), the first m.pos() ids of `ids`).
    bool take(NemoModel& m, const int32_t* ids, const ConvImages& img, Kind kind, std::string& err);
    /// Live slot -> model.  Then the checkpoints deeper than it are dropped (their K/V cells get overwritten).
    bool restore_live(NemoModel& m, int idx, std::vector<float>& logits, std::string& err);
    /// Parks the current conversation (nothing if there is none or no budget); `keep_id` is a parked entry that must stay.
    void park_current(NemoModel& m, uint64_t keep_id);
    /// reset(), then a parked conversation's K/V cells and checkpoint into the model; the entry is removed.
    bool unpark(NemoModel& m, const Choice& c, std::vector<float>& logits, std::string& err);

    size_t parked_count() const { return parked_.size(); }
    size_t parked_bytes() const;
    double last_restore_ms = 0, last_park_ms = 0;

private:
    struct ImgKeys {
        std::vector<std::string> key;
        std::vector<int64_t> start;
        size_t covered(int64_t pos) const {
            size_t n = 0;
            while (n < start.size() && start[n] < pos) ++n;
            return n;
        }
    };
    struct Live {
        NemoCkpt c;
        std::vector<int32_t> ids;
        ImgKeys ik;
        int kind = Chunk;
        uint64_t made = 0, used = 0;
        bool full = false;
    };
    struct Parked {
        std::vector<int32_t> ids;       ///< of the deepest checkpoint
        ImgKeys ik;
        std::vector<NemoCkpt> ck;       ///< deepest first; mamba points into the arena
        std::vector<int> kind;
        int64_t cells = 0;
        size_t off = 0, bytes = 0;
        uint64_t stamp = 0;
    };
    static bool match(const std::vector<int32_t>& ids, int64_t pos, const ImgKeys& ik, const std::vector<int32_t>& prompt,
                      const ConvImages& img);
    static ImgKeys keys_for(int64_t pos, const ConvImages& img);
    int free_slot();
    size_t arena_alloc(size_t bytes);
    void arena_free(size_t off, size_t bytes);
    void drop_parked(size_t i);
    bool take_at(NemoModel& m, const int32_t* ids, int64_t pos, ImgKeys ik, Kind kind, std::string& err);

    int n_ = 0;
    std::vector<Live> live_;
    uint8_t* pool_ = nullptr;
    size_t ckpt_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    size_t arena_cap_ = 0;
    std::map<size_t, size_t> free_;   ///< offset -> length
    std::vector<Parked> parked_;
    uint64_t clock_ = 0;
};

}  // namespace strata::core
