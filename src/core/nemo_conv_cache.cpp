// src/core/nemo_conv_cache.cpp - strata-nemotron: checkpoints and parked conversations (see the header).
#include "strata/core/nemo_conv_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
size_t up4k(size_t v) { return (v + 4095) & ~(size_t) 4095; }
}  // namespace

ConversationCache::~ConversationCache() {
    if (pool_) cudaFreeHost(pool_);
    if (arena_) cudaFreeHost(arena_);
}

bool ConversationCache::init(NemoModel& m, int n_ckpt, int64_t park_mib, std::string& err) {
    n_ = std::max(0, n_ckpt);
    ckpt_bytes_ = up4k(m.ckpt_bytes());
    if (n_ > 0) {
        if (cudaMallocHost((void**) &pool_, (size_t) n_ * ckpt_bytes_) != cudaSuccess) {
            cudaGetLastError();
            err = "no page-locked memory for " + std::to_string(n_) + " checkpoints (" +
                  std::to_string((size_t) n_ * ckpt_bytes_ >> 20) + " MiB)";
            n_ = 0;
            return false;
        }
        live_.assign((size_t) n_, Live());
        for (int i = 0; i < n_; ++i) live_[(size_t) i].c.mamba = pool_ + (size_t) i * ckpt_bytes_;
    }
    if (park_mib > 0 && n_ > 0) {
        const size_t cap = (size_t) park_mib << 20;
        if (cudaMallocHost((void**) &arena_, cap) != cudaSuccess) {
            cudaGetLastError();
            arena_ = nullptr;
            err = "no page-locked memory for the conversation cache (" + std::to_string(park_mib) + " MiB)";
            return false;
        }
        arena_cap_ = cap;
        free_[0] = cap;
    }
    return true;
}

void ConversationCache::sync(const NemoModel& m) {
    for (Live& l : live_)
        if (l.full && l.c.kv_gen != m.kv_gen()) l.full = false;
}

void ConversationCache::clear_live() {
    for (Live& l : live_) l.full = false;
}

int64_t ConversationCache::deepest_live() const {
    int64_t d = -1;
    for (const Live& l : live_)
        if (l.full) d = std::max(d, l.c.pos);
    return d;
}

ConversationCache::ImgKeys ConversationCache::keys_for(int64_t pos, const ConvImages& img) {
    ImgKeys k;
    if (!img.runs) return k;
    for (size_t i = 0; i < img.runs->size() && (*img.runs)[i].first < pos; ++i) {
        k.key.push_back(img.key(i));
        k.start.push_back((*img.runs)[i].first);
    }
    return k;
}

bool ConversationCache::match(const std::vector<int32_t>& ids, int64_t pos, const ImgKeys& ik,
                              const std::vector<int32_t>& prompt, const ConvImages& img) {
    if (pos <= 0 || (size_t) pos > ids.size() || (size_t) pos > prompt.size()) return false;
    if (!std::equal(ids.begin(), ids.begin() + pos, prompt.begin())) return false;
    // the pictures the checkpoint has (partly) read: the same ones, at the same places, with real keys
    const size_t need = ik.covered(pos);
    size_t have = 0;
    if (img.runs)
        for (size_t i = 0; i < img.runs->size() && (*img.runs)[i].first < pos; ++i, ++have) {
            if (have >= need || ik.start[have] != (*img.runs)[i].first) return false;
            if (ik.key[have] == "?" || ik.key[have] != img.key(i)) return false;
        }
    return have == need;
}

ConversationCache::Choice ConversationCache::choose(const std::vector<int32_t>& prompt, const ConvImages& img) const {
    Choice best;
    for (size_t i = 0; i < live_.size(); ++i) {
        const Live& l = live_[i];
        if (l.full && l.c.pos > best.len && match(l.ids, l.c.pos, l.ik, prompt, img)) {
            best.src = Choice::Live;
            best.idx = (int) i;
            best.len = l.c.pos;
        }
    }
    int64_t pl = 0;
    Choice pk;
    for (const Parked& p : parked_)
        for (size_t k = 0; k < p.ck.size(); ++k)
            if (p.ck[k].pos > pl && match(p.ids, p.ck[k].pos, p.ik, prompt, img)) {
                pl = p.ck[k].pos;
                pk.src = Choice::Park;
                pk.id = p.stamp;
                pk.ck = (int) k;
                pk.len = pl;
            }
    if (pl > best.len) return pk;
    return best;
}

int ConversationCache::free_slot() {
    for (size_t i = 0; i < live_.size(); ++i)
        if (!live_[i].full) return (int) i;
    // over N: the least recently used goes, but never the deepest of the conversation
    int64_t deep = deepest_live();
    int v = -1;
    for (size_t i = 0; i < live_.size(); ++i) {
        if (live_[i].c.pos == deep && live_.size() > 1) continue;
        if (v < 0 || live_[i].used < live_[(size_t) v].used) v = (int) i;
    }
    return v < 0 ? 0 : v;
}

bool ConversationCache::take_at(NemoModel& m, const int32_t* ids, int64_t pos, ImgKeys ik, Kind kind, std::string& err) {
    if (n_ <= 0 || pos <= 0) return true;
    int slot = -1;
    for (size_t i = 0; i < live_.size(); ++i)
        if (live_[i].full && live_[i].c.pos == pos) slot = (int) i;   // the same position again: replaced in place
    if (slot < 0) slot = free_slot();
    Live& l = live_[(size_t) slot];
    l.full = false;
    if (!m.checkpoint_save(l.c, err)) return false;
    if (l.c.pos != pos) { err = "checkpoint position mismatch"; return false; }
    l.ids.assign(ids, ids + pos);
    l.ik = std::move(ik);
    l.kind = kind;
    l.made = l.used = ++clock_;
    l.full = true;
    return true;
}

bool ConversationCache::take(NemoModel& m, const int32_t* ids, const ConvImages& img, Kind kind, std::string& err) {
    return take_at(m, ids, m.pos(), keys_for(m.pos(), img), kind, err);
}

bool ConversationCache::restore_live(NemoModel& m, int idx, std::vector<float>& logits, std::string& err) {
    Live& l = live_[(size_t) idx];
    const auto t0 = Clock::now();
    if (!m.checkpoint_restore(l.c, err)) return false;
    logits = l.c.logits;
    l.used = ++clock_;
    for (Live& o : live_)
        if (o.full && o.c.pos > l.c.pos) o.full = false;   // their K/V cells are overwritten from here
    last_restore_ms = ms_since(t0);
    return true;
}

size_t ConversationCache::arena_alloc(size_t bytes) {
    for (auto it = free_.begin(); it != free_.end(); ++it)
        if (it->second >= bytes) {
            const size_t off = it->first, len = it->second;
            free_.erase(it);
            if (len > bytes) free_[off + bytes] = len - bytes;
            return off;
        }
    return (size_t) -1;
}

void ConversationCache::arena_free(size_t off, size_t bytes) {
    auto it = free_.emplace(off, bytes).first;
    auto nx = std::next(it);
    if (nx != free_.end() && it->first + it->second == nx->first) {
        it->second += nx->second;
        free_.erase(nx);
    }
    if (it != free_.begin()) {
        auto pv = std::prev(it);
        if (pv->first + pv->second == it->first) {
            pv->second += it->second;
            free_.erase(it);
        }
    }
}

void ConversationCache::drop_parked(size_t i) {
    arena_free(parked_[i].off, parked_[i].bytes);
    parked_.erase(parked_.begin() + (long) i);
}

size_t ConversationCache::parked_bytes() const {
    size_t b = 0;
    for (const Parked& p : parked_) b += p.bytes;
    return b;
}

void ConversationCache::park_current(NemoModel& m, uint64_t keep_id) {
    if (!parking()) return;
    int deep = -1;
    for (size_t i = 0; i < live_.size(); ++i) {
        if (!live_[i].full) continue;
        if (deep < 0 || live_[i].c.pos > live_[(size_t) deep].c.pos) deep = (int) i;
    }
    if (deep < 0) return;
    // the deepest checkpoint, and the two deepest message boundaries below it (a request that retries or edits from
    // there, which the answer checkpoint cannot serve)
    std::vector<int> pick = {deep};
    for (int round = 0; round < 2; ++round) {
        int best = -1;
        for (size_t i = 0; i < live_.size(); ++i)
            if (live_[i].full && live_[i].kind == Message && std::find(pick.begin(), pick.end(), (int) i) == pick.end() &&
                (best < 0 || live_[i].c.pos > live_[(size_t) best].c.pos))
                best = (int) i;
        if (best >= 0) pick.push_back(best);
    }
    const auto t0 = Clock::now();
    const Live& D = live_[(size_t) deep];
    const int64_t cells = D.c.pos;
    // entries this park supersedes (their ids are a prefix of these, the same pictures) go first
    for (size_t i = parked_.size(); i-- > 0;) {
        const Parked& p = parked_[i];
        if (p.stamp == keep_id || p.ck.empty() || p.ck[0].pos > cells) continue;
        const int64_t pp = p.ck[0].pos;
        const size_t cov = p.ik.covered(pp);
        if (std::equal(p.ids.begin(), p.ids.begin() + pp, D.ids.begin()) && cov == D.ik.covered(pp) &&
            std::equal(p.ik.key.begin(), p.ik.key.begin() + (long) cov, D.ik.key.begin()))
            drop_parked(i);
    }
    const int nck = (int) pick.size();
    const size_t kvb = up4k((size_t) (m.kv_cell_bytes() * (uint64_t) cells));
    const size_t need = (size_t) nck * ckpt_bytes_ + kvb;
    size_t off = arena_alloc(need);
    while (off == (size_t) -1) {
        size_t v = (size_t) -1;
        for (size_t i = 0; i < parked_.size(); ++i)
            if (parked_[i].stamp != keep_id && (v == (size_t) -1 || parked_[i].stamp < parked_[v].stamp)) v = i;
        if (v == (size_t) -1) break;
        std::fprintf(stderr, "strata-nemotron: conv-cache: evicting a parked conversation of %lld tokens (%zu MiB)\n",
                     (long long) parked_[v].ck[0].pos, parked_[v].bytes >> 20);
        drop_parked(v);
        off = arena_alloc(need);
    }
    if (off == (size_t) -1) {
        std::fprintf(stderr, "strata-nemotron: conv-cache: a conversation of %lld tokens (%zu MiB) does not fit the budget\n",
                     (long long) cells, need >> 20);
        return;
    }
    Parked P;
    P.ids = D.ids;
    P.ik = D.ik;
    P.cells = cells;
    P.off = off;
    P.bytes = need;
    P.stamp = ++clock_;
    for (int j = 0; j < nck; ++j) {
        const Live& L = live_[(size_t) pick[(size_t) j]];
        NemoCkpt c = L.c;
        c.mamba = arena_ + off + (size_t) j * ckpt_bytes_;
        std::memcpy(c.mamba, L.c.mamba, m.ckpt_bytes());
        P.ck.push_back(std::move(c));
        P.kind.push_back(L.kind);
    }
    std::string err;
    if (!m.kv_save(cells, arena_ + off + (size_t) nck * ckpt_bytes_, err)) {
        std::fprintf(stderr, "strata-nemotron: conv-cache: parking failed: %s\n", err.c_str());
        arena_free(off, need);
        return;
    }
    parked_.push_back(std::move(P));
    last_park_ms = ms_since(t0);
    std::fprintf(stderr, "strata-nemotron: conv-cache: parked %lld tokens (%d checkpoints) in %.1f ms, %.0f MiB; %zu parked, "
                         "%zu of %zu MiB used\n",
                 (long long) cells, nck, last_park_ms, (double) need / (1 << 20), parked_.size(), parked_bytes() >> 20,
                 arena_cap_ >> 20);
}

bool ConversationCache::unpark(NemoModel& m, const Choice& c, std::vector<float>& logits, std::string& err) {
    size_t i = 0;
    while (i < parked_.size() && parked_[i].stamp != c.id) ++i;
    if (i == parked_.size()) { err = "parked conversation vanished"; return false; }
    Parked& p = parked_[i];
    const auto t0 = Clock::now();
    m.reset();
    clear_live();
    const size_t nck = p.ck.size();
    if (!m.kv_restore(p.cells, arena_ + p.off + nck * ckpt_bytes_, err)) return false;
    const NemoCkpt& ck = p.ck[(size_t) c.ck];
    if (!m.checkpoint_restore(ck, err)) return false;
    logits = ck.logits;
    // the restored state is a live checkpoint again (a branch before this point resumes from it)
    ImgKeys ik = p.ik;
    const size_t cov = ik.covered(ck.pos);
    ik.key.resize(cov);
    ik.start.resize(cov);
    const bool ok = take_at(m, p.ids.data(), ck.pos, std::move(ik), (Kind) p.kind[(size_t) c.ck], err);
    last_restore_ms = ms_since(t0);
    std::fprintf(stderr, "strata-nemotron: conv-cache: unparked %lld tokens (K/V %lld cells) in %.1f ms\n",
                 (long long) ck.pos, (long long) p.cells, last_restore_ms);
    drop_parked(i);
    return ok;
}

}  // namespace strata::core
