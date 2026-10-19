// src/core/nemotron_layout.cpp - see include/strata/core/nemotron_layout.hpp.
#include "strata/core/nemotron_layout.hpp"

#include <cstdio>
#include <map>
#include <set>

namespace strata::core {
namespace {

const char* ARCH = "nemotron_h_moe";

bool key_u(const strata::GgufFile& f, const std::string& k, int64_t& v, std::string& err) {
    const strata::MetaValue* m = f.get(std::string(ARCH) + "." + k);
    if (!m || !m->is_num()) {
        err = "GGUF: " + std::string(ARCH) + "." + k + " is missing";
        return false;
    }
    v = (int64_t) m->num();
    return true;
}

bool key_f(const strata::GgufFile& f, const std::string& k, double& v, std::string& err) {
    const strata::MetaValue* m = f.get(std::string(ARCH) + "." + k);
    if (!m || !m->is_num()) {
        err = "GGUF: " + std::string(ARCH) + "." + k + " is missing";
        return false;
    }
    v = m->num();
    return true;
}

/// A per-layer key: an array of `n` numbers, or one number for every layer.
bool key_arr(const strata::GgufFile& f, const std::string& k, int64_t n, std::vector<int64_t>& v, std::string& err) {
    const strata::MetaValue* m = f.get(std::string(ARCH) + "." + k);
    if (!m) {
        err = "GGUF: " + std::string(ARCH) + "." + k + " is missing";
        return false;
    }
    v.assign((size_t) n, 0);
    if (m->is_num()) {
        for (auto& x : v) x = (int64_t) m->num();
        return true;
    }
    if ((int64_t) m->items.size() < n) {
        err = "GGUF: " + std::string(ARCH) + "." + k + " has " + std::to_string(m->items.size()) + " entries, " +
              std::to_string(n) + " blocks need one each";
        return false;
    }
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (int64_t) m->items[(size_t) i].num();
    return true;
}

struct Want {
    std::string name;
    std::vector<uint64_t> shape;
    std::set<uint32_t> types;   ///< empty: any type the reader knows
    bool mtp = false, expert = false, embd = false;
};

// the type sets: what the engine reads for each role
const std::set<uint32_t> T_F32 = {0};
const std::set<uint32_t> T_NORM = {0, 30, 1};                // F32 (the converter's), BF16, F16
const std::set<uint32_t> T_ANY;                             // a quantized matrix: any type the reader sizes

std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

}  // namespace

int64_t NemotronGeometry::count(NemotronBlock k) const {
    int64_t n = 0;
    for (auto b : blocks) n += b == k;
    return n;
}

std::string NemotronGeometry::pattern() const {
    std::string s;
    for (auto b : blocks) s += (char) b;
    return s;
}

bool nemotron_geometry(const strata::GgufModel& m, NemotronGeometry& g, std::string& err) {
    const strata::GgufFile& f = m.meta();
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || arch->s != ARCH) {
        err = "GGUF: architecture is '" + (arch ? arch->s : std::string("?")) + "', not " + ARCH;
        return false;
    }
    g = NemotronGeometry{};
    int64_t block_count = 0, norm = 0;
    if (!key_u(f, "block_count", block_count, err) || !key_u(f, "embedding_length", g.n_embd, err) ||
        !key_u(f, "ssm.inner_size", g.ssm_d_inner, err) || !key_u(f, "ssm.state_size", g.ssm_d_state, err) ||
        !key_u(f, "ssm.group_count", g.ssm_n_group, err) || !key_u(f, "ssm.conv_kernel", g.ssm_d_conv, err) ||
        !key_u(f, "ssm.time_step_rank", g.ssm_n_head, err) || !key_u(f, "expert_count", g.n_expert, err) ||
        !key_u(f, "expert_used_count", g.n_expert_used, err) ||
        !key_u(f, "expert_feed_forward_length", g.n_ff_exp, err) ||
        !key_u(f, "expert_shared_feed_forward_length", g.n_ff_shexp, err) ||
        !key_u(f, "moe_latent_size", g.moe_latent, err) || !key_f(f, "expert_weights_scale", g.expert_weights_scale, err) ||
        !key_u(f, "expert_weights_norm", norm, err) || !key_u(f, "attention.key_length", g.head_dim, err))
        return false;
    // the converter writes the RMS norms' epsilon as attention.layer_norm_epsilon (llama.cpp reads either)
    std::string ignore;
    if (!key_f(f, "attention.layer_norm_rms_epsilon", g.rms_eps, ignore) &&
        !key_f(f, "attention.layer_norm_epsilon", g.rms_eps, err))
        return false;
    g.expert_weights_norm = norm != 0;
    if (!key_u(f, "nextn_predict_layers", g.n_nextn, ignore)) g.n_nextn = 0;
    g.n_layers = block_count - g.n_nextn;

    std::vector<int64_t> heads, kv, ff;
    if (!key_arr(f, "attention.head_count", block_count, heads, err) ||
        !key_arr(f, "attention.head_count_kv", block_count, kv, err) ||
        !key_arr(f, "feed_forward_length", block_count, ff, err))
        return false;
    for (int64_t i = 0; i < g.n_layers; ++i) {
        const bool attn = kv[(size_t) i] > 0, moe = ff[(size_t) i] > 0;
        if (attn && moe) {
            err = "GGUF: block " + std::to_string(i) + " has both attention heads and a feed-forward length";
            return false;
        }
        g.blocks.push_back(attn ? NemotronBlock::Attention : moe ? NemotronBlock::Moe : NemotronBlock::Mamba);
        if (attn) {
            if (g.n_head && (heads[(size_t) i] != g.n_head || kv[(size_t) i] != g.n_head_kv)) {
                err = "GGUF: attention blocks disagree on their head counts";
                return false;
            }
            g.n_head = heads[(size_t) i];
            g.n_head_kv = kv[(size_t) i];
        }
    }
    const strata::TensorInfo* emb = m.find("token_embd.weight");
    if (!emb || emb->shape.size() != 2 || (int64_t) emb->shape[0] != g.n_embd) {
        err = "GGUF: token_embd.weight is missing or not [n_embd, n_vocab]";
        return false;
    }
    g.n_vocab = (int64_t) emb->shape[1];
    if (g.ssm_n_head <= 0 || g.ssm_d_inner % g.ssm_n_head || g.ssm_n_head % g.ssm_n_group || !g.n_head ||
        g.n_head % g.n_head_kv) {
        err = "GGUF: inconsistent Mamba or attention geometry";
        return false;
    }
    return true;
}

bool nemotron_check(const strata::GgufModel& m, const NemotronGeometry& g, NemotronCheck& out, std::string& err) {
    out = NemotronCheck{};
    const uint64_t E = (uint64_t) g.n_embd, V = (uint64_t) g.n_vocab, NE = (uint64_t) g.n_expert;
    const uint64_t DI = (uint64_t) g.ssm_d_inner, CD = (uint64_t) g.ssm_conv_dim(), H = (uint64_t) g.ssm_n_head;
    const uint64_t L = (uint64_t) g.moe_latent, FF = (uint64_t) g.n_ff_exp, SH = (uint64_t) g.n_ff_shexp;
    const uint64_t QD = (uint64_t) (g.n_head * g.head_dim), KD = (uint64_t) (g.n_head_kv * g.head_dim);

    std::vector<Want> want;
    want.push_back({"token_embd.weight", {E, V}, T_ANY, false, false, true});
    want.push_back({"output_norm.weight", {E}, T_NORM});
    want.push_back({"output.weight", {E, V}, T_ANY});
    auto blk = [](int64_t i, const char* s) { return "blk." + std::to_string(i) + "." + s; };
    auto add_attn = [&](int64_t i, bool mtp) {
        want.push_back({blk(i, "attn_q.weight"), {E, QD}, T_ANY, mtp});
        want.push_back({blk(i, "attn_k.weight"), {E, KD}, T_ANY, mtp});
        want.push_back({blk(i, "attn_v.weight"), {E, KD}, T_ANY, mtp});
        want.push_back({blk(i, "attn_output.weight"), {QD, E}, T_ANY, mtp});
    };
    auto add_moe = [&](int64_t i, bool mtp) {
        want.push_back({blk(i, "ffn_gate_inp.weight"), {E, NE}, T_ANY, mtp});
        want.push_back({blk(i, "exp_probs_b.bias"), {NE}, T_F32, mtp});
        want.push_back({blk(i, "ffn_latent_down.weight"), {E, L}, T_ANY, mtp});
        want.push_back({blk(i, "ffn_latent_up.weight"), {L, E}, T_ANY, mtp});
        want.push_back({blk(i, "ffn_up_exps.weight"), {L, FF, NE}, T_ANY, mtp, true});
        want.push_back({blk(i, "ffn_down_exps.weight"), {FF, L, NE}, T_ANY, mtp, true});
        want.push_back({blk(i, "ffn_up_shexp.weight"), {E, SH}, T_ANY, mtp});
        want.push_back({blk(i, "ffn_down_shexp.weight"), {SH, E}, T_ANY, mtp});
    };
    for (int64_t i = 0; i < g.n_layers; ++i) {
        want.push_back({blk(i, "attn_norm.weight"), {E}, T_NORM});
        switch (g.blocks[(size_t) i]) {
        case NemotronBlock::Mamba:
            want.push_back({blk(i, "ssm_in.weight"), {E, DI + CD + H}, T_ANY});
            want.push_back({blk(i, "ssm_conv1d.weight"), {(uint64_t) g.ssm_d_conv, CD}, T_NORM});
            want.push_back({blk(i, "ssm_conv1d.bias"), {CD}, T_NORM});
            want.push_back({blk(i, "ssm_dt.bias"), {H}, T_NORM});
            want.push_back({blk(i, "ssm_a"), {1, H}, T_F32});
            want.push_back({blk(i, "ssm_d"), {1, H}, T_NORM});
            want.push_back({blk(i, "ssm_norm.weight"), {DI / (uint64_t) g.ssm_n_group, (uint64_t) g.ssm_n_group}, T_NORM});
            want.push_back({blk(i, "ssm_out.weight"), {DI, E}, T_ANY});
            break;
        case NemotronBlock::Attention: add_attn(i, false); break;
        case NemotronBlock::Moe: add_moe(i, false); break;
        }
    }
    for (int64_t i = g.n_layers; i < g.n_layers + g.n_nextn; ++i) {
        want.push_back({blk(i, "nextn.enorm.weight"), {E}, T_NORM, true});
        want.push_back({blk(i, "nextn.hnorm.weight"), {E}, T_NORM, true});
        want.push_back({blk(i, "nextn.eh_proj.weight"), {2 * E, E}, T_ANY, true});
        want.push_back({blk(i, "nextn.shared_head_norm.weight"), {E}, T_NORM, true});
        want.push_back({blk(i, "attn_norm.weight"), {E}, T_NORM, true});
        want.push_back({blk(i, "post_attention_norm.weight"), {E}, T_NORM, true});
        add_attn(i, true);
        add_moe(i, true);
    }

    std::set<std::string> seen;
    std::map<std::string, int64_t> types;
    uint64_t blob = 0;
    bool blob_same = true;
    auto account = [&](const strata::TensorInfo* t, size_t shard, const Want& w) -> bool {
        if (!m.in_bounds(*t, shard)) {
            err = "GGUF: " + t->name + " (" + t->type_name() + ") lies outside its file or has an unknown type";
            return false;
        }
        const uint64_t b = strata::tensor_payload_bytes(*t);
        if (w.embd) out.bytes_embd += b;
        else if (w.mtp) out.bytes_mtp += b;
        else if (w.expert) out.bytes_experts += b;
        else out.bytes_dense += b;
        types[t->type_name()] += 1;
        seen.insert(t->name);
        ++out.tensors;
        return true;
    };
    for (const Want& w : want) {
        size_t shard = 0;
        const strata::TensorInfo* t = m.find(w.name, &shard);
        if (!t) {
            err = "GGUF: " + w.name + " is missing";
            return false;
        }
        if (t->shape != w.shape) {
            err = "GGUF: " + w.name + " is " + shape_str(t->shape) + ", the geometry needs " + shape_str(w.shape);
            return false;
        }
        if (!w.types.empty() && !w.types.count(t->type)) {
            err = "GGUF: " + w.name + " is " + t->type_name() + ", not a type this role takes";
            return false;
        }
        if (!account(t, shard, w)) return false;
        // an NVFP4 matrix carries its per-tensor scale beside it: `<name without .weight>.scale`
        const size_t dot = w.name.rfind(".weight");
        const std::string stem = dot != std::string::npos && dot + 7 == w.name.size() ? w.name.substr(0, dot) : w.name;
        for (const char* side : {".scale", ".input_scale"}) {
            size_t s2 = 0;
            const strata::TensorInfo* st = m.find(stem + side, &s2);
            if (!st) {
                if (t->type == 40 && std::string(side) == ".scale") {
                    err = "GGUF: " + w.name + " is NVFP4 but has no " + stem + ".scale (its per-tensor scale)";
                    return false;
                }
                continue;
            }
            const uint64_t n = st->elements();
            if (st->type != 0 || (n != 1 && !(w.expert && n == NE))) {
                err = "GGUF: " + st->name + " is " + st->type_name() + " " + shape_str(st->shape) +
                      ", a scale is F32 [1] (or [n_expert] for the experts)";
                return false;
            }
            Want ws = w;
            if (!account(st, s2, ws)) return false;
        }
    }
    // one expert's blob: up + down slice of a backbone layer, the same in every layer when the types are
    for (int64_t i = 0; i < g.n_layers; ++i) {
        if (g.blocks[(size_t) i] != NemotronBlock::Moe) continue;
        const strata::TensorInfo* up = m.find(blk(i, "ffn_up_exps.weight"));
        const strata::TensorInfo* dn = m.find(blk(i, "ffn_down_exps.weight"));
        const uint64_t b = (strata::tensor_payload_bytes(*up) + strata::tensor_payload_bytes(*dn)) / NE;
        if (blob && b != blob) blob_same = false;
        blob = b;
    }
    out.expert_blob_bytes = blob_same ? blob : 0;
    for (size_t s = 0; s < m.size(); ++s)
        for (const auto& t : m.shard(s).tensors())
            if (!seen.count(t.name)) {
                err = "GGUF: " + t.name + " (" + t.type_name() + " " + shape_str(t.shape) +
                      ") is in the file but not in the geometry";
                return false;
            }
    for (auto& kv : types) out.types.emplace_back(kv.first, kv.second);
    return true;
}

}  // namespace strata::core
