// src/core/nemotron_check_main.cpp - strata-nemotron stage 1 test: open a Nemotron-H GGUF, read its geometry and check
// every tensor (nemotron_layout.hpp), and compare the geometry with the values the HF config states.
//
// The expected values are Nemotron 3.5 Super VL's (config.json `llm_config`, checked in stage 0); a different model of
// the family passes the tensor check and fails only the expectation lines, which name what differs.
//
// Usage: strata-nemotron-check <model.gguf or any shard>
#include "strata/core/nemotron_layout.hpp"

#include <cstdio>
#include <cstring>
#include <exception>

using strata::core::NemotronBlock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    try {
        const strata::GgufModel m = strata::GgufModel::open(argv[1]);
        strata::core::NemotronGeometry g;
        std::string err;
        if (!strata::core::nemotron_geometry(m, g, err)) {
            std::fprintf(stderr, "FAIL geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("blocks %lld (+%lld MTP): %lld mamba, %lld moe, %lld attention\n  %s\n", (long long) g.n_layers,
                    (long long) g.n_nextn, (long long) g.count(NemotronBlock::Mamba), (long long) g.count(NemotronBlock::Moe),
                    (long long) g.count(NemotronBlock::Attention), g.pattern().c_str());
        std::printf("embd %lld, vocab %lld, rms eps %g\n", (long long) g.n_embd, (long long) g.n_vocab, g.rms_eps);
        std::printf("mamba: d_inner %lld = %lld heads x %lld, groups %lld, state %lld, conv %lld (conv dim %lld, in_proj %lld)\n",
                    (long long) g.ssm_d_inner, (long long) g.ssm_n_head, (long long) g.ssm_head_dim(),
                    (long long) g.ssm_n_group, (long long) g.ssm_d_state, (long long) g.ssm_d_conv,
                    (long long) g.ssm_conv_dim(), (long long) g.ssm_in_dim());
        std::printf("attention: %lld q heads, %lld kv heads, head %lld\n", (long long) g.n_head, (long long) g.n_head_kv,
                    (long long) g.head_dim);
        std::printf("moe: %lld experts, top %lld, latent %lld, ff %lld, shared ff %lld, scale %g, norm %d\n",
                    (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.moe_latent, (long long) g.n_ff_exp,
                    (long long) g.n_ff_shexp, g.expert_weights_scale, (int) g.expert_weights_norm);

        strata::core::NemotronCheck c;
        if (!strata::core::nemotron_check(m, g, c, err)) {
            std::fprintf(stderr, "FAIL tensors: %s\n", err.c_str());
            return 1;
        }
        std::printf("tensors: %lld checked, every tensor of the file accounted for;", (long long) c.tensors);
        for (auto& t : c.types) std::printf(" %s %lld", t.first.c_str(), (long long) t.second);
        const double gib = 1024.0 * 1024 * 1024;
        std::printf("\nbytes: experts %.2f GiB (%.3f MiB per expert), dense %.2f GiB, MTP %.2f GiB, embeddings %.2f GiB\n",
                    c.bytes_experts / gib, c.expert_blob_bytes / (1024.0 * 1024), c.bytes_dense / gib, c.bytes_mtp / gib,
                    c.bytes_embd / gib);

        // Nemotron 3.5 Super VL, from config.json llm_config
        struct { const char* what; long long have, want; } exp[] = {
            {"blocks", g.n_layers, 88}, {"mtp blocks", g.n_nextn, 1}, {"mamba", g.count(NemotronBlock::Mamba), 40},
            {"moe", g.count(NemotronBlock::Moe), 40}, {"attention", g.count(NemotronBlock::Attention), 8},
            {"n_embd", g.n_embd, 4096}, {"n_vocab", g.n_vocab, 131072}, {"ssm d_inner", g.ssm_d_inner, 8192},
            {"ssm heads", g.ssm_n_head, 128}, {"ssm groups", g.ssm_n_group, 8}, {"ssm state", g.ssm_d_state, 128},
            {"ssm conv", g.ssm_d_conv, 4}, {"q heads", g.n_head, 32}, {"kv heads", g.n_head_kv, 2}, {"head dim", g.head_dim, 128},
            {"experts", g.n_expert, 512}, {"top-k", g.n_expert_used, 22}, {"latent", g.moe_latent, 1024},
            {"expert ff", g.n_ff_exp, 2688}, {"shared ff", g.n_ff_shexp, 5376},
        };
        int bad = 0;
        for (auto& e : exp)
            if (e.have != e.want) {
                std::printf("MISMATCH %s: %lld, config.json says %lld\n", e.what, e.have, e.want);
                ++bad;
            }
        const char* want_pattern = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";
        if (g.pattern() != want_pattern) {
            std::printf("MISMATCH block order: %s\n", g.pattern().c_str());
            ++bad;
        }
        if (g.expert_weights_scale != 5.0 || !g.expert_weights_norm) {
            std::printf("MISMATCH routing: scale %g norm %d, config.json says 5.0 and true\n", g.expert_weights_scale,
                        (int) g.expert_weights_norm);
            ++bad;
        }
        std::printf(bad ? "FAIL: %d mismatches against config.json\n" : "PASS\n", bad);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
