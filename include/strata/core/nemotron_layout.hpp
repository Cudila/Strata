// include/strata/core/nemotron_layout.hpp - strata-nemotron stage 1: the Nemotron-H (LatentMoE) model's geometry and
// the check of every tensor of its GGUF.
//
// Strata's `ModelGeometry` (layout.hpp) is qwen4exp's and stays as it is: this is a second geometry, chosen by
// `general.architecture == "nemotron_h_moe"`, and nothing on the qwen4exp path reads it.
//
// THE MODEL, as the GGUF of tools/nemotron/convert.py stores it (ggml order: shape[0] varies fastest):
//
//   88 blocks, each   x += mixer(rms_norm(x) * attn_norm)   with ONE mixer per block:
//     mamba      ssm_in [4096, 18560] -> z 8192 | xBC 10240 | dt 128;  ssm_conv1d [4, 10240] (+bias);  ssm_dt.bias,
//                ssm_a (= -exp(A_log)) and ssm_d [1, 128];  ssm_norm [1024, 8] (8 groups);  ssm_out [8192, 4096]
//     attention  attn_q [4096, 4096], attn_k / attn_v [4096, 256], attn_output [4096, 4096]; GQA 32/2, head 128, no RoPE
//     moe        ffn_gate_inp [4096, 512] (+ exp_probs_b.bias, the selection bias);  ffn_latent_down [4096, 1024];
//                ffn_up_exps [1024, 2688, 512], ffn_down_exps [2688, 1024, 512] (relu^2, no gate);
//                ffn_latent_up [1024, 4096];  shared ffn_up_shexp [4096, 5376], ffn_down_shexp [5376, 4096]
//   block 88 is the MTP head: nextn.{enorm, hnorm, eh_proj [8192, 4096], shared_head_norm}, an attention sub-layer
//   (attn_norm, attn_q/k/v/output) and a moe sub-layer (post_attention_norm, the moe set).
//
//   An NVFP4 tensor carries its per-tensor scale as `<name>.scale` ([1], or [n_expert] for the experts: one per
//   expert) and the checkpoint's static activation scale as `<name>.input_scale` (W4A4 serving; unused for W4A16).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

enum class NemotronBlock : char { Mamba = 'M', Moe = 'E', Attention = '*' };

struct NemotronGeometry {
    int64_t n_embd = 0, n_vocab = 0, n_layers = 0;    ///< n_layers = the backbone's blocks, MTP not counted
    int64_t n_nextn = 0;                               ///< MTP blocks after the backbone (block n_layers)
    std::vector<NemotronBlock> blocks;                 ///< n_layers entries
    double rms_eps = 0;

    // Mamba-2
    int64_t ssm_d_inner = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_d_conv = 0, ssm_n_head = 0;
    int64_t ssm_head_dim() const { return ssm_n_head ? ssm_d_inner / ssm_n_head : 0; }
    int64_t ssm_conv_dim() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }
    int64_t ssm_in_dim() const { return ssm_d_inner + ssm_conv_dim() + ssm_n_head; }

    // attention
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0;

    // MoE
    int64_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0, moe_latent = 0;
    double expert_weights_scale = 0;
    bool expert_weights_norm = false;

    int64_t count(NemotronBlock k) const;
    std::string pattern() const;                       ///< "MEMEMEM*E..." like the config's layers_block_type
};

/// Reads the geometry from the model's metadata and its per-layer arrays (a block is attention where
/// attention.head_count_kv[i] > 0, moe where feed_forward_length[i] > 0, mamba otherwise - llama.cpp's rule).
/// False and `err` when the architecture is not nemotron_h_moe or a key is missing or inconsistent.
bool nemotron_geometry(const strata::GgufModel& m, NemotronGeometry& g, std::string& err);

/// What `nemotron_check` found, for the report and the tests.
struct NemotronCheck {
    int64_t tensors = 0;                       ///< tensors checked (every one in the file is accounted for)
    std::vector<std::pair<std::string, int64_t>> types;   ///< ggml type name -> tensors
    uint64_t bytes_experts = 0, bytes_dense = 0, bytes_mtp = 0, bytes_embd = 0;
    uint64_t expert_blob_bytes = 0;            ///< one expert's up + down rows (backbone layers; 0 if they differ)
};

/// Every tensor the geometry implies: present, of a type the engine can read, with the exact shape, inside its file,
/// NVFP4 tensors with their `.scale`; and no tensor in the file that the geometry does not account for.
/// Returns false with the first problem in `err`.
bool nemotron_check(const strata::GgufModel& m, const NemotronGeometry& g, NemotronCheck& out, std::string& err);

}  // namespace strata::core
