# strata-nemotron, stage 0: an inventory of Strata for Nemotron 3.5 Super VL

Date: 2026-10-17. Base: `main` @ a0116b27 (Strata 0.1.43.x). Model: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-{BF16,NVFP4}`.
(Russian original: `ru/STAGE0_INVENTORY.md`.)

Sources: `config.json`, the safetensors headers of both checkpoints (all shards, metadata only), `hf_quant_config.json`,
`transformers/models/nemotron_h/modeling_nemotron_h.py` (5.15.1), `third_party/llama.cpp/src/models/nemotron-h*.cpp`.
There was no code yet: this was the material for the review before stage 1.

## 1. The model: what was checked, and where the task's description was off

| parameter | value (checked) | difference from the task |
|---|---|---|
| Blocks | 88: 40 Mamba-2, 40 MoE, 8 attention. Each block is RMSNorm → one mixer → residual. There is no separate FFN after attention | — |
| Order | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME` (M = mamba, E = moe, * = attn) | irregular: take it from `layers_block_type` |
| hidden | 4096, vocab 131072, embeddings and lm_head not tied | — |
| MoE | 512 routed, top-22, 1 shared | matches |
| Router | input: the full 4096 hidden, FP32 GEMV, sigmoid. Selection by `sigmoid + e_score_correction_bias`; weights from the sigmoid without the bias, normalized to their sum, ×5.0 (`routed_scaling_factor`). n_group = 1, so the group step is degenerate | Strata has softmax top-10 |
| Latent projection | yes: `fc1_latent_proj` 4096→1024 before the experts, `fc2_latent_proj` 1024→4096 after the sum. Both BF16 | confirmed |
| Expert | **no gate**: `down(relu²(up(x)))`, up 2688×1024, down 1024×2688, in the 1024 latent | Strata has SwiGLU (gate + up + down) |
| Shared expert | relu², 4096→5376→4096, in the full hidden, the same normalized x as input, no scalar gate | Strata's shared expert has a sigmoid gate |
| Attention | 32 Q heads, 2 KV (GQA 16:1), head_dim 128, no bias, scale 1/√128, **no RoPE (NoPE)**: `apply_rotary_pos_emb` is declared but never called; llama.cpp also says "dense NoPE attention". No window | no RoPE |
| Mamba-2 | in_proj 4096→18560 = z 8192 \| xBC 10240 \| dt 128; conv1d k=4 over 10240 channels, with bias, SiLU; 128 heads × 64, n_groups 8 (B, C shared by 16 heads), d_state 128; dt = softplus(dt + dt_bias) with no clamp (`time_step_limit` 0..inf); A = −exp(A_log), a scalar per head; D skip per head; gated RMSNorm: `rms_over_groups_of_1024(y·silu(z))·w`; out_proj 8192→4096; chunk_size 128; the SSM cache in FP32 | — |
| MTP | 1 module (`num_nextn_predict_layers` = 1) of two sublayers: [0] `eh_proj(concat(enorm(emb), hnorm(h)))` 8192→4096 → norm → NoPE attention (its own KV); [1] norm → MoE 512/top-22 with latent and shared → `final_layernorm` → the shared lm_head. Check the concat order against llama.cpp in stage 6 | — |
| Context | `max_position_embeddings` 1048576 | — |

**The NVFP4 checkpoint is mixed (`MIXED_PRECISION`), not pure NVFP4:**

| tensors | format in the checkpoint | size |
|---|---|---|
| routed experts (40×512 + 512 in MTP), shared experts | NVFP4: U8 `[out, in/2]` (2×E2M1 per byte), `weight_scale` F8_E4M3 `[out, in/16]`, `weight_scale_2` F32 per tensor, plus `input_scale` F32 (a static global activation scale for W4A4) | 59.06 GiB experts (2.95 MiB per expert), 0.92 GiB shared |
| Mamba in_proj/out_proj, attention qkvo, lm_head | **FP8 E4M3 per tensor** (`weight_scale` a scalar, static `input_scale`) | 2.83 + 1.25 + 0.27 + 0.5 GiB |
| embeddings, latent fc1/fc2, router, conv1d, A/D/dt, norms, MTP attention and eh_proj | BF16 (the router bias F32) | 1 + 0.63 + 0.16 GiB … |
| KV | `kv_cache_quant_algo: FP8`; k/v scales exist only for the MTP attention | — |

The task's description of NVFP4 ("FP4 E2M1, FP8 E4M3 scale per 16, FP32 per tensor") matches. Additions:
1. `input_scale`: the vLLM/TRT-LLM reference computes W4A4;
2. the nibble order (low = even element) was to be checked in stage 1 against modelopt;
3. ggml has its own NVFP4 (`GGML_TYPE_NVFP4` = 40): a 64-value superblock of 4 UE4M3 scales + 32 bytes, with no
   global scale in the block. Repacking the bytes is lossless, and `weight_scale_2` goes into a separate tensor (as
   `conversion/base.py::_repack_nvfp4` does).

There is also an `NVFP4-LiLiCorr` variant, not in the task. A decision was needed on whether to look at it.

## 2. Memory budget (an estimate, not a measurement)

| what | size |
|---|---|
| dense weights in VRAM (the NVFP4 checkpoint as is, without embeddings) | ~6.6 GiB |
| MTP (experts 1.48 + the rest) | ~1.65 GiB |
| attention KV, BF16: 8 KiB/token (8 layers × 2 heads × 128 × K, V) | 32 MiB @4K, 256 MiB @32K, 1 GiB @128K, 8 GiB @1M |
| SSM state per sequence: 40 × 128 × 64 × 128 × FP32 | 160 MiB (+ conv 4.9 MB) |
| left for the expert cache on 32 GB | ~19 GiB ≈ 6,600 of 20,480 experts (~32%) |
| experts in RAM (NVFP4) | 59 GiB of 128 GB: no low-RAM modes needed |

Against swift15 (IQ3_XXS), by Strata's docs: 480 expert reads per token of ~1.75 MB each (~0.84 GB/token) and ~56%
of the experts in VRAM. Nemotron has 880 reads of 2.95 MiB (~2.7 GB/token) and ~32% in VRAM. Expert traffic per token
is about 3× larger, and the share in VRAM 1.7× smaller.

An estimate of the decode ceiling without MTP: at a 60% hit rate, the misses take ~1 GiB/token from DDR5 (~60 GB/s in
practice), which is ~17 ms, so ≤ 55–60 tok/s. This is **an estimate**; only a profile gives the hit rate. It follows
that stage 5 (our own ~2.5–3 bpw expert quants) matters more for speed than the task suggests.

**A limit for our own quants:** the expert's `down_proj` has an input of 2688 = 10.5 × 256. K-quants and
IQ2_XS/IQ3_XXS/IQ3_S (block 256) **cannot be used** for it. Only formats whose block divides 2688 fit: Q2_0 (64),
Q4_0/Q5_0/Q8_0/IQ4_NL/MXFP4 (32), NVFP4 (64), Q1_0 (128). Up (input 1024), shared, in_proj and out_proj have no such
limit.

## 3. Strata's optimizations

Status: **as is**, **with changes** (which), **not applicable** (why).

A general remark: Strata is a one-architecture engine. `ModelGeometry` hard-codes qwen4exp (`is_qsa_layer = l%4==3`,
`check_all` requires exactly 12 QSA layers, hc=4, PLE), and the layer composition (`layer.cpp`, `session.cpp`,
`verify.cpp`, `mtp.cpp`) is written for it. "An architecture type in the loader" in practice means a separate branch
of layer composition and graphs. What gets reused is the kernels and the subsystems around them: the expert tier, the
CPU pool, speculation, the sampler, the server.

### 3.1 Speculation and verify

| # | optimization (where) | status | what to do |
|---|---|---|---|
| 1 | MTP drafter on the GPU, its own graphs per step (`core/mtp.cpp`) | with changes | Rewrite the forward for Nemotron's MTP: eh_proj + 2 norms, NoPE GQA instead of QSA with hyper-connections, MoE top-22 with latent. Keep all 512 MTP experts in VRAM (1.48 GiB), as Strata keeps its 708 MB. The mechanics (graph capture per step, the drafter's KV, drafter prefill, chain) stay |
| 2 | Verify window: T tokens in one graph, bit-identical to greedy (`core/verify.cpp`) | with changes | A new window graph for 88 blocks. **SSM rollback:** Strata already solves it for GDN: "the window does not touch the state, `commit(n_keep)` replays the accepted steps from saved inputs". For Mamba-2, the same as option A: store xBC after the conv input and dt per position (~41 KB/pos/layer). Option B: a 160 MiB snapshot before the window, ~0.1–0.2 ms per copy. Choose by measurement, as the task says |
| 3 | Multi-token kernels in the window (multi-column MMVQ exact, T-token GDN) | with changes | A T-token Mamba step is needed (sequential over T inside the kernel, state in registers/SMEM, no writes out) |
| 4 | Speculation controller, draft policy, `--spec-min-p` (`spec/controller.cpp`, `draft_policy.cpp`) | as is | model-independent |
| 5 | Prompt lookup / suffix drafter (`spec/suffix_drafter.cpp`) | as is | — |
| 6 | Coupled drafts + Gumbel-max (`STRATA_SPEC_COUPLED/GUMBEL`) | as is | — |
| 7 | A vocabulary subset for the drafter's head (`--draft-vocab`, `tools/draft_vocab.py`) | with changes | Rebuild the subsets (en/cyrillic/CJK) for Nemotron's tokenizer (131072). The drafter's head comes from the FP8 lm_head, not from a GGUF head |
| 8 | Route-resident bias for MTP (`STRATA_ROUTE_RESIDENT_MTP`) | not applicable for now | opt-in; by Strata's measurements it did not get faster. Do not port |

### 3.2 Experts: placement, cache, CPU

| # | optimization | status | what to do |
|---|---|---|---|
| 9 | VRAM expert cache, filled from a profile, `--expert-cache auto` (`core/expert_cache.cpp`) | with changes | A blob of 2.95 MiB instead of ~1.75, 20,480 (layer, expert) pairs against 24,576, top-22. VRAM budget: SSM states and windows instead of PLE/GR |
| 10 | Expert frequency profile (`--dump-routing`, `tools/make_profile.py`, `--expert-profile-save`) | with changes | Parameterize the profile format by the number of MoE layers (40, not 48), experts and k. Take the first profile on a corpus (code/chat/ru/en), as Strata did for its own |
| 11 | Adaptive tier: RAM↔VRAM exchanges during a conversation (`--adapt-every`, `--adapt-async`, ADAPT_OVERLAP, exchange rotation, DMA batch) | as is | the logic depends only on the blob size, not the model |
| 12 | A CPU pool that computes misses "in place" (`kernels/cpu/pool.cpp`) | as is | pool and affinity (`--pool-workers`, `--aux-cpus`) |
| 13 | CPU expert kernels: IQ/Q2_0 AVX2/AVX-512, ggml-cpu dot (`kernels/cpu/*`) | with changes | An NVFP4 path with relu² and no gate is needed. The vendored ggml has `ggml_vec_dot_nvfp4_q8_0` (x86, AVX2). For the i7-14700KF (AVX2 + AVX-VNNI, **no AVX-512**), write a kernel with an E2M1→int8 LUT + VNNI after `iq_avx2`. First the ggml path, for correctness |
| 14 | Share of misses over PCIe against the CPU, auto-tuned (`--pcie-frac`, the formula L/(L+1.85R)) | as is | re-measure the 1.85 constant on this machine |
| 15 | Route tail skip: a missed expert with rank ≥ 7 of 10 for every token of the window is not computed (`STRATA_ROUTE_TAIL_SKIP`) | with changes | Re-tune the threshold for top-22 (a KL gate, as in Strata). At k=22 and scale 5.0 the tail can be heavier. Enable only after measuring KL |
| 16 | Router look-ahead, prefetching the next layer (`STRATA_LOOKAHEAD*`, Foresight slots) | with changes, low priority | Nemotron's router looks at the hidden state, and MoE blocks come after a Mamba/attention block, so the prediction is one block ahead. Foresight did not pay off in Strata's measurements |
| 17 | Low-RAM modes (mmap/resident/budget, file tier, IO prefetch, arena mmap) | not applicable | 59 GiB of experts fit in 128 GB. Do not touch or test the code |
| 18 | Peer/remote experts, a second GPU, layer split, pipeline windows | not applicable | one card |

### 3.3 Dense layers, kernels, graphs

| # | optimization | status | what to do |
|---|---|---|---|
| 19 | Native MMVQ for dense projections (ggml types, multi-column) (`cuda/native_mmvq.cu`) | with changes | ggml has no FP8 E4M3 per tensor. Options in 4.2. BF16 GEMV (`bf16_gemv.cu`) as is for the latent and the router |
| 20 | Router top-10 softmax (`cuda/router_top10.cu`, `native_router.cu`) | with changes | A new kernel: FP32 GEMV 512×4096, sigmoid + bias, top-22, normalization, ×5. The window (`moe_route_window`) after the same pattern |
| 21 | Shared expert (`cuda/shared_expert.cu`), "shared early": the shared expert runs while the host drives the CPU pool | with changes | relu², no gate, no scalar sigmoid, NVFP4. The "shared early" idea carries over as a whole |
| 22 | Grouped hit kernels for VRAM experts (`s2_expert_grouped.cu`, `native_moe.cu`) | with changes | relu², two GEMVs instead of three, input 1024 (latent), NVFP4 W4A16 (decode) |
| 23 | Doorbell, MoE split into pre/post graphs around the CPU pool | as is | `pre[l] = norm→route→publish→shared`, `post[l] = combine→fc2_latent→residual`. Only fc1_latent on the GPU in between |
| 24 | CUDA graphs per layer and per window (`core/graph.cpp`, `session.cpp`) | with changes | capture new graphs; the mechanism is the same |
| 25 | PDL (programmatic dependent launch) | as is | — |
| 26 | Thread-block clusters for selection/argmax on sm_90+ (`STRATA_ARGMAX_MULTI`) | as is | argmax/sampler over vocab 131072. The QSA part does not apply |
| 27 | GDN kernels: conv step, fused preprocess, recurrent step, chunked WY prefill (`fused_gdn.cu`, `native_gdn*.cu`, `gdn_rec_parity.cu`) | not applicable as kernels | A different recurrence. Take the structure (conv ring, fused preprocess, `GdnBuffers`, zero_state) as a template for `MambaBuffers`. For prefill, the vendored llama.cpp has a ready chunked SSD (`ggml-cuda/ssm-scan.cu` + `ssm-conv.cu`) to start from |
| 28 | QSA: sparse attention with an indexer, block selection, decode/prompt attention (`qsa*.cu`, `native_qsa*.cu`) | not applicable | Nemotron has dense GQA. A dense flash-decode is needed for head_dim 128, GQA 16, NoPE. Candidates: `native_flash_attn.cu` (diagnostic now, ≤ 256) or fattn from the vendored ggml for prefill |
| 29 | RoPE/mRoPE/YaRN, rope scaling (`rope*.cu`, `rope_scaling.hpp`) | not applicable | NoPE |
| 30 | Gated residual / hyper-connections hc=4 (`gr.cu`, `fused_gr.cu`, `native_gr_*`) | not applicable | Nemotron has a plain residual |
| 31 | PLE: a 28.8 GB n-gram table on SSD (`ngram/`, `ple*.cu`, `--ple-io`) | not applicable | Nemotron has no per-layer embeddings |
| 32 | Head (`core/native_head.cpp`), the drafter's Q4 head | with changes | lm_head FP8 → our own format (see 4.2); the drafter's head = a subset of rows |
| 33 | Sampler, logit bias, greedy argmax (`cuda/sampler.cu`, `logit_bias`) | as is | — |
| 34 | Control vector / "experimental speed projection" (`cvec.cu`, `--cvec-mode project`) | with changes, later | Out of scope for this task |

### 3.4 KV, prefill, conversation cache

| # | optimization | status | what to do |
|---|---|---|---|
| 35 | KV int8 / q4 with Hadamard / k8v4 (`kv_q8.cu`, `kv_q4.cu`) | with changes | Only 8 attention layers, head_dim 128 instead of 256. The KV is 1 GiB @128K anyway; int8 gives ~500 MiB more for experts. BF16 first, int8 later by KL |
| 36 | KV streaming from RAM from 64K (`--kv-resident`, `kv_stream.cu`), `--kv-grow` | not applicable below 128K | at 8 KiB/token the gain is ≤ 1 GiB. Revisit for 1M |
| 37 | Chunked prefill up to 8192 (32768 opt-in), `--prefill auto`, borrowing cache slots for buffers (`prefill/prefill.cpp`) | with changes | Mamba blocks through a chunked SSD (27), attention blocks through dense flash, MoE as is |
| 38 | Streaming the next layer's experts over PCIe during the current attention, a ring in bytes (`STRATA_RING_BYTES`) | as is | a different blob size, the same logic. The Mamba block can be used for overlap: it costs more than attention |
| 39 | ggml MMQ for experts in prefill, fused int8 TC (`moe_mmq.cu`, `moe_fused*.cu`) | with changes | The vendored ggml already has NVFP4 MMQ with **native FP4 MMA on Blackwell** (`mma.sync kind::mxf4nvf4 … ue4m3`, `mmq-config-blackwell.cuh`, `mmq-instance-nvfp4.cu`). Needs relu² without a gate and the per-tensor scale. This is the basis of "the GEMM for prefill" in stage 5 |
| 40 | CPU share in short prefill chunks (`STRATA_PREFILL_CPU_SHARE`, by default < 1024) | with changes | needs the same NVFP4 CPU path (13) |
| 41 | Prefix cache, conversation cache, message boundary, parking, snapshots (`core/conversation_*`) | with changes | The conversation state = the KV of 8 layers + **SSM 160 MiB + conv** per resume point. Strata already does this for GDN. The image formats and size checks change |
| 42 | Batching (`--batch`, batch groups) | with changes, later | An SSM state per slot. Not needed for one user |

### 3.5 Loading, platform, server

| # | optimization | status | what to do |
|---|---|---|---|
| 43 | GGUF reader + "pack" (`artifact/gguf_reader`, `tools/iq_pack.py`, `native_experts.txt`) | with changes | see 4.1 |
| 44 | Read-ahead at start, pinned arena, cudaHostRegister in slices | as is | — |
| 45 | Watchdog, stall report, decode timing / GPU stage profiles | as is | rename the GPU stages for the new blocks |
| 46 | OpenAI/Anthropic server, web app, MCP (`serve/`) | with changes | Nemotron's chat template, its reasoning parser (`ultra_v3_reasoning_parser.py`), the tool-call format |
| 47 | HIP / SYCL / Intel / Strix Halo / older GPUs | not applicable | CUDA sm_120 only. New files under `if CUDA`; do not break the HIP build |
| 48 | Vision encoder (`strata-vision`, mmproj) | not applicable now | RADIO ViT-H, later per the task |
| 49 | Parity tests after `*_parity.cpp` + `ref/*.py` | as is | the same style for `mamba_parity`, `nemotron_moe_parity`, `attn_nope_parity` |

## 4. Decisions for the review

1. **The loading path.** Proposed: NVFP4 safetensors → GGUF with the vendored llama.cpp's converter (it already
   handles `nemotron_h` with the latent, MTP and repacking modelopt NVFP4 into `GGML_TYPE_NVFP4`) → a Strata pack.
   - Pros: `gguf_reader`, the pack, `native_experts.txt` and the ggml kernels work without a new loader. llama.cpp on
     the same file gives **a runnable end-to-end reference** for stage 7: full BF16 does not fit, this does.
   - Con: it dequantizes FP8 (point 2).
   - Alternative: our own safetensors loader. More code, full control.
   - BF16 for our own quants goes through the same converter anyway, with `--outtype` and an imatrix.
2. **The FP8 tensors** (in_proj, out_proj, qkvo, lm_head, 4.85 GiB in all):
   - (a) BF16: exact, but +4.85 GiB of VRAM, which is ~1,650 fewer experts;
   - (b) Q8_0 (`fp8_as_q8`): nearly lossless, FP8's size + 6%, and Strata's kernels exist;
   - (c) our own FP8 per-tensor type in the pack + FP8 GEMV/MMA on sm_120: bit-identical to the checkpoint, but a new
     kernel.

   Proposed: (b), with KL measured against (a) on one layer.
3. **The NVFP4 reference.** vLLM computes W4A4: the activations also go to NVFP4 through `input_scale`. A decode GEMV
   is more natural as W4A16: more accurate than the reference, but not bit-identical. Proposed: take a W4A16 dequant
   in PyTorch (what the model "means") as the logit reference, and measure W4A4 as a deviation. FP4 MMA in prefill is
   W4A4 by necessity.
4. **SSM rollback in MTP.** Option A (replay from saved inputs, as GDN in Strata) first, option B (a snapshot) as an
   A/B.
5. **Workspace.** A `strata-nemotron` branch in its own worktree with its own build. The production Strata checkout
   (the swift15 / Flash-Next slots) is not touched, and its local changes do not go into the branch.
6. **Downloads.**
   - NVFP4 (~70 GiB) right away, for stage 1.
   - BF16 (~232 GiB, 64 shards) is needed for the per-layer references and our own quants. Proposed: download shards
     on demand, since a per-layer test needs only the shard with its layer. The whole thing only for stage 5.
   - 1.5 TB of disk free.

## 5. What will not carry over (preliminary)

QSA (sparse attention with an indexer), RoPE/YaRN, hyper-connections/GR, the PLE table on SSD, low-RAM modes,
multi-GPU/peer, HIP/SYCL. All for one reason: Nemotron does not have them, or one machine with 128 GB does not need
them (see the table). Also IQ2_XS/IQ3_XXS/IQ3_S/K-quants for the experts' `down_proj`, because of the 2688 row length.

## 6. Review outcome (2026-10-17) and the checks after it

Decisions:
1. Loading through llama.cpp's converter → GGUF → pack.
2. Quantize the FP8 tensors (in/out_proj, qkvo, lm_head) to **NVFP4**: −2.2 GiB of VRAM against FP8, ≈ +750 experts
   in the cache. Measure KL against BF16 for these tensors. (Stage 1 later settled on a mix: see STAGE1_LOAD.md.)
3. The logit reference: NVFP4 weights × BF16 activations; W4A4 as a separate deviation.
4. NVFP4 downloaded right away, BF16 by shard; also look at IQ3_XXS.
5. Look at LiLiCorr.

**LiLiCorr is not a weight variant but a separate drafter** (`LiLiCorrDraftModel`, vLLM `method dflash`):
- a block DFlash drafter: 8 tokens in parallel, non-causal, 5 dense GQA 32/8 layers with SWA 8192;
- its input is the target model's hidden state from layers 1/22/43/64/85;
- plus a two-layer correlation head over the top-8 candidates;
- 1.13B parameters + an NVFP4 copy of lm_head (0.54B).

NVIDIA reports an acceptance length of 4.74 (T=0) / 4.36 (T=1) with 7 drafted tokens on SPEED-Bench, and recommends
it for DGX Spark (which is also memory-bandwidth-bound). For us it is a stage-6 candidate next to MTP. The risk: an
8-token window raises the union of missed experts. By Strata's measurements that is already 3.05× a single token at
T=4, and DFlash2 measured −15..−35% in another of our engines. Ready GGUFs: `ggml-org/...-GGUF` and
`unsloth/...-GGUF/LiLiCorr/` (1.2–1.3 GiB).

**IQ3_XXS and ready-made GGUFs.** Ready builds exist, and all of them solve the 2688 problem the same way, with
`down_proj` in 32/64-block types:

| build | size | up_exps | down_exps | dense | KL to BF16* |
|---|---:|---|---|---|---:|
| unsloth/lmstudio NVFP4 | 75–78 GB | NVFP4 | NVFP4 | — | 0.0253–0.0256 |
| AtomicChat AD-IQ4_NL | 74.8 GB | — | IQ4_NL | Q8_0/Q6_K | 0.0133 |
| AtomicChat AD-IQ3_XXS | 65.3 GB | IQ3_XXS (34), IQ4_XS (6) | IQ4_NL / Q5_0 | Q8_0/Q6_K | 0.0352 |
| unsloth UD-IQ3_XXS | 59.8 GB | IQ3_XXS (25), IQ2_XS/IQ2_S/IQ4_XS | IQ4_NL (29), Q2_0 (11) | Q6_K/Q8_0 | 0.0670 |
| AtomicChat AD-Q2_0_L | 55.8 GB | — | Q2_0 on "quiet" blocks | Q8_0/Q6_K | 0.0722 |
| AtomicChat AD-Q2_0_S | 45.7 GB | IQ2_S (34) | Q2_0 (34) | Q6_K/Q8_0 | 0.1470 |

\* AtomicChat's measurements (the repository README, one setup, logs in `AtomicChat/...-GGUF-metrics`), not ours.

Strata already has all of these types: GSQ-RCO packs with IQ3_XXS/IQ2_S/IQ4_XS gate/up and IQ4_NL/Q2_0 down, AVX2 CPU
kernels for IQ*, ggml-cpu for the rest. So our own quantization from BF16 in stage 5 can start from the ready files.

Expert size, and how many fit in ~19 GiB:

| experts | per expert | fit | share |
|---|---:|---:|---:|
| NVFP4 | 2.95 MiB | ~6,600 | 32% |
| IQ3_XXS + IQ4_NL | 2.48 MiB | ~7,800 | 38% |
| IQ3_XXS + Q2_0 | 1.74 MiB | ~11,200 | 55% |
| IQ2_S + Q2_0 | 1.58 MiB | ~12,300 | 60% |

An observation: AD-IQ4_NL is the same size as NVFP4 with half the KL. So the "native" NVFP4 is not the best 4.5-bit
option for quality. Its advantage is FP4 tensor cores in prefill.

The vendored llama.cpp's converter registers only `NemotronHForCausalLM`. The VL wrapper (`NemotronH_Omni_Reasoning_V3`,
`llm_config`, the `language_model.` prefix) has to be fed through a shim directory (config = `llm_config`, symlinked
shards, vision tensors skipped). To be checked in stage 1.
