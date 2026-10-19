---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- nemotron-3.5
- nemotron-h
- mamba2
- moe
- nvfp4
- gguf
- mtp
- strata
---

# Nemotron 3.5 Super (120B-A12B), text-only GGUF for one 32 GB GPU: NVFP4 experts, MTP included

**What this is.** A text-only GGUF of
[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4),
made for running on **one consumer GPU with the experts in system RAM**. The routed experts are NVIDIA's own NVFP4
weights, repacked without loss. The 1.6 GB multi-token-prediction (MTP) block is kept. The vision tower is **not**
included.

It was built for [strata-nemotron](<FORK_URL>), a Nemotron-H fork of the [Strata](https://github.com/Niko1221/Strata)
engine. The MTP-free file also loads in llama.cpp.

Files are named after their lowest-bit format, NVFP4, which holds over 90% of the bytes. Each model is split in two
with llama.cpp's `gguf-split` (Hugging Face caps a file at 50 GB). Give either engine the path of any shard:

| model | shards | size | MTP | runs in |
|---|---|---:|---|---|
| `Nemotron-3.5-Super-120B-A12B-NVFP4` | `-00001-of-00002.gguf`, `-00002-of-00002.gguf` | 71.6 GB | yes | strata-nemotron |
| `Nemotron-3.5-Super-120B-A12B-NVFP4-noMTP` | `-00001-of-00002.gguf`, `-00002-of-00002.gguf` | 69.8 GB | no | strata-nemotron, llama.cpp |

The split files hold the same tensors as the single file. strata-nemotron gives the same perplexity on both
(3.2120 on 8 × 1024 wikitext tokens).

## What is in each tensor class

The NVIDIA checkpoint is mixed-precision: NVFP4 experts, FP8 per-tensor Mamba and attention projections, BF16 for the
rest. GGUF has no FP8 type with a per-tensor scale, so those layers had to go somewhere. Each option was measured
(below), and this mix was chosen as the fastest one with acceptable quality:

| tensors | format here | size |
|---|---|---:|
| routed experts (40 MoE layers × 512, and 512 in MTP) | NVFP4, NVIDIA's values bit for bit (repacked to ggml `NVFP4`) | 59.1 GiB + MTP |
| Mamba `in_proj` / `out_proj` (FP8 in the checkpoint) | NVFP4 (our encoder, same block-scale search as ModelOpt) | |
| attention q/k/v/o, `lm_head` (FP8 in the checkpoint) | Q8_0 | |
| everything else (router, shared experts, latent fc1/fc2, norms, embeddings, Mamba conv/A/D) | BF16 / F32, as in the checkpoint | |
| dense total | | 4.98 GiB |

## Quality

`llama-perplexity`, wikitext-2 test, 16 × 1024 tokens. The baseline is the same experts with every FP8 tensor
expanded exactly to BF16, so it holds the checkpoint's own values. Only the dense FP8 layers differ between rows:

| FP8 layers stored as | dense, GiB | PPL / baseline | mean KL | top-1 agreement |
|---|---:|---:|---:|---:|
| BF16 (baseline) | 11.57 | 1 | 0 | 100% |
| Q8_0, all of them | ~6.2 | 1.0000 ± 0.0030 | 0.025 | 95.0% |
| **this file: Mamba in/out NVFP4, attention + lm_head Q8_0** | **4.98** | **1.0164 ± 0.0041** | **0.048** | **92.7%** |
| NVFP4, all of them | 4.60 | 1.0290 ± 0.0049 | 0.071 | 90.9% |

The model is sensitive to these few dense layers. They are ~6% of the weights, yet putting all of them in NVFP4 costs
about three times more KL than NVFP4 on all the experts. `lm_head` (+1.3% PPL) and Mamba `in_proj` (+0.9%) cost the
most. Even plain Q8_0 gives KL 0.025 while PPL does not move. Our reading, not tested separately: near-tied choices in
the top-22 router flip on any small perturbation.

Answers were checked end to end on 25 short prompts: facts, arithmetic, code, SQL, bash, RU/EN translation, formulas.
Greedy, through an OpenAI-compatible API. strata-nemotron and llama.cpp each got 25/25 correct; 21/25 answers were
byte-identical. Needle-in-a-haystack at 32K, 128K, 256K, 512K and 1M tokens (depths 10/50/90%): 15/15 found.

## Speed

One machine: RTX 5090 32 GB, i7-14700KF, 128 GB DDR5-5200, PCIe 5, Linux. strata-nemotron. 1 GB of VRAM is kept free
for the desktop. Greedy, 256 generated tokens on wikitext:

| context | prefill, tok/s | decode with MTP, tok/s | decode without MTP, tok/s | tokens per MTP window |
|---|---:|---:|---:|---:|
| 4K | 2 277 | 118 | 96 | 1.94 |
| 32K | 9 575 | 186 | 109 | 3.06 |
| 128K | 9 132 | 165 | 95 | 2.99 |

Prefill on a second prompt, with the expert cache warm: 16K at 12 700 tok/s, 256K at 7 440 tok/s; 1M tokens take
340 s. In chat (temperature 1.0), decode with MTP runs at 155–209 tok/s. Peak use: ~30 GB VRAM and ~105 GB RSS
(the expert file is pinned in RAM). **You need 128 GB of RAM.**

For comparison, on the same machine and the same weights (the MTP-free file), llama.cpp took 220 s for the 25
prompts, against 19 s for strata-nemotron.

## How to run

**strata-nemotron** (CUDA, Blackwell sm_120 for the FP4 prompt path). Build and serve instructions: <FORK_URL>.

```
strata-nemotron --gguf Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf --max-context 1048576 --chunk 16384 \
  --spec 5 --spec-min-p 0.8 --skip-w 0.4 --evict lfu --prompt-skip-w 0.3 --prompt-skip-max 8192 --serve
```

`serve/server.py` in the fork wraps the engine in an OpenAI-compatible API.

**llama.cpp**: use `Nemotron-3.5-Super-120B-A12B-NVFP4-noMTP-00001-of-00002.gguf` with the experts on the CPU (`-ot exps=CPU` or `--n-cpu-moe`).
llama.cpp does not load NVFP4 tensors inside the MTP block, so the MTP file does not load there.

## How it was made

`tools/nemotron/convert.py` in the fork, a wrapper around llama.cpp's converter:

```
python tools/nemotron/convert.py --fp8 nvfp4 --nvfp4-only "mixer\.(in|out)_proj" --fp8-as-q8 [--no-mtp] ...
llama-gguf-split --split --split-max-size 45G model.gguf Nemotron-3.5-Super-120B-A12B-NVFP4
```

(`llama-gguf-split` has to come from a llama.cpp build that knows the NVFP4 type, 40.)

On this checkpoint, the stock converter has to be worked around in four places. The wrapper does this without
patching llama.cpp:
1. it drops the vision tensors;
2. it renames the MTP tensors before they are packed;
3. it reads `n_routed_experts` (otherwise all 59 GiB of experts pile up in RAM);
4. it treats `quant_method: modelopt_mixed` as ModelOpt for FP8 dequantization.

## Limits

- Text only: the vision and video tower is not in these files. Unsloth's `mmproj-BF16.gguf` (projector `nemotron_omni`,
  current llama.cpp) should pair with the noMTP file for images in llama.cpp; this was not tested. strata-nemotron has
  no vision yet.
- The speed figures are from one machine. The fast prompt path (FP4 tensor cores) needs an RTX 50-series / sm_120 GPU.
- Not tested: Windows, other GPUs, other RAM sizes, downstream benchmarks beyond the checks above.
- The reasoning modes and the chat template are NVIDIA's (`chat_template.jinja` from the checkpoint).

## License

The weights are NVIDIA's and are governed by the [OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/),
the same as the base model. This repository only changes their storage format. NVIDIA's model card describes the
model's intended use, training data, bias and safety. Read it as well:
[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4).
