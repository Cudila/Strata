---
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- nemotron-3.5
- nemotron-h
- mamba2
- moe
- nvfp4
- mtp
- strata
- gguf
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
---

# Nemotron-3.5-Super-120B-A12B — NVFP4 + Strata engine for one RTX 50-series card

### **186 t/s decode and 9,575 t/s prefill at 32K** on one RTX 5090 with the experts in RAM: 6× llama.cpp's decode on the same weights, with NVIDIA's own NVFP4 experts and the MTP head kept.

> **Blackwell only: RTX 50-series (sm_120).** Every optimization of the engine is built on Blackwell's FP4 tensor-core
> instructions (`mma … kind::mxf4nvf4.block_scale`, NVFP4 × NVFP4 with its native scales), which no earlier
> generation has. The engine is compiled for `sm_120a` and does not run on RTX 40/30-series or on AMD.
>
> **Vision works.** Images go through strata-vision with Unsloth's
> [`mmproj-BF16.gguf`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) and a patched
> llama.cpp (see *Images* below). Its embeddings match NVIDIA's own code to a per-token cosine of ≥ 0.998. Video is
> not done yet.
>
> **16 GB VRAM + 64 GB RAM: in development.** Cold experts will be streamed from an NVMe SSD (VRAM cache → RAM cache →
> SSD), so the model can run on a 16 GB RTX 50-series card in a 64 GB machine. Today it needs 32 GB of VRAM and 128 GB
> of RAM (see *Hardware configurations*).

This repo has a GGUF of the text model of
[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4),
made for [strata-nemotron](https://github.com/Cudila/Strata). That is a fork of the
[Strata](https://github.com/Niko1221/Strata) engine that runs this model on one consumer card: 63 GB of experts in
pinned RAM, a VRAM cache in front of them, and every NVFP4 GEMM of the prompt on Blackwell's FP4 tensor cores.

The routed experts are NVIDIA's NVFP4 values, bit for bit, repacked to ggml's `NVFP4` without loss. The 1.6 GB
multi-token-prediction (MTP) block is kept: it is what takes decode from ~100 to ~186 t/s. The vision tower is not
in this file: it is Unsloth's mmproj (see *Images*).

## Performance

| Metric | strata-nemotron, MTP on | strata-nemotron, MTP off | llama.cpp, same weights* |
| :----- | :---------------------: | :----------------------: | :----------------------: |
| **Decode, 4K prompt** | **118 t/s** | 96 t/s | 19.4 t/s |
| **Decode, 32K prompt** | **186 t/s** | 109 t/s | — |
| **Decode, 128K prompt** | **165 t/s** | 95 t/s | — |
| **Decode, 256K prompt** | 138 t/s | 79 t/s | — |
| **Decode, 512K prompt** | 107 t/s | 65 t/s | — |
| **Decode, 1M prompt** | **111 t/s** | 47 t/s | — |
| **Prefill, 4K** | 2,277 t/s | 2,447 t/s | ~30–40 t/s (short prompts) |
| **Prefill, 32K** | **9,575 t/s** | 9,521 t/s | — |
| **Prefill, 128K** | 9,132 t/s | 9,059 t/s | — |
| **Prefill, 256K** | 7,336 t/s | 7,301 t/s | — |
| **Prefill, 512K** | 5,075 t/s | 5,013 t/s | — |
| **Prefill, 1M** | 3,125 t/s (335 s) | 3,124 t/s | — |
| **25 test prompts, end to end** | **19 s** | — | 220 s |
| **Context** | 1,048,576 | 1,048,576 | — |

\* llama.cpp cannot load NVFP4 inside the MTP block, so it ran an MTP-free variant of this file
(`convert.py --no-mtp`) with `--fit on`.

Greedy, 256 generated tokens on wikitext-2 (test + train for the 256K–1M rows; "1M" is 1,047,552 prompt tokens, the window minus 1K), one process per row, so the expert cache starts cold. At 1M the K/V (~6.5 GiB) has taken 56 slots a layer from the expert cache, which is why MTP matters most there: 2.3× decode. MTP is 5 drafts with
a 0.8 probability cut-off, and the "tokens a window" figures are 1.94 / 3.06 / 2.99 / 2.99 / 2.35 / 3.84 at 4K / 32K / 128K / 256K / 512K / 1M. In chat
(temperature 1.0) decode with MTP runs at 155–209 t/s.

With the cache warm (the second prompt in a row), prefill runs at:
- 16K: 12,700 t/s;
- 64K: 11,270 t/s;
- 256K: 7,440 t/s;
- 1M tokens: 340 s.

Needle in a haystack at 32K, 128K, 256K, 512K and 1M (depths 10/50/90%) found 15/15.

Up to ~8K, a prompt is slower than you would expect from the 32K row. Top-22 of 512 experts touches almost every
expert of every layer even on a short prompt, so ~40 GB has to cross PCIe whatever the length. From 16K on, compute
dominates.

## Quality

| FP8 layers of the checkpoint stored as | dense, GiB | PPL / checkpoint | mean KL | top-1 |
| :------------------------------------- | ---------: | :--------------: | ------: | ----: |
| BF16 (= the checkpoint's own values) | 11.57 | 1 | 0 | 100% |
| Q8_0, all of them | ~6.2 | 1.0000 ± 0.0030 | 0.025 | 95.0% |
| **this file: Mamba in/out NVFP4, attention + lm_head Q8_0** | **4.98** | **1.0164 ± 0.0041** | **0.048** | **92.7%** |
| NVFP4, all of them | 4.60 | 1.0290 ± 0.0049 | 0.071 | 90.9% |

How these were measured:
- `llama-perplexity`, wikitext-2 test, 16 × 1024 tokens;
- the experts are the same NVFP4 in every row, and only the dense FP8 layers differ;
- the baseline row holds the checkpoint's own values, so the ratios are against NVIDIA's release, not against BF16
  training weights.

These dense layers are ~6% of the weights, yet putting all of them in NVFP4 costs about three times the KL of NVFP4 on
all the experts. `lm_head` (+1.3% PPL) and Mamba `in_proj` (+0.9%) are the expensive ones. This file takes the middle
row: 2 GiB less VRAM than Q8_0 everywhere, which is ~700 more experts in the cache.

strata-nemotron's fast prompt path adds about ×1.008 PPL on top of this: ×1.003 for FP4 activations, ×1.004–1.005 for
int8 attention. Decode does not use it, and speculative greedy is bit-identical to plain greedy (300/300 tokens).

End to end, 25 prompts went through the OpenAI API (facts, arithmetic, code, SQL, bash, RU/EN translation, formulas;
greedy). strata-nemotron and llama.cpp each got **25/25** correct, and 21 of the 25 answers were byte-identical.

## Parameters

| Parameter | Value |
| :-------- | :---: |
| **routed experts** (40 layers × 512, + 512 in MTP) | NVFP4, NVIDIA's values (59.1 GiB + 1.5 GiB) |
| **Mamba `in_proj` / `out_proj`** (FP8 in the checkpoint) | NVFP4 (our encoder, ModelOpt's scale search) |
| **attention q/k/v/o, `lm_head`** (FP8 in the checkpoint) | Q8_0 |
| **router, shared experts, latent fc1/fc2, norms, embeddings** | BF16 / F32, as in the checkpoint |
| **MTP block** | included (1 layer, its own 512 experts) |
| **vision tower** | not in this file: Unsloth's `mmproj-BF16.gguf` + strata-vision |
| **shards** | 2 (44.3 GB + 27.3 GB), `gguf-split` |

## How to run it

The file runs in strata-nemotron. It needs an RTX 50-series card (sm_120), and its prompt path is built for sm_120a.
Build it:

```sh
git clone -b strata-nemotron https://github.com/Cudila/Strata strata-nemotron && cd strata-nemotron
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j --target strata-nemotron
```

The engine on its own:

```sh
./build/strata-nemotron --gguf Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf \
    --max-context 1048576 --kv-base 65536 --chunk 16384 --vram-reserve-mib 1024 \
    --spec 5 --spec-min-p 0.8 --skip-w 0.4 --tail-skip 0 --prefetch 0 --evict lfu \
    --prompt-skip-w 0.3 --prompt-skip-max 8192 --serve
```

With the OpenAI/Anthropic-compatible server (tool calls, reasoning, the chat template). Put both shards in
`models/`, write the server's tokenizer folder from the GGUF once, then start it:

```sh
python tools/strata_tokenizer.py --gguf models/Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf --out models
python serve/server.py --engine strata --config docs/nemotron/strata-nemotron.example.json --port 8080
```

The first command writes `models/tokenizer/`: vocabulary, merges and the chat template, all taken from the GGUF. The
example config uses paths relative to the repository. `--gguf` takes either shard: the other one is found next to
it.

On startup the engine prints `loaded in … s: dense 6.63 GiB in VRAM, 140 expert slots per MoE layer (16.1 GiB), K/V
65536 cells (grows to 1048576), experts page-locked in place`. On a 32 GB card a slot count well below 140 means
something else holds VRAM.

### Images

The encoder is a separate process, `strata-vision`. It uses
[Unsloth's `mmproj-BF16.gguf`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) and is
built against llama.cpp `098ce526b` with the patch in the fork:

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout 098ce526b
git -C ../llama.cpp apply $PWD/tools/vision/patches/llama.cpp-nemotron-omni-cpe.patch
cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/../llama.cpp \
    -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_PORTABLE=OFF
cmake --build build-vision -j --target strata-vision
```

With `mmproj-BF16.gguf` in `models/`, the example config's `"vision"` section starts the encoder. The server then
takes OpenAI `image_url` parts and Anthropic image blocks.

| | |
| :--- | :--- |
| **tokens a picture** | 256–3,328: NVIDIA's dynamic tile, 1,024–13,312 patches of 16 px |
| **encode time (GPU)** | 36 ms at 512 tokens, 238 ms at 2,040, 463 ms at 3,300 |
| **VRAM** | 2.9 GB for the encoder; the expert cache drops from 140 to 116 slots a layer |
| **accuracy** | against a float32 transcription of NVIDIA's processor + RADIO + projector: rel err 0.003–0.009, per-token cosine ≥ 0.998 (`tools/nemotron/vision_check.py`) |
| **follow-ups** | a turn after a picture reuses the state (2,061 of 2,107 tokens, 146 ms); a different picture of the same size is never reused |

Notes that cost time if you miss them:

* **Give the desktop its VRAM.** If the card also drives your screen, keep `--vram-reserve-mib 1024` or more. A
  compositor or a screen-sharing portal that runs out of VRAM loses its buffers, and a remote-desktop session can
  lose its capture permission with them.
* **`--skip-w 0.4` and `--prompt-skip-w 0.3` are part of the speed.** A cache miss whose routing weight is below the
  threshold is skipped instead of copied over PCIe. Each costs ×1.005 PPL. `--skip-w` takes decode with MTP from
  ~136 to ~180 t/s, and `--prompt-skip-w` takes a 128-token prompt from 0.7 s to 85 ms.
* **Loading takes 35–45 s.** The 59 GiB of experts are copied into page-locked RAM once, from the page cache if the
  file is in it. Pageable memory would put every PCIe copy through the driver's staging at a fraction of the speed.
* **The KV grows into the expert cache.** Up to `--kv-base` (64K) it is fixed. Beyond that, each 64K page takes
  4 slots of every layer from the cache: 140 slots at 64K, 128 at 256K, 84 at 1M. They are given back on the next
  conversation.
* **llama.cpp does not load this file.** It has no NVFP4 inside the MTP block. `tools/nemotron/convert.py --no-mtp`
  in the fork makes a variant it can load.

## Tested hardware

Everything measured in this card was measured on exactly one machine:

| | |
| :--- | :--- |
| **GPU** | NVIDIA RTX 5090, 32 GB (driver 610.43.03, CUDA 13.3) |
| **CPU** | Intel Core i7-14700KF, 8P + 12E, 28 threads |
| **RAM** | 128 GB DDR5-5200 |
| **PCIe** | 5.0 ×16, ~45 GB/s host→device measured |
| **OS** | Linux (Arch, kernel 7.1) |
| **Engine** | strata-nemotron, [Cudila/Strata](https://github.com/Cudila/Strata) branch `strata-nemotron` |
| **Desktop** | on the same card, 1 GB of VRAM kept free for it |

Windows, AMD, other GPUs, multi-GPU and smaller RAM sizes were not tried.

## Hardware configurations

What decides whether this runs is where its parts live:

| part | size | where |
| :--- | ---: | :--- |
| routed experts (all 20,480 + MTP's 512) | 59.1 GiB + 1.5 GiB | page-locked RAM |
| dense weights + the MTP block | 6.63 GiB | GPU |
| prompt buffers, chunk 16384 + attention scratch | ~5–6 GiB | GPU |
| K/V, int8 K + FP16 V, 64K cells | ~0.4 GiB | GPU, grows in 64K pages |
| expert cache | everything left | GPU, 2.95 MiB a slot, at least 22 slots a layer |

| VRAM | example cards | expert slots a layer | result |
| ---: | :--- | ---: | :--- |
| 32 GB | 5090 | 140 | **measured**: the tables above |
| 24 GB | 5090 Laptop | 72 | **emulated** on the 5090 by leaving the engine 24 GB: prefill 4K 2,382 t/s, 32K 9,153 t/s; decode 121 / 198 t/s at 4K / 32K. Within a few % of the 32 GB row. The 1M context will not fit: its KV takes 56 slots, leaving fewer than 22 |
| 16 GB | 5080, 5070 Ti, 5060 Ti 16G | 13 | **does not start yet**, emulated the same way: the engine needs 22 slots a layer, one token's experts. **In development**, with 64 GB of RAM: cold experts on the SSD |
| 12 GB and below | 5070, 5060 | — | does not start |

The 24 GB row was emulated with `--vram-reserve-mib` on the same card. So it shows what a smaller cache costs, not
what a laptop GPU's lower compute and power limit cost: read it as an upper bound.

That a half-size cache is nearly free is not a fluke. With skipping by routing weight, the misses that remain are
mostly experts whose weight is too small to matter, and long prompts are bound by compute.

**RAM:**
- The page-locked expert arena is 59 GiB and cannot be swapped.
- The process peaked at 105 GiB resident on the 128 GB machine, but that includes the file's pages in the page cache,
  which the kernel can drop.
- 96 GB is probably enough, but was not tried. 64 GB is not: the arena alone does not fit next to the system.

## Files

| File | Description |
| :--- | :---------- |
| `Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf` | shard 1 (44.3 GB): the metadata and 423 tensors |
| `Nemotron-3.5-Super-120B-A12B-NVFP4-00002-of-00002.gguf` | shard 2 (27.3 GB): the other 766, with the embeddings, `lm_head` and the MTP block |

1189 tensors: BF16 88, F32 824, NVFP4 244, Q8_0 33. sha256: shard 1 `98b3cd49…839be6`, shard 2 `60933aa7…b50ac3`.

## How it was made

1. **Conversion.** `tools/nemotron/convert.py` in the fork wraps llama.cpp's converter:
   `--fp8 nvfp4 --nvfp4-only "mixer\.(in|out)_proj" --fp8-as-q8`. On this checkpoint the stock converter needs four
   workarounds, which the wrapper applies without patching llama.cpp:
   - it drops the vision tensors;
   - it renames the MTP tensors before they are packed;
   - it reads `n_routed_experts` (otherwise all 59 GiB of experts pile up in RAM);
   - it treats `quant_method: modelopt_mixed` as ModelOpt for the FP8 dequantization.
2. **FP8 → NVFP4** for the Mamba projections. It uses the same block-scale search as ModelOpt: the minimum-MSE
   neighbour among the E4M3 codes, not rounded amax/6. On the experts this encoder gives a relative error of 0.0812
   against ModelOpt's 0.0813.
3. **The choice of mix.** The tensor classes were measured one by one in llama.cpp, as in the Quality table. The mix
   was then picked for speed at an acceptable perplexity.
4. **Splitting.** `llama-gguf-split --split-max-size 45G`. `llama-gguf-split` has to come from a llama.cpp build that
   knows the NVFP4 type (40). The split file gives the same perplexity in strata-nemotron as the single file: 3.2120 on
   8 × 1024 tokens.
5. **Checks.** `strata-nemotron-check` checks every tensor against `config.json`: name, type, exact shape, file
   bounds. Every kernel of the engine has a parity test against a float64 transcription of the model's equations.

## Caveats

* Images work, video does not yet. Stock llama.cpp's `nemotron_omni` stretches RADIO's position table on
  non-square pictures (per-token cosine down to 0.32 against NVIDIA's code). The fork ships a patch for it.
  strata-vision also uses NVIDIA's own tile size and resampling, which differ from mtmd's generic dynamic size.
* Blackwell (sm_120) only: the engine's kernels use FP4 tensor-core instructions that earlier GPUs do not have.
* One machine, one card (*Tested hardware*). The 24 GB row is an emulation; the smaller rows follow from the
  engine's own refusal message.
* PPL ×1.016 against the checkpoint is a wikitext mean. No downstream benchmarks were run beyond the 25-prompt check.
* The reasoning modes, the chat template and the model's behaviour are NVIDIA's. This repo changes only how the
  weights are stored.

## Credits

The base model is by [NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4). The engine
is a fork of [Strata](https://github.com/Niko1221/Strata), and the conversion goes through
[llama.cpp](https://github.com/ggml-org/llama.cpp). The weights are NVIDIA's and are governed by the
[OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/). NVIDIA's model card covers intended use, training
data, bias and safety.
