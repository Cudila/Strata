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

### **186 t/s decode and 9,610 t/s prefill at 32K** on one RTX 5090 with the experts in RAM (130 t/s in the precise decode mode): 6× llama.cpp's decode on the same weights, with NVIDIA's own NVFP4 experts and the MTP head kept.

> **Blackwell only: RTX 50-series (sm_120).** Every optimization of the engine is built on Blackwell's FP4 tensor-core
> instructions (`mma … kind::mxf4nvf4.block_scale`, NVFP4 × NVFP4 with its native scales), which no earlier
> generation has. The engine is compiled for `sm_120a` and does not run on RTX 40/30-series or on AMD.
>
> **Vision works.** Images go through strata-vision with Unsloth's
> [`mmproj-BF16.gguf`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) and a patched
> llama.cpp (see *Images* below). Its embeddings match NVIDIA's own code to a per-token cosine of ≥ 0.998. Video is
> not done yet.
>
> **16 GB cards work** (5080, 5070 Ti, 5060 Ti 16G; with 128 GB of RAM). With MTP and the precise decode: 53–78 t/s
> decode, ~4,600 t/s prefill at 32K, decode PPL ×1.010. Measured by emulation on the 5090 (`--vram-limit-mib 15800`),
> see *Hardware configurations*. **64 GB of RAM is in development:** cold experts streamed from an NVMe SSD.

This repo has a GGUF of the text model of
[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4),
made for [strata-nemotron](https://github.com/Cudila/Strata). That is a fork of the
[Strata](https://github.com/Niko1221/Strata) engine that runs this model on one consumer card: 63 GB of experts in
pinned RAM, a VRAM cache in front of them, and every NVFP4 GEMM of the prompt on Blackwell's FP4 tensor cores.

The routed experts are NVIDIA's NVFP4 values, bit for bit, repacked to ggml's `NVFP4` without loss. The 1.6 GB
multi-token-prediction (MTP) block is kept: it takes decode at 32K from 109 to 186 t/s (fast mode) and from 52 to 130 t/s (precise). The vision tower is not
in this file: it is Unsloth's mmproj (see *Images*).

## Performance

strata-nemotron has two decode modes. They differ only in what happens to a cache miss whose routing weight is low:
- **fast** (`--skip-w 0.4 --prefetch 0`): the miss is skipped;
- **precise** (`--skip-w 0.4 --prefetch 1`): the next layer's likely experts are prefetched, so far fewer misses are
  skipped.

Prefill is the same in both.

| Decode | MTP on, fast | MTP on, precise | MTP off, fast | MTP off, precise | llama.cpp, same weights* |
| :----- | :---: | :---: | :---: | :---: | :---: |
| **4K prompt** | 118 t/s | 88 t/s | 96 t/s | 52 t/s | 19.4 t/s |
| **32K prompt** | **186 t/s** | **130 t/s** | 109 t/s | 52 t/s | — |
| **128K prompt** | 165 t/s | 85 t/s | 95 t/s | 50 t/s | — |
| **256K prompt** | 138 t/s | 85 t/s | 79 t/s | 44 t/s | — |
| **512K prompt** | 107 t/s | 69 t/s | 65 t/s | 41 t/s | — |
| **1M prompt** | 111 t/s | 80 t/s | 47 t/s | 31 t/s | — |
| **decode PPL against no skipping** | ×1.091 | ×1.008 | ×1.091 | ×1.008 | — |

| Prefill | MTP on | MTP off | llama.cpp, same weights* |
| :----- | :---: | :---: | :---: |
| **4K** | 2,368 t/s | 2,295 t/s | ~30–40 t/s (short prompts) |
| **32K** | **9,610 t/s** | 9,390 t/s | — |
| **128K** | 8,894 t/s | 9,047 t/s | — |
| **256K** | 7,313 t/s | 7,339 t/s | — |
| **512K** | 5,094 t/s | 5,095 t/s | — |
| **1M** | 3,090 t/s (339 s) | 3,142 t/s | — |

| | strata-nemotron | llama.cpp, same weights* |
| :--- | :---: | :---: |
| **25 test prompts, model loaded** | 19 s fast, 30 s precise | — |
| **the same, including the load** | 93 s (precise, with the vision encoder) | 220 s |
| **Context** | 1,048,576 | — |

\* llama.cpp cannot load NVFP4 inside the MTP block, so it ran an MTP-free variant of this file
(`convert.py --no-mtp`) with `--fit on`.

Greedy, 256 generated tokens on wikitext-2 (test + train for the 256K–1M rows; "1M" is 1,047,552 prompt tokens, the window minus 1K), one process per row, so the expert cache starts cold. At 1M the K/V (~6.5 GiB) has taken 56 slots a layer from the expert cache, which is why MTP matters most there: 2.4× decode (fast), 2.6× (precise). MTP is 5 drafts with
a 0.8 probability cut-off, and the "tokens a window" figures in the precise mode are 2.23 / 3.56 / 2.05 / 2.56 / 2.21 / 3.76 at 4K / 32K / 128K / 256K /
512K / 1M. The decode PPL is `--ppl-decode` on wikitext-2 (4 × 1024), against the same engine with no skipping.

![Speed of every request strata-nemotron v3 served through llama-swap: prompt and generation histograms](https://huggingface.co/Cudecnik/Nemotron-3.5-Super-120B-A12B-NVFP4-Strata-Blackwell/resolve/main/assets/nemotron-speed-histogram.png)

*Real use, not a benchmark:* all 363 requests the v3 engine served through llama-swap in the **fast** decode mode, on
the machine of *Tested hardware* (everyday chat plus the checks in this card: tool calls, images, the 512K–1M needle
tests), with MTP and the sampling the clients asked for. Generation is 181 · 229 · 247 t/s (p50 · p95 · p99) and never
below 106 t/s. Prompt is 1,257 t/s at p50, because most requests are short follow-ups whose prompt is a few hundred
tokens, and 9,169 t/s at p95.

The fastest prompts in the histogram, through the server:
- 15.6K tokens: 11,397 t/s;
- 62K tokens: 10,499 t/s;
- 1M tokens: ~3,050 t/s (340 s).

The engine alone (`--bench`, the second prompt in a row, so the expert cache is warm) is ~10% faster: 12,700 t/s at 16K,
11,270 at 64K, 7,440 at 256K. Through the server, a prompt is read in two parts: up to the last complete message,
where the state is saved for the next turn, and then the rest. The cache is also not always warm there.

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

strata-nemotron's prompt path adds about ×1.008 PPL on top of this: ×1.003 for FP4 activations, ×1.004–1.005 for int8
attention. Decode adds ×1.008 in the precise mode and ×1.091 in the fast one (see *Performance*). Speculative greedy is
bit-identical to plain greedy (300/300 tokens).

End to end, 25 prompts went through the OpenAI API (facts, arithmetic, code, SQL, bash, RU/EN translation, formulas;
greedy). strata-nemotron (fast mode) and llama.cpp each got **25/25** correct, and 21 of the 25 answers were
byte-identical. In the precise mode: 25/25 in two of three runs, 24/25 in one (a reasoning trace ran past the
1,024-token limit).

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
* **`--skip-w 0.4`, `--prefetch` and `--prompt-skip-w 0.3` are part of the speed.** A cache miss whose routing weight
  is below the threshold is skipped instead of copied over PCIe. `--prefetch 1` copies the next layer's likely experts
  ahead of time, so far fewer are skipped: precise decode, ×1.008 PPL. `--prefetch 0` is the fast mode, ×1.091.
  Without skipping at all, decode runs at 43 t/s. `--prompt-skip-w` takes a 128-token prompt from 0.7 s to 85 ms.
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
| 24 GB | 5090 Laptop | 72 | **emulated** on the 5090 by leaving the engine 24 GB: prefill 4K 2,328–2,382 t/s, 32K 9,040–9,153 t/s; decode at 4K / 32K 121 / 198 t/s fast (as the 32 GB row), 64 / 93 t/s precise (−27% against the 32 GB row). The 1M context will not fit: its KV takes 56 slots, leaving fewer than 22 |
| 16 GB | 5080, 5070 Ti, 5060 Ti 16G | 40 (MTP) / 64 | **emulated**: `--vram-limit-mib 15800` (the process stays under 15.8 GB, its CUDA context included), the small-card profile (8192-row chunk, a 48-blob prompt ring), MTP, precise decode. Decode 78 t/s at 4K, 69–74 t/s at 32K–256K; prefill 4,625 t/s at 32K (chunk 8192). Decode PPL ×1.010 (×1.17 in the fast mode: do not use it here). The vision encoder runs on the CPU (`docs/nemotron/strata-nemotron-16gb.example.json`). Context up to 256K tested |
| 12 GB and below | 5070, 5060 | — | does not start |

The 24 GB row was emulated with `--vram-reserve-mib` on the same card, the 16 GB row with `--vram-limit-mib`. So it shows what a smaller cache costs, not
what a laptop GPU's lower compute and power limit cost: read it as an upper bound.

In the fast mode a half-size cache is nearly free: the misses it adds are mostly experts whose weight is below the
skip threshold, and long prompts are bound by compute. In the precise mode those misses are copied, so decode drops
by ~27%.

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
