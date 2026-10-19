# strata-nemotron: stages 7–8 — end-to-end checks and measurements (2026-10-19)

(Russian original: `ru/STAGE7-8_RESULTS.md`.)

Machine: RTX 5090 32 GB (sm_120), i7-14700KF, 128 GB DDR5-5200, PCIe 5 (~45 GB/s host→device).

Model: `Nemotron-3.5-Super-120B-A12B-NVFP4` (experts NVFP4, Mamba in/out NVFP4, attention and lm_head Q8_0, the rest
BF16, MTP). 1 GB of VRAM is always kept free for the desktop, and the engine runs with `--vram-reserve-mib 1024`.

## Stage 7: correctness

**Perplexity** (wikitext-2 test, 40 chunks × 1024, the second half of each chunk scored, prompt path):

| path | PPL | pairwise ΔNLL against the row above |
|---|---:|---|
| llama.cpp on the same weights (the reference, stage 1) | 3.065 (16 × 1024) | — |
| Strata v1 (BF16 path) | 3.034 (16 × 1024) | — |
| Strata v3 before the third wave | 4.0516 (40 × 1024) | — |
| + W8A8 for attention and fc1/fc2 (now) | 4.0533 | −0.0004 ± 0.0022 (noise) |

4.05 and 3.03 come from different chunk sets: 40 chunks against 16.

The accumulated cost of every approximation on the prompt path, against exact BF16, is about ×1.008 PPL:
- W4A4: ×1.003;
- int8 attention: ×1.004–1.005;
- everything else within noise.

The decode path is no worse than v2.

**Comparison with the reference through the API** (`tools/nemotron/api_compare.py`):
- 25 prompts: facts, arithmetic, code, SQL, bash, RU/EN translation, formulas;
- greedy, through the OpenAI API, to a Strata server and to a llama.cpp server on the same weights;
- llama.cpp used `-noMTP`, because it does not load NVFP4 inside the MTP block.

| | correct | time (including load) |
|---|---:|---:|
| strata-nemotron | 25 / 25 | 19 s |
| llama.cpp | 25 / 25 | 220 s |

The answers are byte-identical in 21 of 25. In the other four they share a beginning of 3–101 characters and then
diverge. The kernels' arithmetic differs, so with close logits greedy picks another token. The meaning of the answers
is the same.

**Greedy on long context**: needle in a haystack through the server (`tools/needle_bench.py`, depths 10/50/90%).

| when | lengths | found |
|---|---|---:|
| after v3 | 32K, 128K, 256K, 512K, 1M (1 027 882 tokens) | 15/15 |
| after the second wave | 32K, 256K | 6/6 |
| after the third wave | 256K, 1M (1 029 587 tokens) | 6/6 |

1M takes 340 s now (it was 374). Speculative greedy is bit-identical to plain decode (300/300).

**swift15 is not affected.**
- The shared files are `gguf_reader.hpp`, `serve/server.py` and `tools/strata_tokenizer.py`. CMake only adds new
  libraries.
- The test: the production engine against this branch's `build/strata`, on the swift15 config, with a 3000-token
  prompt.
- The logits after the prompt path (`STRATA_DUMP_FIRST_LOGITS`) are bit-identical, and so are 128 greedy tokens.
- Speed is within noise: decode 193 / 192 tok/s, prefill 4087 / 3998 tok/s (the other way round in another run).
- Tokenizer: 20 texts (with and without special tokens, a tool call, Unicode, code) give identical ids. The server's
  stop tokens are the same.

## Stage 8: measurements

`tools/nemotron/stage8_bench.py`. Setup:
- the same wikitext-2, each model with its own tokenization;
- a prompt of the first N tokens, then 256 greedy tokens;
- one process per row, so the expert cache is cold at the start of the prompt;
- VRAM is the card's `nvidia-smi` peak, including the desktop (~1 GB);
- RSS is the process peak. For Nemotron it includes the experts page-locked in RAM (~63 GB) and the page cache.

| model | context | MTP | prefill, tok/s | decode, tok/s | tokens a window | VRAM peak, MiB | RSS peak, MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| Nemotron | 4K | yes (5) | 2277 | 118 | 1.94 | 30159 | 104680 |
| Nemotron | 4K | no | 2447 | 96 | — | 30159 | 104999 |
| Nemotron | 32K | yes (5) | 9575 | 186 | 3.06 | 30159 | 105313 |
| Nemotron | 32K | no | 9521 | 109 | — | 30159 | 105255 |
| Nemotron | 128K | yes (5) | 9132 | 165 | 2.99 | 30159 | 105264 |
| Nemotron | 128K | no | 9059 | 95 | — | 30155 | 105426 |
| Nemotron | 256K | yes (5) | 7336 | 138 | 2.99 | 30143 | 102840 |
| Nemotron | 256K | no | 7301 | 79 | — | 30143 | 103384 |
| Nemotron | 512K | yes (5) | 5075 | 107 | 2.35 | 30143 | 103116 |
| Nemotron | 512K | no | 5013 | 65 | — | 30155 | 102989 |
| Nemotron | 1M (1 047 552) | yes (5) | 3125 | 111 | 3.84 | 30151 | 102787 |
| Nemotron | 1M (1 047 552) | no | 3124 | 47 | — | 30151 | 100082 |
| swift15 | 4K | yes (4) | 3311 | 117 | 1.74 | 30915 | 46985 |
| swift15 | 32K | yes (4) | 6457 | 128 | 1.61 | 30919 | 47125 |
| swift15 | 128K | yes (4) | 6623 | 173 | 2.53 | 30919 | 47130 |

The 256K–1M rows (added 2026-10-19) run on wikitext-2 test + train (2.87M tokens) with the rebased build;
"1M" is the window minus 1K. At 1M the K/V has taken 56 slots a layer from the expert cache.

swift15 does not run without MTP: its native pack requires `--spec ≥ 2`.

Decode in the table is greedy on wikitext. In chat (temperature 1.0), Nemotron with speculation runs at 155–163 tok/s,
and 177–209 through the server.

Prefill with the cache warm (the second prompt in a row):

| prompt | time | tok/s |
|---|---:|---:|
| 128 tokens | 98 ms | — |
| 2K | 0.77 s | 2650 |
| 8K | 1.0 s | 8000 |
| 16K | 1.29 s | 12 700 |
| 64K | 5.8 s | 11 270 |
| 256K | 35 s | 7 440 |
| 1M, through the server | 340 s | ~3030 |

Up to ~8K, Nemotron's prompt is slower than swift15's. Top-22 of 512 experts touches almost all experts of every
layer, so the time is set by ~40 GB of copies over PCIe (OPTIMIZATION_NOTES.md, 2.2). From 32K on, Nemotron is
1.4–1.5× faster in prefill, and its decode with speculation is the same or higher.

## Not ported or not done

- Vision and video (later, per the task).
- Streaming the KV from RAM for 1M. Nemotron's attention is full, so that would cost ~0.2 s per token. Instead, the
  KV lives in VRAM in pages, taking space from the expert cache.
- Strata optimizations that do not apply to this architecture: the QSA indexer and sparse attention (Nemotron has
  neither), PLE, GDN.
- The CPU share of experts on prefill (swift15 has it for chunks < 1024): not ported and not measured.
- The prompt path is not bit-identical to the decode path (W4A4, int8 attention, SSD). Bit identity holds within
  decode: speculation equals plain decode.
