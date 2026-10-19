# strata-nemotron: engine v1 (2026-10-17)

(Russian original: `ru/ENGINE_V1.md`.)

## What works

- **The engine binary.** `build/strata-nemotron` (`src/program/nemo_main.cpp`, `src/core/nemo_model.cpp`) is a separate
  binary. It speaks Strata's engine protocol (`GEN` / `T` / `DONE`, `STOP`, `QUIT`), so the regular `serve/server.py`
  drives it: the OpenAI/Anthropic API, the chat template, reasoning, tool calls.
- **The server config** is `serve/server.py --config <config>.json`; see `strata-nemotron.example.json`.
- **Placement in memory:**
  - dense weights in VRAM in their GGUF formats (5.0 GiB);
  - experts in a pinned arena in RAM (59 GiB, ~30 s to start);
  - an LRU expert cache in VRAM (~180 slots per MoE layer, ~21 GiB);
  - BF16 KV for the context, FP32 SSM states.
- **Conversation reuse.** The state is saved at the boundary of the last finished message, so the next turn reads only
  what is new. Measured: 3597 of 3632 tokens came from the snapshot, and the prompt took 1.1 s instead of 23.5 s.
- **Prefill on tensor cores.**
  - Dense matrices and experts: a BF16 copy of the weight (exact for NVFP4, with scale_2 as alpha) plus cuBLAS.
  - Mamba and the shared expert: sub-chunks of 2048 rows.
  - Experts: groups of ≤ 32, whose outputs are summed straight into `routed` (no `parts` buffer).
  - Decode and windows of ≤ 15 tokens: our own CUDA-core kernels (`nemo_gemm`).

## Checks

| check | result |
|---|---|
| PPL wikitext-2, 16 × 1024 (BOS at the start of each piece) | 3.0337 (FP32 path) / 3.0347 (tensor cores); llama.cpp on the same weights 3.065 ± 0.069, exact FP8 3.016 |
| greedy answer to "quicksort" | the same text as llama.cpp, word for word (the first paragraphs) |
| server test set | plain / tool call / tool result + system in the middle + high / developer: all pass |
| the server's tokenizer against HF tokenizer.json | 2896 wikitext lines + test strings + 2 chats: 0 differences |
| `nemo_gemm_parity` (NVFP4/Q8_0/BF16/F32, M = 1..100) | ≤ 6.2e-7 of RMS against fp64; a token's row is bit-identical for any M |

## Found and fixed along the way (shared code)

- `serve/server.py`: `stop_ids` took every token of the string `<|endoftext|>`. Nemotron has no such token, and its
  piece `<` ended every tool call at `<function=`. Now only single-token strings go into the stop set. For Qwen both
  strings are single tokens, so swift15 behaves the same.
- `serve/server.py` + `tools/strata_tokenizer.py`: the pre-tokenizer now comes from the pack's `tokenizer.json`. For
  `pixtral`/`tekken` that is the regex from the model's tokenizer.json plus `ignore_merges`; `qwen35` works as before.
- `gguf_reader.hpp`: numeric metadata arrays of ≤ 4096 items are kept whole; `map()`.

## Speed (RTX 5090, 14700KF, DDR5-5200, greedy)

| | v1 (FP32 CUDA cores, chunk 2048) | then (cuBLAS BF16, chunk 8192) | llama.cpp, same weights |
|---|---:|---:|---:|
| decode, short prompt | 21.5 tok/s | ~21 tok/s | 19.2–19.6 tok/s |
| prefill 8K | 151 tok/s | 1270 tok/s | not measured |

Profile of an 8K prefill (chunk 8192, with syncs): experts' compute 39%, their PCIe copies 22%; Mamba 21%;
attention 7%. Profile of decode: experts ~59% (PCIe misses under LRU, hit rate 56–60%), Mamba 22%.

## Next for speed

1. Decode: a CPU lane for misses (the CPU computes part of the missed experts in parallel with PCIe); a frequency
   profile instead of LRU; then MTP (stage 6).
2. Expert quants smaller than NVFP4 (stage 5): IQ3_XXS/IQ2_S + IQ4_NL/Q2_0. That gives ×1.7 experts in VRAM and half
   the bytes per miss.
3. Prefill: overlap expert copies with compute (a separate stream); FP4 tensor cores instead of BF16.
