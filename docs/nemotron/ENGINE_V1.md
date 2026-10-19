# strata-nemotron: движок v1 и слот llama-swap (2026-10-17)

## Что работает

- `build/strata-nemotron` (`src/program/nemo_main.cpp`, `src/core/nemo_model.cpp`) — отдельный бинарь ветки, протокол
  движка Strata (`GEN` / `T` / `DONE`, `STOP`, `QUIT`), поэтому его ведёт штатный `serve/server.py` (OpenAI/Anthropic
  API, шаблон, reasoning, tool calls).
- Слот `nemotron-3.5-super` в llama-swap → `serve/server.py --config strata-nemotron-mix.json` (ветка);
  запасной `nemotron-3.5-super-llamacpp` на llama.cpp-glm5next.
- Размещение: плотные веса в VRAM в форматах GGUF (5.0 GiB), эксперты — pinned-арена в RAM (59 GiB, старт ~30 с),
  LRU-кэш экспертов в VRAM (~180 слотов на MoE-слой, ~21 GiB), KV BF16 на контекст, SSM-состояния FP32.
- Переиспользование диалога: снимок состояния на границе последнего завершённого сообщения; следующий ход читает
  только новое (замер: 3597 из 3632 токенов взяты из снимка, prompt 1.1 с вместо 23.5 с).
- Prefill на tensor cores: плотные матрицы и эксперты — BF16-копия веса (точная для NVFP4, scale_2 → alpha) +
  cuBLAS; Mamba и shared-эксперт подчанками по 2048 строк, эксперты группами по ≤32, их выходы сразу суммируются в
  `routed` (без буфера `parts`). Decode и окна ≤ 15 токенов — свои CUDA-core ядра (`nemo_gemm`).

## Проверки

| Проверка | Результат |
|---|---|
| PPL wikitext-2, 16 × 1024 (BOS в начале куска) | 3.0337 (FP32-путь) / 3.0347 (tensor cores); llama.cpp на том же GGUF 3.065 ± 0.069, точный FP8 3.016 |
| Жадный ответ на «быструю сортировку» | дословно совпадает с llama.cpp (первые абзацы) |
| Проверочный набор через llama-swap | plain / tool call / tool result + system посреди + high / developer — все проходят |
| Токенизатор сервера против HF tokenizer.json | 2896 строк wikitext + проверочные строки + 2 чата: 0 расхождений |
| `nemo_gemm_parity` (NVFP4/Q8_0/BF16/F32, M = 1..100) | ≤ 6.2e-7 от RMS против fp64; строка токена побитно не зависит от M |

## Найдено и исправлено по дороге (общий код)

- `serve/server.py`: `stop_ids` брал все токены строки `<|endoftext|>`; у Nemotron её нет, и её кусок `<` обрывал
  каждый tool call на `<function=`. Теперь в стоп-набор идут только однотокенные строки (у Qwen обе однотокенные —
  поведение swift15 то же).
- `serve/server.py` + `tools/strata_tokenizer.py`: pre-токенайзер берётся из `tokenizer.json` пака (`pixtral`/`tekken`:
  регулярка из tokenizer.json модели + `ignore_merges`); `qwen35` — как было.
- `gguf_reader.hpp`: NVFP4 (40) в `block_geometry`; числовые массивы метаданных ≤ 4096 хранятся целиком; `map()`.

## Скорость (RTX 5090, 14700KF, DDR5-5200, greedy)

| | v1 (FP32 CUDA cores, chunk 2048) | сейчас (cuBLAS BF16, chunk 8192) | llama.cpp-glm5next, тот же GGUF |
|---|---:|---:|---:|
| decode, короткий промпт | 21.5 т/с | ~21 т/с | 19.2–19.6 т/с |
| prefill 8K | 151 т/с | 1270 т/с | не мерили |

Профиль prefill 8K (chunk 8192, с синхронизациями): эксперты — счёт 39%, копии по PCIe 22%; Mamba 21%; attention 7%.
Профиль decode: эксперты (промахи по PCIe при LRU, hit rate 56–60%) ~59%, Mamba 22%.

## Дальше по скорости

1. Decode: CPU-лейн для промахов (часть промахнувшихся экспертов считает CPU параллельно с PCIe), профиль частот
   вместо LRU, затем MTP (этап 6).
2. Кванты экспертов меньше NVFP4 (этап 5): IQ3_XXS/IQ2_S + IQ4_NL/Q2_0 — ×1.7 экспертов в VRAM и вдвое меньше байт
   на промах.
3. Prefill: перекрыть копии экспертов со счётом (отдельный поток), FP4 tensor cores вместо BF16.
