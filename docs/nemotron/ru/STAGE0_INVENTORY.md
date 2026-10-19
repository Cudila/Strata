# strata-nemotron, этап 0: инвентаризация Strata под Nemotron 3.5 Super VL

Дата: 2026-10-17. База: `main` @ a0116b27 (Strata 0.1.43.x). Модель: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-{BF16,NVFP4}`.
Источники: `config.json`, заголовки safetensors обоих чекпоинтов (все шарды, только метаданные), `hf_quant_config.json`,
`transformers/models/nemotron_h/modeling_nemotron_h.py` (5.15.1), `third_party/llama.cpp/src/models/nemotron-h*.cpp`.
Кода пока нет: это материал для ревью перед этапом 1.

## 1. Модель: что проверено и где описание в таске неточно

| Параметр | Значение (проверено) | Расхождение с таской |
|---|---|---|
| Блоки | 88: 40 Mamba-2, 40 MoE, 8 attention. Каждый блок = RMSNorm → один миксер → residual. Отдельного FFN после attention нет | — |
| Порядок | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME` (M=mamba, E=moe, *=attn) | нерегулярный, брать из `layers_block_type` |
| hidden | 4096, vocab 131072, embeddings и lm_head не связаны | — |
| MoE | 512 routed, top-22, 1 shared | совпадает |
| Роутер | вход: полный hidden 4096, FP32 GEMV, sigmoid; выбор по `sigmoid + e_score_correction_bias`, веса из sigmoid без bias, нормировка на сумму, ×5.0 (`routed_scaling_factor`). n_group=1, т.е. групповой шаг вырожден | у Strata softmax top-10 |
| Латентная проекция | есть: `fc1_latent_proj` 4096→1024 перед экспертами, `fc2_latent_proj` 1024→4096 после суммы. Оба BF16 | подтверждено |
| Эксперт | **без гейта**: `down(relu²(up(x)))`, up 2688×1024, down 1024×2688, работает в латенте 1024 | у Strata SwiGLU (gate+up+down) |
| Shared-эксперт | relu², 4096→5376→4096, в полном hidden, вход тот же нормированный x, без скалярного гейта | у Strata есть sigmoid-гейт shared |
| Attention | 32 Q-головы, 2 KV (GQA 16:1), head_dim 128, без bias, scale 1/√128, **без RoPE (NoPE)**: `apply_rotary_pos_emb` объявлена, но не вызывается, llama.cpp тоже пишет «dense NoPE attention». Окна нет | RoPE нет |
| Mamba-2 | in_proj 4096→18560 = z 8192 \| xBC 10240 \| dt 128; conv1d k=4 по 10240 каналам, с bias, SiLU; 128 голов × 64, n_groups 8 (B,C общие на 16 голов), d_state 128; dt = softplus(dt+dt_bias) без клампа (`time_step_limit` 0..inf); A = −exp(A_log) скаляр на голову; D-skip на голову; gated RMSNorm: `rms_по_группам_1024(y·silu(z))·w`; out_proj 8192→4096; chunk_size 128; кэш SSM в FP32 | — |
| MTP | 1 модуль (`num_nextn_predict_layers=1`) из двух подслоёв: [0] `eh_proj(concat(enorm(emb), hnorm(h)))` 8192→4096 → norm → NoPE attention (своя KV); [1] norm → MoE 512/top-22 с латентом и shared → `final_layernorm` → общий lm_head. Порядок concat сверить с llama.cpp на этапе 6 | — |
| Контекст | `max_position_embeddings` 1048576 | — |

**NVFP4-чекпоинт смешанный (`MIXED_PRECISION`), а не чистый NVFP4:**

| Тензоры | Формат в чекпоинте | Объём |
|---|---|---|
| routed-эксперты (40×512 + MTP 512), shared-эксперты | NVFP4: U8 `[out, in/2]` (2×E2M1 на байт), `weight_scale` F8_E4M3 `[out, in/16]`, `weight_scale_2` F32 на тензор, плюс `input_scale` F32 (статический глобальный scale активаций для W4A4) | 59.06 GiB эксперты (2.95 MiB на эксперта), 0.92 GiB shared |
| Mamba in_proj/out_proj, attention qkvo, lm_head | **FP8 E4M3 per-tensor** (`weight_scale` скаляр, статический `input_scale`) | 2.83 + 1.25 + 0.27 + 0.5 GiB |
| embeddings, латент fc1/fc2, роутер, conv1d, A/D/dt, нормы, MTP attention и eh_proj | BF16 (роутерный bias F32) | 1 + 0.63 + 0.16 GiB … |
| KV | `kv_cache_quant_algo: FP8`, k/v scale есть только у MTP attention | — |

Вывод по формату NVFP4 из таски: «FP4 E2M1, FP8 E4M3 scale на 16, FP32 на тензор» совпадает. Добавки: (1) `input_scale`,
то есть референс vLLM/TRT-LLM считает W4A4; (2) порядок нибблов (младший = чётный элемент) проверю на этапе 1 по modelopt;
(3) в ggml свой NVFP4 (`GGML_TYPE_NVFP4`=40): суперблок 64 = 4 scale UE4M3 + 32 байта, глобального scale в блоке нет.
Переупаковка байтов без потерь, `weight_scale_2` уходит в отдельный тензор (так делает `conversion/base.py::_repack_nvfp4`).

Есть ещё вариант `NVFP4-LiLiCorr`, в таске его нет. Нужно решить, смотрим ли его.

## 2. Бюджет памяти (расчёт, не замер)

| Что | Объём |
|---|---|
| Плотные веса в VRAM (NVFP4-чекпоинт как есть, без embeddings) | ~6.6 GiB |
| MTP (эксперты 1.48 + остальное) | ~1.65 GiB |
| KV attention, BF16: 8 KiB/токен (8 слоёв × 2 головы × 128 × K,V) | 32 MiB @4K, 256 MiB @32K, 1 GiB @128K, 8 GiB @1M |
| SSM-состояние на последовательность: 40 × 128 × 64 × 128 × FP32 | 160 MiB (+ conv 4.9 MB) |
| Остаток под кэш экспертов на 32 GB | ~19 GiB ≈ 6,600 из 20,480 экспертов (~32%) |
| Эксперты в RAM (NVFP4) | 59 GiB из 128 GB: low-RAM режимы не нужны |

Сравнение со swift15 (IQ3_XXS) по документации Strata: там 480 чтений эксперта на токен по ~1.75 MB (~0.84 GB/токен) и ~56% экспертов
в VRAM. У Nemotron 880 чтений по 2.95 MiB (~2.7 GB/токен) и ~32% в VRAM. Трафик экспертов на токен примерно в 3 раза больше,
доля в VRAM в 1.7 раза меньше. Оценка потолка decode без MTP: при hit rate 60% промахи дают ~1 GiB/токен из DDR5 (~60 GB/s
реально, замер на Maya), это ~17 мс, то есть ≤55–60 т/с. Это **оценка**, hit rate даст только профиль. Отсюда следует, что этап 5
(свои кванты ~2.5–3 bpw для экспертов) по скорости важнее, чем выглядит в таске.

**Ограничение для своих квантов:** у `down_proj` эксперта вход 2688 = 10.5 × 256. K-кванты и IQ2_XS/IQ3_XXS/IQ3_S (блок 256) к нему
**неприменимы**. Подходят только форматы с блоком, кратным 2688: Q2_0 (64), Q4_0/Q5_0/Q8_0/IQ4_NL/MXFP4 (32), NVFP4 (64), Q1_0 (128).
Для up (вход 1024), shared, in_proj и out_proj ограничения нет.

## 3. Таблица оптимизаций Strata

Статусы: **как есть**, **с доработкой** (какой), **неприменимо** (почему).
Общее замечание: Strata одноархитектурная. `ModelGeometry` жёстко задаёт qwen4exp (`is_qsa_layer = l%4==3`, `check_all` требует
ровно 12 QSA-слоёв, hc=4, PLE), и композиция слоя (`layer.cpp`, `session.cpp`, `verify.cpp`, `mtp.cpp`) написана под неё.
«Тип архитектуры в загрузчике» на деле означает отдельную ветку композиции слоёв и графов. Переиспользуются ядра и подсистемы
вокруг: тир экспертов, CPU-пул, спекуляция, сэмплер, сервер.

### 3.1 Спекуляция и verify

| # | Оптимизация (где) | Статус | Что делать |
|---|---|---|---|
| 1 | MTP-драфтер на GPU, свои графы на шаг (`core/mtp.cpp`) | с доработкой | Переписать forward под Nemotron-MTP: eh_proj+2 нормы, NoPE GQA вместо QSA с hyper-connections, MoE top-22 с латентом. Все 512 MTP-экспертов держать в VRAM (1.48 GiB), как Strata держит свои 708 MB. Механика (захват графов на шаг, KV драфтера, prefill драфтера, chain) остаётся |
| 2 | Verify-окно: T токенов одним графом, побитно равно greedy (`core/verify.cpp`) | с доработкой | Новый граф окна под 88 блоков. **Откат SSM:** Strata уже решает это для GDN способом «окно не трогает состояние, `commit(n_keep)` проигрывает принятые шаги из сохранённых входов». Для Mamba-2 предлагаю то же как вариант A: хранить на позицию xBC после conv-входа и dt (~41 KB/поз/слой). Вариант B: снимок 160 MiB перед окном, ~0.1–0.2 мс на копию. Выбор по замеру, как в таске |
| 3 | Многотокенные ядра в окне (multi-column MMVQ exact, T-token GDN) | с доработкой | Нужен T-токенный Mamba-шаг (последовательно по T внутри ядра, состояние в регистрах/SMEM, без записи наружу) |
| 4 | Контроллер спекуляции, draft policy, `--spec-min-p` (`spec/controller.cpp`, `draft_policy.cpp`) | как есть | не зависит от модели |
| 5 | Prompt lookup / suffix drafter (`spec/suffix_drafter.cpp`) | как есть | — |
| 6 | Coupled drafts + Gumbel-max (`STRATA_SPEC_COUPLED/GUMBEL`) | как есть | — |
| 7 | Подмножество словаря для головы драфтера (`--draft-vocab`, `tools/draft_vocab.py`) | с доработкой | Пересобрать подмножества (en/cyrillic/CJK) под токенизатор Nemotron (131072). Голова драфтера из FP8 lm_head, а не из GGUF-head |
| 8 | Route-resident bias для MTP (`STRATA_ROUTE_RESIDENT_MTP`) | неприменимо пока | opt-in, по замерам Strata быстрее не стал. Не переносить |

### 3.2 Эксперты: размещение, кэш, CPU

| # | Оптимизация | Статус | Что делать |
|---|---|---|---|
| 9 | VRAM-кэш экспертов, заполнение по профилю, `--expert-cache auto` (`core/expert_cache.cpp`) | с доработкой | Размер блоба 2.95 MiB вместо ~1.75, 20,480 пар (слой, эксперт) против 24,576, top-22. Подсчёт бюджета VRAM: SSM-состояния и окна вместо PLE/GR |
| 10 | Профиль частот экспертов (`--dump-routing`, `tools/make_profile.py`, `--expert-profile-save`) | с доработкой | Формат профиля параметризовать числом слоёв MoE (40, а не 48), экспертов и k. Первый профиль снять на корпусе (код/чат/рус/англ), как Strata снимала свой |
| 11 | Адаптивный тир: обмены RAM↔VRAM по ходу разговора (`--adapt-every`, `--adapt-async`, ADAPT_OVERLAP, exchange rotation, DMA batch) | как есть | логика не зависит от модели, только от размера блоба |
| 12 | CPU-пул, считающий промахи «на месте» (`kernels/cpu/pool.cpp`) | как есть | пул и аффинити (`--pool-workers`, `--aux-cpus`) |
| 13 | CPU-ядра экспертов: IQ/Q2_0 AVX2/AVX-512, ggml-cpu dot (`kernels/cpu/*`) | с доработкой | Нужен путь NVFP4 + relu² без гейта. В vendored ggml есть `ggml_vec_dot_nvfp4_q8_0` (x86, AVX2). Для i7-14700KF (AVX2 + AVX-VNNI, **AVX-512 нет**) написать своё ядро LUT E2M1→int8 + VNNI по образцу `iq_avx2`. Сначала ggml-путь для корректности |
| 14 | Доля промахов по PCIe против CPU, авто-подбор (`--pcie-frac`, формула L/(L+1.85R)) | как есть | константу 1.85 перемерить на этой машине |
| 15 | Route tail skip: промахнувшийся эксперт с рангом ≥7 из 10 у всех токенов окна не считается (`STRATA_ROUTE_TAIL_SKIP`) | с доработкой | Порог для top-22 подобрать заново (KL-гейт как в Strata). При k=22 и scale 5.0 хвост может быть тяжелее. Включать только после замера KL |
| 16 | Router look-ahead, предвыборка следующего слоя (`STRATA_LOOKAHEAD*`, Foresight slots) | с доработкой, низкий приоритет | Роутер Nemotron смотрит на hidden, а MoE-блоки идут через Mamba/attention-блок, так что предсказание через 1 блок. Foresight в Strata по замерам не окупился |
| 17 | Low-RAM режимы (mmap/resident/budget, file tier, IO prefetch, arena mmap) | неприменимо | 59 GiB экспертов влезают в 128 GB. Код не трогать, не тестировать |
| 18 | Peer/remote experts, второй GPU, layer split, pipeline windows | неприменимо | одна карта |

### 3.3 Плотные слои, ядра, графы

| # | Оптимизация | Статус | Что делать |
|---|---|---|---|
| 19 | Native MMVQ для плотных проекций (ggml-типы, multi-column) (`cuda/native_mmvq.cu`) | с доработкой | FP8 E4M3 per-tensor в ggml нет. Варианты в п. 5.2. BF16-GEMV (`bf16_gemv.cu`) берём как есть для латента и роутера |
| 20 | Роутер top-10 softmax (`cuda/router_top10.cu`, `native_router.cu`) | с доработкой | Новое ядро: FP32 GEMV 512×4096, sigmoid + bias, top-22, нормировка, ×5. Окно (`moe_route_window`) по тому же образцу |
| 21 | Shared expert (`cuda/shared_expert.cu`), «shared early»: shared считается, пока хост ведёт CPU-пул | с доработкой | relu² без гейта и без скалярного sigmoid, NVFP4. Идея «shared early» переносится целиком |
| 22 | Grouped hit-ядра VRAM-экспертов (`s2_expert_grouped.cu`, `native_moe.cu`) | с доработкой | relu², два GEMV вместо трёх, вход 1024 (латент), NVFP4 W4A16 (decode) |
| 23 | Doorbell, сплит MoE на pre/post граф вокруг CPU-пула | как есть | схема `pre[l] = norm→route→publish→shared`, `post[l] = combine→fc2_latent→residual`. Между ними только fc1_latent на GPU |
| 24 | CUDA graphs на слой и на окно (`core/graph.cpp`, `session.cpp`) | с доработкой | захват новых графов, механизм тот же |
| 25 | PDL (programmatic dependent launch) | как есть | — |
| 26 | Thread-block clusters для выбора/argmax на sm_90+ (`STRATA_ARGMAX_MULTI`) | как есть | argmax/сэмплер по vocab 131072. QSA-часть неприменима |
| 27 | GDN-ядра: conv-шаг, fused preprocess, рекуррентный шаг, chunked WY prefill (`fused_gdn.cu`, `native_gdn*.cu`, `gdn_rec_parity.cu`) | неприменимо как ядра | Другая рекурренция. Структуру (conv-кольцо, fused preprocess, буферы `GdnBuffers`, zero_state) взять как шаблон для `MambaBuffers`. Для prefill есть готовый chunked SSD в `third_party/llama.cpp/ggml-cuda/ssm-scan.cu` + `ssm-conv.cu`, взять за основу |
| 28 | QSA: разреженный attention с индексером, block selection, decode/prompt attention (`qsa*.cu`, `native_qsa*.cu`) | неприменимо | У Nemotron плотный GQA. Нужен плотный flash-decode для head_dim 128, GQA 16, NoPE. Кандидаты: `native_flash_attn.cu` (сейчас диагностический, ≤256) или fattn из vendored ggml для prefill |
| 29 | RoPE/mRoPE/YaRN, rope scaling (`rope*.cu`, `rope_scaling.hpp`) | неприменимо | NoPE |
| 30 | Gated residual / hyper-connections hc=4 (`gr.cu`, `fused_gr.cu`, `native_gr_*`) | неприменимо | у Nemotron обычный residual |
| 31 | PLE: n-gram таблица 28.8 GB на SSD (`ngram/`, `ple*.cu`, `--ple-io`) | неприменимо | у Nemotron нет per-layer embeddings |
| 32 | Head (`core/native_head.cpp`), Q4-голова драфтера | с доработкой | lm_head FP8 → свой формат (см. 5.2), голова драфтера = подмножество строк |
| 33 | Сэмплер, logit bias, greedy argmax (`cuda/sampler.cu`, `logit_bias`) | как есть | — |
| 34 | Control vector / «experimental speed projection» (`cvec.cu`, `--cvec-mode project`) | с доработкой, позже | Механизм проще (один residual-поток). Сам вектор для Qwen, для Nemotron его нет. Аблитерация вне таски |

### 3.4 KV, prefill, кэш разговоров

| # | Оптимизация | Статус | Что делать |
|---|---|---|---|
| 35 | KV int8 / q4 с Адамаром / k8v4 (`kv_q8.cu`, `kv_q4.cu`) | с доработкой | Только 8 attention-слоёв, head_dim 128 вместо 256. KV и так 1 GiB @128K, int8 даёт +~500 MiB под экспертов. Сначала BF16, int8 потом по KL |
| 36 | KV streaming из RAM от 64K (`--kv-resident`, `kv_stream.cu`), `--kv-grow` | неприменимо до 128K | при 8 KiB/токен выгода ≤1 GiB. Вернуться при 1M |
| 37 | Чанковый prefill до 8192 (32768 opt-in), `--prefill auto`, займ слотов кэша под буферы (`prefill/prefill.cpp`) | с доработкой | Mamba-блоки через chunked SSD (п. 27), attention-блоки через dense flash, MoE как есть |
| 38 | Стриминг экспертов следующего слоя по PCIe во время attention текущего, ring в байтах (`STRATA_RING_BYTES`) | как есть | размер блоба другой, логика та же. Перекрывать можно Mamba-блоком, он дороже attention |
| 39 | MMQ ggml для экспертов в prefill, fused int8 TC (`moe_mmq.cu`, `moe_fused*.cu`) | с доработкой | В vendored ggml уже есть NVFP4 MMQ с **нативным FP4 MMA на Blackwell** (`mma.sync kind::mxf4nvf4 … ue4m3`, `mmq-config-blackwell.cuh`, `mmq-instance-nvfp4.cu`). Нужны relu² без гейта и учёт per-tensor scale. Это основа «GEMM для prefill» из этапа 5 |
| 40 | CPU share в коротких prefill-чанках (`STRATA_PREFILL_CPU_SHARE`, по умолчанию <1024) | с доработкой | нужен тот же CPU-путь NVFP4 (п. 13) |
| 41 | Префиксный кэш, conversation cache, message boundary, parking, snapshots (`core/conversation_*`) | с доработкой | Состояние разговора = KV 8 слоёв + **SSM 160 MiB + conv** на точку возобновления. Для GDN Strata это уже делает. Поменяются форматы образа и проверки размеров |
| 42 | Batching (`--batch`, batch groups) | с доработкой, позже | SSM-состояние на слот. Для одного пользователя не нужен |

### 3.5 Загрузка, платформа, сервер

| # | Оптимизация | Статус | Что делать |
|---|---|---|---|
| 43 | GGUF reader + «пак» (`artifact/gguf_reader`, `tools/iq_pack.py`, `native_experts.txt`) | с доработкой | см. п. 5.1 |
| 44 | Read-ahead при старте, pinned arena, cudaHostRegister слайсами | как есть | — |
| 45 | Watchdog, stall report, decode timing / GPU stages профили | как есть | этапы GPU переименовать под новые блоки |
| 46 | Сервер OpenAI/Anthropic, web app, MCP (`serve/`) | с доработкой | Шаблон чата Nemotron, reasoning-парсер (`ultra_v3_reasoning_parser.py`), формат tool calls. Наш локальный патч `reasoning-loop-wrap` проверить на новом шаблоне |
| 47 | HIP / SYCL / Intel / Strix Halo / старые GPU | неприменимо | только CUDA sm_120. Новые файлы под `if CUDA`, сборку HIP не ломать |
| 48 | Vision-энкодер (`strata-vision`, mmproj) | неприменимо сейчас | RADIO ViT-H, по таске позже |
| 49 | Parity-тесты по образцу `*_parity.cpp` + `ref/*.py` | как есть | тот же стиль для `mamba_parity`, `nemotron_moe_parity`, `attn_nope_parity` |

## 4. Решения на ревью

1. **Путь загрузки.** Предлагаю: NVFP4 safetensors → GGUF штатным конвертером vendored llama.cpp (уже умеет `nemotron_h` с латентом,
   MTP и переупаковку modelopt NVFP4 в `GGML_TYPE_NVFP4`) → пак Strata. Плюсы: работают `gguf_reader`, пак, `native_experts.txt` и
   ggml-ядра без нового загрузчика, а llama.cpp на том же файле даёт **исполняемый сквозной референс** для этапа 7 (весь BF16 не
   влезает, а этот влезает). Минус: FP8 он деквантует (п. 2). Альтернатива: свой safetensors-загрузчик, больше кода, полный контроль.
   BF16 для своих квантов в любом случае идёт через тот же конвертер с `--outtype` и imatrix.
2. **FP8-тензоры (in_proj, out_proj, qkvo, lm_head, всего 4.85 GiB).** (а) BF16: точно без потерь, но +4.85 GiB VRAM, это ~1,650
   экспертов меньше; (б) Q8_0 (`fp8_as_q8`): почти без потерь, размер как у FP8 + 6%, ядра Strata уже есть; (в) свой FP8
   per-tensor тип в паке + GEMV/MMA FP8 на sm_120: побитно как чекпоинт, но новое ядро. Предлагаю (б), с замером KL против (а)
   на одном слое.
3. **Референс для NVFP4.** vLLM считает W4A4 (активации тоже в NVFP4 через `input_scale`). Decode-GEMV естественнее делать
   W4A16. Это точнее референса, но не побитно. Предлагаю эталоном логитов взять W4A16-деквант в PyTorch (то, что «модель
   значит»), а W4A4 мерить как отклонение. FP4 MMA в prefill неизбежно W4A4.
4. **Откат SSM в MTP.** Вариант A (replay из сохранённых входов, как GDN в Strata) первым, вариант B (снимок) как A/B.
5. **Рабочее место.** Ветка `strata-nemotron` в отдельном worktree `~/llm/strata-nemotron` и своя сборка там. Продовый
   `~/llm/strata` (слоты swift15 / flash-next) не трогается, его локальные правки (`serve/server.py`, `local-patches/`) в ветку не
   попадают.
6. **Скачивание.** NVFP4 ~70 GiB сразу (этап 1). BF16 ~232 GiB (64 шарда) нужен для послойных референсов и своих квантов.
   Предлагаю качать шарды по требованию: послойные тесты требуют только шард с нужным слоем. Целиком только к этапу 5.
   На диске 1.5 TB свободно.

## 5. Что не перенесётся (предварительно)

QSA (разреженное внимание с индексером), RoPE/YaRN, hyper-connections/GR, PLE-таблица на SSD, low-RAM режимы, multi-GPU/peer,
HIP/SYCL. Все по одной причине: в Nemotron этого нет или на одной машине с 128 GB это не нужно (см. таблицу).
Отдельно: IQ2_XS/IQ3_XXS/IQ3_S/K-кванты для `down_proj` экспертов из-за длины строки 2688.

## 6. Итог ревью (2026-10-17) и проверки после него

Решения: (1) загрузка через конвертер llama.cpp → GGUF → пак; (2) FP8-тензоры (in/out_proj, qkvo, lm_head) квантовать в
**NVFP4** (−2.2 GiB VRAM против FP8, ≈ +750 экспертов в кэше), KL замерить против BF16 этих тензоров; (3) эталон логитов —
NVFP4-веса × BF16-активации, W4A4 — отдельным отклонением; (4) NVFP4 качать сразу, BF16 по шардам, плюс проверить IQ3_XXS;
(5) смотреть LiLiCorr.

**LiLiCorr — не вариант весов, а отдельный драфтер** (`LiLiCorrDraftModel`, vLLM `method dflash`): блочный DFlash-драфтер
(8 токенов параллельно, некаузальный, 5 слоёв dense GQA 32/8 с SWA 8192, вход — hidden целевой модели со слоёв 1/22/43/64/85),
плюс двухслойная голова корреляции top-8 кандидатов; 1.13B параметров + NVFP4-копия lm_head (0.54B). NVIDIA: acceptance length
4.74 (T=0) / 4.36 (T=1) при длине черновика 7 на SPEED-Bench, рекомендован для DGX Spark (тоже упор в полосу памяти).
Для нас — кандидат в этап 6 рядом с MTP. Риск: окно в 8 токенов поднимает объединение промахнувшихся экспертов (по замерам
Strata 3.05× одного токена уже при T=4), а DFlash2 на Maya был −15..−35%. Готовые GGUF: `ggml-org/...-GGUF` и
`unsloth/...-GGUF/LiLiCorr/` (1.2–1.3 GiB). Основа реализации — `~/llm/project-maya/src/core/glm_dflash.cu`.

**IQ3_XXS и готовые GGUF.** Есть готовые сборки, и все решают проблему 2688 одинаково — `down_proj` в 32/64-блочных типах:

| Сборка | Размер | up_exps | down_exps | плотные | KL к BF16* |
|---|---:|---|---|---|---:|
| unsloth/lmstudio NVFP4 | 75–78 GB | NVFP4 | NVFP4 | — | 0.0253–0.0256 |
| AtomicChat AD-IQ4_NL | 74.8 GB | — | IQ4_NL | Q8_0/Q6_K | 0.0133 |
| AtomicChat AD-IQ3_XXS | 65.3 GB | IQ3_XXS (34), IQ4_XS (6) | IQ4_NL / Q5_0 | Q8_0/Q6_K | 0.0352 |
| unsloth UD-IQ3_XXS | 59.8 GB | IQ3_XXS (25), IQ2_XS/IQ2_S/IQ4_XS | IQ4_NL (29), Q2_0 (11) | Q6_K/Q8_0 | 0.0670 |
| AtomicChat AD-Q2_0_L | 55.8 GB | — | Q2_0 на «тихих» блоках | Q8_0/Q6_K | 0.0722 |
| AtomicChat AD-Q2_0_S | 45.7 GB | IQ2_S (34) | Q2_0 (34) | Q6_K/Q8_0 | 0.1470 |

\* замеры AtomicChat (README репозитория, один стенд, логи в `AtomicChat/...-GGUF-metrics`), не наши.

Все эти типы у Strata уже есть (GSQ-RCO-паки: IQ3_XXS/IQ2_S/IQ4_XS gate/up, IQ4_NL/Q2_0 down; CPU-ядра AVX2 для IQ*,
ggml-cpu для остального), т.е. своё квантование из BF16 на этапе 5 можно начать с готовых файлов. Размер эксперта и
сколько поместится в ~19 GiB: NVFP4 2.95 MiB (~6,600, 32%); IQ3_XXS+IQ4_NL 2.48 MiB (~7,800, 38%); IQ3_XXS+Q2_0 1.74 MiB
(~11,200, 55%); IQ2_S+Q2_0 1.58 MiB (~12,300, 60%). Наблюдение: AD-IQ4_NL того же размера, что NVFP4, но с вдвое меньшим KL —
«родной» NVFP4 по качеству не лучший 4.5-битный вариант; его плюс — FP4 tensor cores в prefill.

Конвертер vendored llama.cpp регистрирует только `NemotronHForCausalLM`; обёртку VL (`NemotronH_Omni_Reasoning_V3`,
`llm_config`, префикс `language_model.`) нужно подать через shim-каталог (config = `llm_config`, симлинки шардов,
vision-тензоры пропустить) — проверить на этапе 1.
