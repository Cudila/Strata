# strata-nemotron: static review of decode, short increments and vision (2026-10-22)

Implementer-facing. Static reading of commit 2f285a82 (`strata-nemotron-review` = production tree). No GPU runs:
every ms figure below is an **estimate** with its basis stated; the baseline being measured in parallel
(`STRATA_NEMO_HOSTPROF=1`, `STRATA_NEMO_PROFILE=1`, `--bench`) decides the order. Line numbers are
`src/core/nemo_model.cpp` unless a file is named.

Production slot (`strata-nemotron-mix.json`): `--spec 5 --spec-min-p 0.8 --skip-w 0.4 --tail-skip 0 --prefetch 0
--evict lfu --prompt-skip-w 0.3 --prompt-skip-max 8192 --chunk 16384 --kv-base 65536`, sampling
`temperature 1.0, top_p 0.95, top_k 64`, vision encoder resident on the GPU (116 expert slots a layer).

Two facts that frame everything:
- With `--skip-w` > 0 the generated text depends on the expert-cache contents. "Bit-exact" below means: the same
  cache decisions and the same per-row arithmetic, so `--bench` ids (greedy and `--temp 1.0` with a seed) are
  identical to the baseline. Anything that changes eviction/prefetch decisions changes the text even when it is
  "exact" in principle.
- The slot's own log (`strata-nemotron-mix.log`, 19.10, ~110 requests) shows windows of **20–21 ms** in some runs and
  **29–38 ms** in a long series at 94% hit rate, at 3.3–4.3 tokens a window. The per-window budget is not stable
  enough to rank items from one number; take HOSTPROF's `draft / verify / rest` split first.

## 1. Decode window

### 1.1 Estimated budget of one window (Mw ≈ 4.7 rows: 1 + ~3.7 drafts)

| part | bytes / work | est. ms | basis |
|---|---|---:|---|
| dense verify (Mamba in/out NVFP4 2.47 GB, attn Q8 0.30, gate/fc1/fc2 ~1.0, shared NVFP4 0.99, lm_head Q8 0.57) | ~5.3 GB | 3.5–4.0 | tiled GEMV, M ≤ 8 ≈ M = 1, ~1.4 TB/s |
| routed experts in verify, **per (row, rank) entry** | 103 entries × 3.1 MB × 40 ≈ 12.8 GB | 6–9 | `nemo_experts_part` M = 1 per entry (see 1.2 #1); L2 may absorb part of the duplicates |
| MTP drafts (×3.7) | ~0.82 GB each, 70% of it the Q8 lm_head | 2.0–2.5 | eh 64 MB + q/o 64 MB BF16 + experts 68 MB + head 570 MB |
| miss copies on the critical path | 76 µs/expert at 41 GB/s, issued only after the layer's sync | 0–4 | unknown: needs copied-misses per layer per window |
| CPU sampling (temp 1.0) | per sampled row | 1.0–2.0 | see #3 |
| 40 MoE sync bubbles | per layer ~25–45 µs GPU idle | 1.0–1.8 | wake + book + first launch |
| Mamba snapshot / restore / replay | 165 MB D2D each way + 80+80 ops | 0.5–1.0 | see #6 |
| GPU-side launch gaps, tiny kernels | ~1100–1200 ops a window | 0.5–1.0 | ~1 µs gap each, norms/routers on 5 blocks |

Sum 15–25 ms: consistent with the 20 ms windows; the 30+ ms series needs the profile.

### 1.2 Ranked items

| # | where | now | should be | est. ms/window | exact? | risk | LOC |
|---|---|---|---|---:|---|---|---:|
| 1 | 1714–1869, `nemo_moe.cu:184–190`, `nemo_gemv.cu:1010` | verify window computes every (row t, rank j) entry as its own M = 1 GEMV: an expert routed by 4 rows is read 4 times (`nemo_gemv_many`, `nmat` = entries). ENGINE_V2 listed "experts in one pass over the union" as next; still not done | one pass over the distinct experts (`uniq` already exists at 1733), each matrix computes its ≤ 8 rows from a per-matrix (in-row, out-row) list; `gemv_kernel` already does MT rows, it needs a gathered row list instead of `xin + m*ldx` | **1.5–4** (if distinct/entries = 0.55–0.7) | yes: a row's sums do not depend on MT (LPR depends on cols only); test spec greedy == plain | medium (kernel) | ~250 |
| 2 | 2452–2454 | every draft runs the full 131K × 4096 Q8 lm_head (570 MB, ~0.4 ms) to get argmax + prob | draft-only head over a frequency-ordered vocab prefix (e.g. 32K rows, 140 MB, or the same rows permuted in place + id map) | 1.0–1.5 | output exact (verification unchanged); drafts change, acceptance may drop | needs owner OK ("MTP head stays") | ~120 |
| 3 | 2534–2545, `nemo_main.cpp:64–93,410` | temp > 0: per window row, synchronous **pageable** `cudaMemcpy` of 524 KB (legacy stream), then `sample()` allocates a 131K `idx` vector, `iota`, `partial_sort` to top_k (64 from the config; 1024 when unset) with an indirect comparator; GPU idle throughout | one kernel: per row top-k (value, id) of `vlogits` with id tie-break; one pinned D2H of Mw × k × 8 B, one sync; the host runs the same double arithmetic of `sample()` on the k candidates row by row | **1.0–2.0** (0.3–0.5 ms a row × 3–4 rows: copy 50–80 µs + alloc/iota 30–60 µs + partial_sort 0.2–0.4 ms) | yes except exactly tied logits at the k-th place (partial_sort's tie order is unspecified anyway); same rng draws | low | ~150 |
| 4 | 1808–1835, 1862–1867 | a copied miss starts only after the layer's host sync and lands 76 µs/expert later; the misses' half of `nemo_experts_part` waits on `cev`. Prefetch is off in production (`--prefetch 0`) | (a) prefetch only predicted experts with predicted weight ≥ τ (ENGINE_V2 "next", not done: 1916–1941 filters by rank only), (b) a CPU lane for misses | 0–4 (measure first) | changes cache → changes text; quality goes **up** (fewer skips) | medium | 40 / 300 |
| 5 | 1620–1664 | the D2H of ids is queued after fc1 (1626) and, with prefetch, after the next layer's norm/gate/router (1638–1645); the host waits on the whole stream; after the sync the GPU idles during the host's book + launch; the shared expert runs after the experts (2292–2296) | router → D2H ids → `cudaEventRecord`; then queue fc1, `sh_up` into `sh_h` (independent), predicted router; host `cudaEventSynchronize`. `sh_dn` stays after fc2, so the accumulation order into x is unchanged | 0.5–1.0 | yes | low | ~40 |
| 6 | 2514–2518, 2550–2562 | every window: 80 D2D copies (40 × 4 MiB state + 120 KiB conv = 165 MB) to `spec_state`; on any refusal (~70% of windows) 80 copies back + conv/scan replay of `a` rows | double buffer: the verify conv/scan read `state[cur]`, write `state[alt]` (kernels get separate in/out pointers); full accept → swap; partial → replay `a` rows from the untouched `cur`. Cheaper first step: one batched copy kernel over a pointer table instead of 80 memcpys | 0.4–0.8 (batched copy alone ~0.2) | yes | low–medium | 150 (60) |
| 7 | whole window | ~1100–1200 stream ops a window (40 Mamba × 6, 8 attn × ~8, 40 MoE × ~14, drafts ~28 each, snapshot 80, restore/replay ~160) | kernel fusion of rms_norm into the next GEMV's staging and of router + bias; CUDA graphs per segment (1.3) | 0.3–1.0 | fusion: per-row order must match | medium | 200+ |
| 8 | 1762–1789 | LFU `victim()` scans all slots per miss and calls `std::exp` per slot each time | per pass, compute each slot's decayed score once (same floats, ticks constant within the pass), then pick | 0.05–0.2 | yes (identical floats) | low | ~30 |
| 9 | 2365, 2469–2470 | draft chain: 2 syncs a draft, but the first (2365) finds the stream already drained; per draft the GPU idles only ~5–15 µs before the next launch (host launches ~28 ops in ~0.1 ms against ~0.6 ms of GPU work) | leave it | ≤ 0.1 | — | — | — |

### 1.3 The 2026-10-21 list, checked against the code

1. **40 host syncs in MoE routing**: confirmed (1664, one per MoE layer per forward). Cost refined: 40 × 25–45 µs ≈
   1–1.8 ms, of which #5 recovers half without restructuring. Removing them needs device-side planning (hit table
   in VRAM, meta built by a kernel as `nemo_resident_expert_ptrs` does for the MTP block, `nemo_ops.cu:208`), but the
   copied misses still need the host, so the GPU still waits on layers with a miss: the gain over #5 is ~0.5–1 ms.
2. **~700 launches, no CUDA graph, "−3 ms"**: rejected as stated. The count is ~1100–1200, but between syncs the
   host is ahead of the GPU (a segment between MoE layers is ~20–25 launches ≈ 0.1 ms host against ~0.3–0.5 ms GPU),
   so host launch cost is hidden; what is real is the post-sync restart (in #5) and the per-kernel GPU gap (#7).
   What prevents capture today: (a) the expert lists, their split into ready/waiting (`na`, `nb_`) and so the grids
   come from host data (1838–1867); (b) misses issue copies conditionally (1808–1835) and the overflow ring;
   (c) Mw varies per window (spec_min_p stop, 2477); (d) `pos_` enters kernel parameters (attention `p0`, split
   count 2529–2531, `img_overlay`); (e) `gemm` loops 8-row pieces (763). Minimal capturable sub-chains: per segment
   "weighted sum → fc2 → sh_up/sh_dn → next Mamba/attention block → next MoE rms/gate/router/fc1 → D2H ids" for
   each Mw (40 × 8 graphs), with `p0`/split counts read from device memory; and one draft step (M = 1) graph. Worth
   doing only after #1, #3, #5.
3. **CPU sampling with pageable copies**: confirmed and worse than stated (per-row 131K partial_sort, a fresh 512 KB
   vector each call): #3.
4. **Host-driven MTP chain, ~10 syncs**: refined: ~5 effective, ≤ 0.1 ms. The draft cost is the full-vocab head (#2).
5. **Mamba snapshot every window**: confirmed (2514–2518, 165 MB, 80 ops); #6.
6. **O(slots × misses) victim search**: confirmed, small; with `--evict lfu` it is `std::exp` per slot per miss; #8.

## 2. Short increments at long context (30–400 tokens onto 60K+)

Path: `run_serve` (`nemo_main.cpp:355–394`) → `restore()` (1508) → `forward(prompt[reused:boundary])` → `snapshot()`
(1486) → `forward(prompt[boundary:])` (the generation prompt, ~5 tokens) → first sample → spec windows.

Fixed costs that do not scale with the increment (estimates):

| cost | where | est. | fix |
|---|---|---:|---|
| the agent's own previous answer is re-read: the snapshot is at the last `<|im_end|>\n` **before** generation (`nemo_main.cpp:363–368`), so every turn prefills the previous turn's answer (and its kept reasoning: the template keeps reasoning after the last user message, `chat_template.jinja:165,200`) at short-prompt speed (~850–2650 tok/s) | `nemo_main.cpp:360–392` | 0.1–1 s a turn | a second checkpoint after generation, used when the new prompt's ids extend `prompt + generated[:-1]`; check offline first how often the re-rendered ids match (`STRATA_NEMO_DUMP_PROMPT`) |
| a second full model pass for the ~5-token tail after `snapshot()`: 88 blocks, 40 host syncs, MoE prompt path with its own miss copies via the ring | `nemo_main.cpp:383–393` | 10–30 | feed the tail through the decode path (T ≤ 8: LFU slots, copies stay cached), or capture the Mamba/conv state at row b inside one forward |
| dense weight conversion per chunk: for T ≥ 16 every NVFP4 dense matrix is re-converted tiled→N4 (`gemm`, 794–804: the 2-entry cache never hits across layers), Q8/BF16 → Q8N for attention and fc1/fc2 (816–819); ~5 GB read+write | 760–840 | ~5–7 | GEMM loaders that read the tiled layout (as the expert GEMM reads ggml rows) |
| attention for 9 ≤ T ≤ ~700 at long context: the prompt tensor-core kernel has no split-K, grid = ceil(T/8) × 2 KV heads (`nemo_attn.cu:519–526`): at T = 64 that is 16 blocks on 170 SMs streaming 60K+ keys | `nemo_attn.cu:513` | est. 3–8 at 64K, 15–30 at 256K, per chunk (8 layers + MTP) | split-K over key ranges for 8 < T ≤ ~512, scratch sized for it; measure with `nemo_attn_bench 65536 64` |
| pageable D2H in the prompt MoE: ids (1668), weights (1671), predicted ids/weights (1656–1657) into `std::vector`s: synchronous staging, 2–4 per layer | 1647–1674 | 2–4 | pinned buffers (they exist for decode: `h_dids`, `h_dw`) |
| 40 host syncs + host planning per chunk; `ExpertRows` and group lists rebuilt from vectors | 1673, 1949–2112 | 2–8 | — (inherent to host planning) |
| MTP prompt rows: two calls (pending row, then T−1 rows), each starting with `cudaStreamSynchronize` (2365); BF16 MTP matrices re-converted to Q8N per call when M ≥ 16 | 2319–2325 | ~0.5 | one call (copy `M.h` in front of the rows) |
| `restore()` / `snapshot()`: 80 D2D + sync; `snapshot` adds a synchronous pageable 512 KB logits copy (1505); the end of `forward` another (2346) | 1486–1529 | ~0.5–1 | batched copy kernel, pinned logits |
| prompt-path misses go to the ring (`pq_gen`, 475–484) and leave the cache afterwards; the first decode windows miss them again and skip them by weight | 464–489 | quality + a few ms | optional: let a short chunk's copied misses take LFU victims |
| GENI: all images of the conversation read and copied every turn (section 3) | `nemo_main.cpp:333–345` | 30–60 ms per resident picture | V2 |

### 2.1 Conversation reuse: one snapshot is too few

`run_serve` keeps one `snap_prompt` and one VRAM snapshot (`snap_state`, 165 MB). Any side request (a title, a
summarizer, a sub-agent) calls `reset()` (`nemo_main.cpp:375`, `nemo_model.cpp:1481`: `snap_pos_ = -1`), and a branch
before the last boundary (an edited message, a retry of an earlier turn) re-reads the whole context: 60K at ~9–11K
tok/s is 6–7 s, 256K ~35 s.

What a checkpoint needs: K/V are positional (cells are written at `pos`; `kv_append` at `pos_`), so rolling back
inside one conversation needs only `pos_` + the Mamba state + MTP pending state (`M.h`, 16 KB) + the logits row.
Mamba state: 40 × (4 MiB + 120 KiB) = 165 MB. In pinned host RAM: D2H/H2D at ~45 GB/s ≈ 4 ms each way.

Design:
- **Checkpoint ring in pinned RAM** (e.g. 32 × 165 MB = 5.3 GB; the machine has ~60 GB free beside the 63 GB arena):
  at every message boundary and every 16K-chunk boundary of a prompt, plus after generation (2. row 1). Keyed by
  (position, hash of ids[0:pos], image keys). Reuse = the deepest checkpoint whose ids are a prefix of the new
  prompt; a branch at position p invalidates checkpoints > p (their K/V cells get overwritten).
- **Parking** a conversation on `reset()`: K/V cells [0, pos) of the 9 owners (8 attention + MTP; per cell
  9 × 2 heads × (128 + 4 + 256) B = 6,984 B → 60K: 458 MB, 256K: 1.8 GB) + its last checkpoint, D2H in ~10 ms
  (60K) / ~40 ms (256K); un-parking the same. Pool sized by `--park-mib`, LRU. A side request then costs ~20 ms of
  copies instead of a 6 s re-read. Strata's flash-next has a parking path (`conversation_*` in `src/core`) to
  borrow from; per the 21.10 review its parking is still pageable there — use pinned from the start.
- Engine-only (the engine already sees full prompts); the server needs no change. ~300 LOC; exact (copies only).

## 3. Vision pipeline

Request → `Service.prepare` (`serve/server.py:3961–4071`) → `Vision.encode` (2484) → `strata-vision` ENC →
`.sve` file → combined file → `GENI` → `read_sve` → `set_image_rows` → `img_overlay` per chunk (main + MTP).

| step | code | now | est. cost (1920×1080 PNG screenshot → 2040 tokens; 12 MP JPEG photo → 3300 tokens) |
|---|---|---|---|
| base64 / download | server.py 2405–2412 | fine | ~ms |
| `normalize` | 2440–2471 | JPEG: PIL opens the header for EXIF only; PNG/BMP/GIF pass; **WebP etc.: full PIL decode, then PNG encode (zlib)** | WebP 1080p: 50–150 ms of PNG encode, then stb decodes it again |
| write `.img`, ENC round trip | 2490–2500 | files in the vision dir (`mkdtemp`, /tmp) | ~1 ms |
| encode takes the request FIFO turn | 3990–4002 | serial with the engine (an earlier concurrent run hung) | no overlap at all |
| stb decode | strata_vision.cpp:274 | single-threaded; stb's JPEG is slow | PNG 1080p 40–60 ms; 12 MP JPEG 120–200 ms |
| `nvidia_resize` | strata_vision.cpp:83–131 | single-threaded, double precision, both axes always run (even when an axis keeps its length), horizontal pass walks columns, 146 MB float copy of a 12 MP input | 1080p (1080→1088 only): 60–90 ms; 12 MP → 2112×1600: 250–400 ms |
| mtmd preprocess | mtmd-image.cpp:770–791, 52–55 | the size matches → buffer copy; then f32 normalize on CPU; pageable H2D of the f32 image | 10–20 ms |
| graph build + sched alloc per image | clip.cpp:4755–4760 | rebuilt every image (sizes differ); buffers reserved by the 2048² warm-up | 1–3 ms |
| RADIO-H on CUDA | nemotron-omni.cpp, clip.cpp:787–800 | BF16 weights, FA **enabled**: `cp.warmup = false` (strata_vision.cpp:201) means the first encode (strata-vision's own 2048² warm-up) runs clip's `warmup()` (clip.cpp:4746 → 3928), which resolves AUTO → ENABLED when CUDA supports D = 80 (it does; a WARN line would say otherwise) | FLOPs: 2040 tokens ≈ 21 TFLOP, 3300 tokens ≈ 46 TFLOP; at the measured ~100 TFLOP/s → ~210 / ~460 ms |
| projector | build_ffn | RMSNorm → 5120×20480 → relu² → 20480×4096 | 3300 tokens: ~1.25 TFLOP, 6–10 ms |
| output | strata_vision.cpp:311–319 | D2H (pageable) + `fwrite` of n × 16 KB | 33–54 MB: 10–25 ms |
| combined file | server.py 4065–4071, 127–133 | **every request re-copies the `.sve` of every picture in the conversation** into one file | 30–60 ms per resident picture per turn |
| engine read | nemo_main.cpp:97–126, 333–345; nemo_model.cpp:882–910 | reads the whole file, then memcpys **all** rows into pinned memory (grows with `cudaMallocHost`), before the reuse check | ~10–20 ms per picture per turn |
| rows into the model | 395–402, 1582, 2370 | pinned H2D per chunk, main + MTP | negligible |
| text prefill of the image tokens | — | 2040–3300 rows at short-prompt speed (2K warm: 0.77 s) | **the largest part of TTFT**: 0.7–1.3 s |

So the "36–463 ms" in VISION_MMPROJ.md is strata-vision's `OK … ms` (t0 at 273 to the write at 320): it includes
decode, resize, preprocessing and the file write, not only the GPU. For a 12 MP photo the CPU part is about as large
as the GPU part.

What can be cut, by gain/effort:
- **V1** resize: skip an axis whose length is unchanged (with scale 1 the taps are cubic(0) = 1, cubic(±1) =
  cubic(2) = 0: an exact identity) and split both passes over 8–16 threads by output rows/columns. Bit-exact.
  −40…−80 ms (screenshot), −200…−350 ms (photo). ~60 LOC.
- **V2** GENI images: send per-picture `.sve` paths (they are cached by hash) instead of a combined file, and let
  the engine read only runs at positions ≥ the reused prefix (decide reuse first, then load). Bit-exact. −40…−80 ms
  per resident picture per turn in agent sessions with screenshots. ~120 LOC (server + engine).
- **V3** `normalize`: write BMP (or PPM) instead of PNG for converted formats. Bit-exact pixels. −50…−150 ms per
  WebP. ~5 LOC.
- **V4** decode JPEG with libjpeg-turbo (PIL already has it) and hand strata-vision raw RGB. Pixels change by ±1
  LSB against stb (NVIDIA's processor uses PIL, so this moves *towards* the reference); `vision_check.py` cos ≥ 0.998.
  −80…−150 ms on photos.
- **V5** elastic encoder: the server already has `"elastic": true` (2318–2324) but the engine has no `VRAM`
  command. Implement it with `band_drop`/`band_add` (279–322). +24 slots a layer while no picture was seen for
  `elastic_idle_s`; the first picture after idle pays ~1–2 s of encoder start. ~120 LOC.
- **V6** F16 mmproj: ggml-cuda runs F16 weights with FP16 accumulation (2× BF16's FP32-accum rate on GeForce
  Blackwell) for the linear layers (~50% of the FLOPs at 2K tokens, ~35% at 3.3K). Lossy; RADIO has high-norm
  register tokens, so check for overflow; `vision_check.py` cos ≥ 0.998 on all four test images. est. −15…−30% GPU
  encode.
- **V7** VRAM (2.9 GB): weights 1.68 GB (842M BF16), a second CUDA context ~0.4–0.5 GB, compute buffer reserved
  for the 2048² warm-up ~0.6–0.8 GB (est.). `max_tokens` in the vision config shrinks the buffer (and the
  prefill, which is the larger TTFT term) at a quality cost. In-engine encoding would drop the second context and
  could borrow the chunk scratch / prompt ring (idle while the encode holds the FIFO turn) as work memory, and could
  stream layer weights from pinned RAM (1.7 GB at 41 GB/s = 41 ms, hidden behind 200–460 ms of compute): ~2.5 GB
  back to the expert cache. Large (ggml inside the engine, or a native RADIO port); design only.
- **V8** overlap the encode with the prompt prefix: needs the image's token count before the encode (port
  `nvidia_tile` to Python: it depends only on the image size) and an engine that waits for a run's rows when its
  chunk reaches them. Only pays when the prefix is not reused (first turn); depends on V7 or on solving the
  concurrent-GPU hang that made the encode take the FIFO turn.
- Per-image graph rebuild, pos-embed interpolation (an 84 MB permute+interpolate per image), batching tiles
  (Nemotron uses one tile per image): each ≤ a few ms — not worth it.

## 4. Task list

Order by gain/effort. Every task: an env switch, default off until validated; then a default flip in its own change.
"ids identical" = `strata-nemotron --bench` on the 4 standard prompts, greedy and `--temp 1.0` (seed 1), ids equal
to a same-day baseline run, plus spec-greedy == plain-decode 300/300.

### Measurements first (no engine change)

| task | what | who | LOC |
|---|---|---|---:|
| M0 | `STRATA_NEMO_ROUTE_TRACE` on a 2K-token chat with spec 5; script: per verify pass (T > 1) distinct experts / entries, per layer; copied misses per window from `--bench` counters | Qwen (script) | ~80 |
| M1 | `STRATA_NEMO_HOSTPROF=1`: draft / verify / rest per window, book / launch per MoE layer, at temp 0 and 1.0 | baseline agent | 0 |
| M2 | `nemo_attn_bench 65536 64`, `… 262144 64` vs `… 8`: the no-split cost of short chunks | baseline agent | 0 |
| M3 | offline: how often the next turn's ids extend `prompt + generated[:-1]` (agent logs with `STRATA_NEMO_DUMP_PROMPT`) | Qwen (script) | ~60 |

### Bit-exact

| task | env | gain (est.) | test | deps | LOC |
|---|---|---|---|---|---:|
| E1 GPU top-k + one pinned candidate copy for sampled verify rows (#3) | `STRATA_NEMO_GPU_TOPK` | 1–2 ms/window at temp > 0 (5–10%) | ids identical at `--temp 1.0` (report tie count) | — | 150 |
| E2 event-based ids wait; fc1, sh_up, predicted router queued behind it (#5) | `STRATA_NEMO_SYNC_EV` | 0.5–1 ms/window | ids identical; `moe_parity` | — | 40 |
| E3 union experts in verify windows (#1) | `STRATA_NEMO_UNION` | 1.5–4 ms/window | ids identical; `nemo_gemm_parity` extended to gathered rows | M0 (go if distinct/entries ≤ 0.75) | 250 |
| E4 batched Mamba snapshot/restore copy kernel, then double buffer (#6) | `STRATA_NEMO_SSM_DB` | 0.2 → 0.4–0.8 ms/window | `mamba_parity`; ids identical | — | 60 + 150 |
| E5 LFU per-pass scores (#8) | (none needed: identical floats; keep `STRATA_NEMO_LFU_FAST` for the A/B) | 0.05–0.2 ms/window | ids identical | — | 30 (Qwen) |
| E6 pinned buffers for prompt-path D2H and the logits copies (1505, 1656–1671, 2346) | `STRATA_NEMO_PINNED_RB` | 2–4 ms per short chunk | PPL bit-identical, ids identical | — | 40 (Qwen) |
| E7 checkpoint ring in pinned RAM + parking on reset (2.1) | `STRATA_NEMO_CKPT=N`, `--park-mib` | side request / branch at 60K: 6 s → ~20 ms | resumed generation ids == uninterrupted run; PPL unchanged | — | 300 |
| E8 post-generation checkpoint (2, row 1) | `STRATA_NEMO_CKPT_GEN` | 0.1–1 s per agent turn | ids identical to a cold run of the same prompt (Mamba state is exact) | M3, E7 | 80 |
| V1 resize: identity-axis skip + threads | `STRATA_VISION_RESIZE_MT` | 40–350 ms per picture | `.sve` byte-identical to baseline | — | 60 |
| V2 per-picture files, engine loads only unreused runs | `STRATA_NEMO_GENI_FILES` | 40–80 ms per resident picture per turn | `vision_check.py`; the 7 end-to-end cases of VISION_MMPROJ.md | — | 120 |
| V3 BMP instead of PNG in `normalize` | — | 50–150 ms per WebP | `.sve` identical | — | 5 (Qwen) |
| V5 engine `VRAM` command → elastic encoder | config `elastic` | +24 slots a layer when idle | text PPL identical with/without; image cases | — | 120 |

### Numerically equivalent but not bit-identical (prompt path)

| task | env | gain | test / expected PPL | LOC |
|---|---|---|---|---:|
| P1 split-K prompt attention for 8 < T ≤ 512 at ≥ 16K keys (2) | `STRATA_NEMO_ATTN_PSPLIT` | est. 3–8 ms (64K) to 15–30 ms (256K) per short chunk | `attn_parity`; PPL 40 × 1024 ΔNLL within ±0.002 (summation order only) | 200 |
| P2 tail (≤ 8 tokens after the snapshot) through the decode path | `STRATA_NEMO_TAIL_DECODE` | 10–30 ms per turn | PPL n/a (template tokens); ids of a fixed chat differ only after the tail; check 25-prompt `api_compare.py` | 60 |
| P3 W4A4/W8A8 loaders that read the tiled layout (no per-chunk conversion) | `STRATA_NEMO_TILED_A4` | 5–7 ms per chunk | `nemo_fp4_parity`, PPL identical if the loader reproduces N4 exactly (then it is bit-exact) | 250 |

### Lossy

| task | gain | expected cost | rule check |
|---|---|---|---|
| L1 weight-filtered prefetch (#4a): prefetch predicted experts with predicted weight ≥ τ, on the copy stream one layer ahead | 0–4 ms/window (hides miss copies) | none: PPL should **fall** from ×1.091 toward ×1.008 (fewer skipped misses) | measure `--ppl-decode` 4 × 1024 and `--bench`; take it if tok/s ≥ today |
| L2 draft head over a vocab prefix (#2) | 1–1.5 ms/window | output exact; acceptance −0…−3% (est.) | owner sign-off (MTP head); measure tokens/window on 4 prompts |
| V4 libjpeg-turbo decode | 80–150 ms per photo | cos vs reference ≥ 0.998 (likely closer) | — |
| V6 F16 mmproj | 15–30% of GPU encode | cos ≥ 0.998 on 4 images; watch for inf | — |

Dependencies: M0 → E3; M1 → whether L1 or E3 comes first; M3 → E8; E7 → E8; V5 before V7.

## 5. Docs that disagree with the code or config

- OPTIMIZATION_NOTES §5 says "the slot runs `--skip-w 0.4 --prefetch 1`" (PPL ×1.008). `strata-nemotron-mix.json`
  runs `--prefetch 0`: the ×1.091 row. ENGINE_V2's "skip-w 0.4: PPL ×1.005" is the same setting measured differently;
  §5's decode-path measurement is the one that applies to the slot.
- ENGINE_V3 "the shared expert is queued before the router syncs": prompt path only (1627–1635); in decode it runs
  after the experts (2292–2296).
- OPTIMIZATION_NOTES "anything copied on every step has to go through pinned memory": the prompt MoE still reads ids,
  weights and predicted ids into pageable vectors (1656–1671); `spec_step`'s sampled path (2539), `snapshot()` (1505)
  and `forward`'s logits (2346) are pageable too.
- VISION_MMPROJ "encoding on the GPU takes 36–463 ms": the figure is the whole ENC (CPU decode + resize + file
  write included).
- strata_vision.cpp:202–206 speaks of "head size 72 in this encoder": RADIO-H's head is 80 (1280/16); the comment comes
  from the Qwen tower.
- The 2026-10-21 review's "CUDA graph −3 ms" and "~10 syncs in the draft chain" do not hold (1.3).
