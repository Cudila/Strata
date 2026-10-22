# strata-nemotron review (2026-10-22) — engine + vision, static reading against the baseline

Scope: the production slot `nemotron-3.5-super` (tree `~/llm/strata-nemotron-main` @ 2f285a82, fast preset
`--spec 5 --spec-min-p 0.8 --skip-w 0.4 --prefetch 0 --evict lfu`, 1M context, vision on the GPU).
Baseline numbers: `scratchpad/nemo-review/BASELINE.md` (RTX 5090, 32K warm decode, 93 windows).
This note is for implementers. Line numbers refer to the review worktree (identical to production).

## 0. The window as measured

| | ms | share |
|---|---:|---:|
| draft (mtp_draft, up to 5 steps) | 2.2 | 12 % |
| verify forward, of which: | 15.2 | 84 % |
| — experts (gemv_kernel, one (t,j) entry per launch block) | 7.9 | 43 % |
| — Mamba (40 blocks) | 3.1 | 17 % |
| — dense + router | 1.8 | 10 % |
| — fc2 + shared | 1.5 | 8 % |
| — attention (8 blocks) | 0.8 | 4 % |
| — head | 0.4 | 2 % |
| — ids sync (host idle, measured by PROFILE) | 0.26 | 1 % |
| rest (pick, rollback, MTP entries) | 0.6 | 3 % |
| **window** | **18.2** | 3.2–4.4 tokens → 177–206 tok/s |

nsys: 926 kernel launches, 50 `cudaStreamSynchronize`, 324 `cudaMemcpyAsync` per window; the host sits in
`cudaStreamSynchronize` 85 % of the time; the **GPU is idle 12.8 % of the window (~2.3 ms)**.
So the window is ~87 % GPU-busy and that GPU time is mostly bytes: the experts' weights (one read per (t,j)
entry), the dense matrices and the lm_head.

Correction to the earlier static guess (report §5, 21.10): the host is NOT the bottleneck in the fast preset.
Host work (launching ~900 kernels, bookkeeping ~20 µs a MoE layer) hides behind the GPU except at the 50 drains.
A CUDA graph would save almost nothing; removing the drains saves at most the 2.3 ms of idle.

## 1. Decode window — ranked

### D1. Expert weights are read once per (t,j) entry, not once per expert  — est. −1.0…−2.0 ms, bit-exact
`nemo_model.cpp:1859-1867` → `nemo_experts_part` → `nemo_gemv_many` (`nemo_gemv.cu:1010`): `dispatch<D>(…, M=1,
nmat=n)` — every entry is its own matrix with M=1. A verify window of 5–6 rows × 22 = 110–132 entries hits
~80–95 distinct experts: 20–40 % of the entries re-read an expert's 2.75 MB (NVFP4 up+down). L2 (96 MB on
GB202) catches some of it only when the duplicate entries run close in time; they are 22 apart in the row-major
list. Measure first (round 2, item D), then: group the entries by expert on the host (the list is built there
already, `nemo_model.cpp:1839-1851`), give each expert its M ≤ 8 rows and run the multi-row GEMV (the dense
path's `tc_gemv`/`gemv_kernel<MT>` already computes 8 rows "for the price of one" when the layout is tiled —
the slots hold ggml rows, so the ggml-row kernel needs an M>1 variant with the x rows gathered by index).
Acceptance: `moe_parity` + the bench ids (`--bench … --spec 5` prints the generated ids; an exact change keeps them).

### D2. One drain per MoE layer with nothing in flight — est. −1.0…−1.8 ms, bit-exact
`nemo_model.cpp:1660-1664`: router → D2H ids (+weights) → `cudaStreamSynchronize` → host books → launches.
At the sync the queue is empty, so the GPU idles for the drain + bookkeeping + first-launch latency (~20 µs book
+ ~25 µs drain/ramp, ×40 = ~1.8 ms; PROFILE shows 0.26 ms for "ids sync" alone but it cannot see the ramp).
Two cheap moves give the GPU work to chew while the host books:
- (a) issue the **shared expert before the sync** in decode as the prompt path already does (`:1627-1635`
  vs `:2292-2296`). Order of the two residual adds changes (x += shared, then += fc2 instead of the reverse):
  fp-add order, not bit-identical to today but identical to the prompt path's order. ~5 lines.
- (b) **hits resolved on the device**: keep a device mirror of `slot_of` per layer (NE int32; host patches
  the few changed entries with a tiny H2D after each pass). A kernel builds the "ready" part of the meta
  (pointers, scales, idx) from ids + mirror + weights (skip rule `bw < skip_w` is computable there: max weight
  per expert over the rows) and the hits' `nemo_experts_part` is launched **before** the sync. The host then
  waits only for ids, books the misses (copies on `cst`, LFU/LRU) and launches the "waiting" part as now.
  In the fast preset most misses are skipped, so the waiting part is often empty and the drain disappears
  behind the hits' compute (~0.15 ms a layer, more than the bookkeeping). ~250 lines; bit-exact.
  This is the first half of the Strata doorbell scheme and does not need a graph.

### D3. LFU victim search recomputes `exp()` per slot per miss — est. −0.1…−0.3 ms, bit-exact
`nemo_model.cpp:1762-1783`: for every miss, a loop over all slots with `std::exp(ld * (ticks - score_at))`.
Decayed scores change only with `ticks`, constant within a pass: decay once per pass into a scratch vector
(116–140 exps a layer), then victims are a min-scan or a small heap. Part of the "book" 20 µs.

### D4. Sampled path: full-vocab pageable copy + CPU partial_sort per window row — est. −1…−4 ms **when sampling**
`nemo_model.cpp:2536-2545` + `nemo_main.cpp:64-93`: for each kept row, `cudaMemcpy` (sync, pageable) of 131072
floats (512 KB), `std::iota` over 131K, `partial_sort` to top-k (64, or 1024 when top_k=0), softmax, draw.
~1 ms a row on this CPU, 3–4 rows a window → 3–4 ms, i.e. up to +20 % of the window. The baseline ran
temperature 0 and never saw it; the production slot runs temperature 1.0 (round 2, item A measures it).
Fix: one kernel over all Mw rows of `vlogits`: top-k (k = effective top_k, ≤ 1024) values + ids per row into a
pinned buffer (one async D2H of Mw × k × 8 B), then the SAME CPU code (temperature, top-p, min-p, draw) over
the k candidates. The draw is identical to today's except for ties at the k-th logit. Also for the first token of
a request (`nemo_main.cpp:411`) and the non-spec path. ~150 lines.

### D5. Mamba state snapshot every window — est. −0.2…−0.4 ms, bit-exact
`spec_step` `:2514-2518` copies 40 × 4 MB state (+conv) D2D before every window (160 MB, 80 memcpys), and
on a partial accept copies them back (`:2555-2558`) and replays. Instead run the window's scan **out of place**:
`mamba_ssm_scan` reads `b.state` and writes `spec_state[i]`; on a full accept swap the two pointers, on a partial
accept replay `a` rows in place from the untouched `b.state` (as today). Needs an output-state argument on the
scan kernel (`mamba.cu:113/260`). The same for conv. 80 memcpys and 160–320 MB of D2D traffic leave the window.

### D6. The draft chain: 2 drains per draft, and the lm_head read 5 times — est. −0.5…−1.2 ms
`mtp_rows` `:2365` starts with `cudaStreamSynchronize(st)` only to protect `M.stage` (pinned) from a previous
H2D still in flight — replace with an event recorded after that H2D (`cudaEventSynchronize`, usually already
passed). `mtp_draft` `:2469-2470` drains to read the token: inherent while the chain is host-driven. The
device-driven chain (argmax → `d_tok`; the next draft's embedding gather reads the row by index straight from the
page-locked `I.embd` through its device pointer; the min-p stop as a device flag every kernel checks and
exits on) turns 10 drains into 1 — ~0.3 ms — ~300 lines, bit-exact; lower priority than D1/D2.
Bytes: each draft runs the full lm_head (Q8_0 131072×4096 = 537 MB → ~0.3 ms) and the verify window reads it
once more: 6 × 0.3 = 1.8 ms a window (10 %) is lm_head traffic. Options, **lossy for acceptance only** (the
verify head stays Q8_0, so generated tokens are unchanged for greedy; for sampling the draws are matched
against the main model's distribution either way): an NVFP4 copy of lm_head for the drafts (+268 MB VRAM,
−0.15 ms a draft) and/or a truncated draft vocabulary (the 32K most frequent ids: −0.22 ms a draft). The user's
"MTP stays stock" rule was about Strata's MTP; here the head is ours — the owner decides.

### D7. Launch count (926) — not worth a graph
Mamba: 6 launches × 40; attention 6 × 8; MoE ~12 × 40; norms. With D2 the host runs ahead and these hide
behind GPU time. A fused "conv + scan + gated-norm for T ≤ 8" kernel would drop 80 launches (~0.2 ms) — later.

Sum of D1–D6 at the fast preset: −3…−6 ms of 18.2 (−16…−33 %) for greedy; plus the sampled-path item when
the client samples. Honest ceiling: the GPU-busy part is ~15.8 ms of bytes at ~75 % of peak bandwidth; the
window will not go below ~12–13 ms with these weights and MTP head.

## 2. Short increments at long context (agentic TTFT)

Baseline: 30 fresh tokens on 75K → 232 ms; the fit is **281 ms fixed + 0.139 ms/token**. The fixed part is the
prompt path's per-layer structure for a chunk of T ≤ 8192 rows (`:1619` pskip, `:1647` ppred):
1. per MoE layer: router + fc1 + shared, D2H ids and weights, `cudaStreamSynchronize` (`:1673`), host plan
   (`rows_of.build` over 512 experts, groups, lists), `pq_gen` for the layer (+ the next layer's predicted misses),
   copies through the ring on `cst`, grouped GEMMs waiting on `pcopy`;
2. the misses: with `--prompt-skip-w 0.3` a miss is copied only when its largest routing weight over the chunk is
   ≥ 0.3; what remains is PCIe time at ~41 GB/s (2.75 MB an expert) — round 2 item B measures bytes/chunk;
3. after the prompt: `snapshot()` (`:1486-1506`: 160 MB D2D + sync + a **pageable** 512 KB D2H of the logits),
   `mtp_rows` for the chunk's rows, and `restore()` at the next request (160 MB D2D + sync).
Candidates once B's numbers are in: the pageable logits copy in snapshot (pinned: trivial); the prompt-path
drain per layer (same shape as D2(b): resolve hits on the device, book the misses while the hits compute);
the per-layer planning (`plan` ms from HOSTPROF).

### E1. Conversation reuse is a single snapshot — the big e2e item
`nemo_main.cpp:260,355-376,383-391`: one `snap_prompt`; a request continues only if the previous prompt (up to
its last complete message) is a prefix of the new one. Any side request (another conversation) or any branch
before that boundary → the whole context is read again: **78K tokens = 7.7 s** (baseline D). With Hermes' per-turn
background review this is every user turn (the review's requests move the snapshot past the user's branch point).
Two pieces, exact:
- **Checkpoints inside a conversation** (~200 lines): keep N (e.g. 6) snapshots = Mamba state + conv (40 × 4 MB +
  small) + `mtp.h` + pending flags + `pos` + the prompt ids up to that point, in page-locked host RAM (D2H 160 MB
  ≈ 6 ms). Take one at every chunk boundary of a long read and at the message boundary. On a request, pick the
  longest checkpoint whose ids are a prefix of the new prompt. The K/V pages are positional: cells beyond the
  checkpoint are simply overwritten, so no K/V copy is needed **as long as no other conversation ran in between**
  (a generation counter on the K/V guards that).
- **Parking other conversations** (~300–400 lines): when a different conversation arrives, park the current one:
  its checkpoint + the K/V cells [0, pos) of the 9 K/V owners (≈ 7 KB a token: 60K = 0.4 GB, 300K = 2 GB; H2D
  back at ~26 GB/s = 15–80 ms). 4 slots in host RAM is 8–10 GB here. This is Strata's `--conversation-cache`;
  it turns the 7.7 s into ~0.1 s.
Order: checkpoints first (they alone fix the branch/review case), parking second.

## 3. Vision

Baseline: encode 22 ms (300 tokens) … 470 ms (3311 tokens) standalone; cold TTFT with the picture 515 … 1371 ms;
image tokens prefill at 0.8–4.3K tok/s vs ~10K for text; the encoder holds 2886 MiB.

Pipeline, request → model rows:
1. `server.py` `Vision.encode` (`:2484`): base64 decode, `normalize` (PIL opens a JPEG just to read EXIF; PNG/BMP/GIF
   pass through), sha256, write `<key>.img` to the temp dir, `ENC` line, wait, then `read_sve` in the engine.
2. `strata_vision.cpp`: `mtmd_helper_bitmap_init_from_file` = **stb_image decode** (PNG 1920×1080 ≈ 30–60 ms,
   single thread); `nvidia_resize` (`:83-131`) = **two passes of antialiased bicubic in double, scalar, one thread**
   (≈ 2 × W×H×3×taps MACs: ~60–100 ms at 1920×1088, ~150 ms at 2560×1440 — likely the largest CPU piece);
   `mtmd_tokenize` = clip's own CPU preprocessing (normalize mean/std, to f32 planes); `mtmd_encode_chunk` = the
   RADIO graph on CUDA (13312 patch tokens at the largest tile: ~17 TFLOP of linears + ~29 TFLOP of attention
   → ~200–250 ms is real GPU work there; at 2048 patches ~20 ms) + projector; `fwrite` of n × 4096 floats
   (3311 tokens = 54 MB) to the temp dir.
3. engine: `read_sve` (fread 54 MB into a vector), `set_image_rows` (sync + `cudaMallocHost` re-pin when the
   buffer grows + memcpy), `img_overlay` H2D per chunk.
4. then the prompt itself: a 300–3300-token chunk is in the "1K–8K prompts are bound by ~1 s of expert copies"
   regime (OPTIMIZATION_NOTES) — that is the 0.8–4.3K tok/s, not a vision cost.
Round 2 item C splits step 2 by stage. Expected levers, in order:
- **V1. `nvidia_resize` multithreaded** (rows/columns are independent; keep double so the output is bit-identical):
  −50…−130 ms a large picture, ~40 lines. Then float + SIMD if still visible (not bit-identical, within the
  vision_check tolerance).
- **V2. decode**: stb PNG is slow; a JPEG client is already 2× faster. libjpeg-turbo/libpng or a decode on the
  Python side with Pillow-SIMD and raw RGB handed over (the `.img` file could be raw RGB + header: no second
  decode). −20…−50 ms.
- **V3. file round trips**: write the `.sve` as BF16 (half the bytes; the engine converts to f32 on the GPU) or
  put both files in `/dev/shm` (the server's `work_dir` prefers the system temp dir — check it is tmpfs here);
  pin `img_host` once at the maximum size instead of re-pinning on growth. −10…−40 ms at 3300 tokens.
- **V4. the ViT on the GPU**: confirm flash attention is on (clip `flash_attn_type` auto → on when the backend
  supports it, `clip.cpp:3928`); try a Q8_0 mmproj (ggml's int8 MMQ on the linears, −0.85 GB VRAM; the attention
  is unchanged; check `vision_check.py` cos ≥ 0.998). Lossy but small; −20…−30 % of the GPU part.
- **V5. overlap**: encode runs before the engine sees the request. The text before the picture (system prompt,
  history) is in the snapshot already, so the gain is bounded by the encode time of the LAST picture; a `GENI`
  that names a file "to be written" and an engine that waits for it before the image's chunk would hide the
  encode behind the prompt read up to the image. Only worth it after V1–V3.
- **V6. VRAM**: 2886 MiB = 1.7 GB BF16 weights + ~1.2 GB compute/FA buffers sized by the 2048² warm-up. Q8_0
  mmproj (V4) gives back 0.85 GB; the server's `"elastic": true` (encoder up on demand, down after idle, the
  engine keeps `elastic_mib` free) gives the expert cache the whole 2.9 GB (+24 slots a layer, 116 → 140) for
  text-only use at the cost of a ~1.7 s encoder start on the first picture after idle. Owner's call.
- Not worth it now: the encoder inside the engine process (one CUDA context; a big refactor for ~0.5 GB and a
  pipe).

## 4. Tasks (Sonnet writes; env switch default-off until validated; acceptance in brackets)

| # | task | exact? | est. gain | LOC | depends |
|---|---|---|---|---|---|
| T1 | D2(a) shared expert before the ids sync in decode | fp-order | −0.5…−1 ms/win | 10 | — |
| T2 | D3 LFU decay once per pass | yes | −0.1…−0.3 | 30 | — |
| T3 | D5 out-of-place Mamba scan + pointer swap | yes | −0.2…−0.4 | 120 | — |
| T4 | D4 top-k on the device for sampling (+ first token) | ties only | −1…−4 when sampling | 150 | — |
| T5 | D6 event instead of the drain at `mtp_rows` start | yes | −0.1…−0.2 | 10 | — |
| T6 | D1 per-expert multi-row GEMV for the verify window | yes | −1…−2 | 250 | round-2 D |
| T7 | D2(b) hits resolved on the device, misses booked behind them | yes | −1…−1.5 | 250 | T1 |
| T8 | E1a Mamba checkpoints in a conversation (N snapshots, prefix pick) | yes | side/branch: 7.7 s → tail only | 200 | — |
| T9 | E1b parking whole conversations (K/V to host RAM) | yes | side request: 7.7 s → ~0.1 s | 350 | T8 |
| T10 | V1 multithreaded resize (+ V3 bf16 .sve, /dev/shm, pin once) | yes | −60…−170 ms/picture | 80 | round-2 C |
| T11 | V4 Q8_0 mmproj A/B (vision_check + timing) | lossy | −20…−30 % GPU encode, −0.85 GB | 0 (tooling) | round-2 C |
| T12 | D6b device-driven draft chain | yes | −0.3 | 300 | T5 |
| T13 | D6c NVFP4 / truncated draft head | acceptance only | −0.7…−1.1 | 150 | owner's call |

Acceptance for the exact decode tasks: `--bench <ids> --ctx 32768 --gen 256 --spec 5` ids unchanged against
the production binary (greedy), `mamba_parity`/`moe_parity` where a kernel changes, then tok/s A/B (3 runs,
warm, 32K) with llama-swap stopped. For T4: the draw sequence with a fixed seed equals production's on a
corpus except at top-k ties. For T8/T9: logits of a resumed conversation bit-equal to a cold read (dump
`head_out`). For T10: `vision_check.py` cos unchanged (multithreading in double keeps bytes identical).

## 5. Cross-check with the Opus pass (`REVIEW_2026-10-22.md`, same day)

Agreed independently: D1 (entries vs distinct experts, its top item), D2 (the 40 drains; it proposes queuing fc1 /
sh_up / the next router behind an event on the ids copy — the 40-line version of D2(a)), D4 (CPU sampling at
temperature 1.0), D5 (Mamba double buffer), D6 (lm_head per draft), "a CUDA graph buys ~nothing", E1 (checkpoints
need only Mamba state + MTP row + pos; parking copies K/V out), V1 (resize), V6 (elastic encoder: the engine's
`VRAM` command is the missing piece).

Taken from it, not in my list above:
- **E2. The previous answer is re-read every turn**: the snapshot sits at the boundary *before* generation, so the
  next turn re-prefills the answer it just generated (0.1–1 s a turn). A checkpoint after generation fixes it when
  the next prompt re-renders the answer identically (reasoning stripped by the client breaks it) — measure how often
  the next turn's ids extend the previous ones before building it. Folds into T8.
- **E3. The generation-prompt tail is a second full pass**: `feed(boundary)` then `feed(prompt.size())`
  (`nemo_main.cpp:383-393`) runs the ~5 header tokens as their own chunk: 40 more drains and plans, 10–30 ms a
  turn. Folds into the short-increment work after round 2 B.
- **E4. Dense weights re-converted every prompt chunk** (`gemm`, `fresh`/`n4_of`, `:794-804`, `:816-819`): for a
  30–400-row chunk the N4/Q8N conversion of every dense matrix is ~5–7 ms of pure traffic. Levers: a resident
  converted copy for the small matrices, or the 8-row tiled GEMV below some T. Measure in round 2 B first.
- **E5. Prompt attention has no split-K for short chunks**: ceil(T/8)×2 blocks over 64K–256K keys → est. 3–8 ms at
  64K, 15–30 ms at 256K per increment. `nemo_attn_bench 65536 64` confirms; the decode split-K kernel is the model.
- **V7. One combined embeddings file per request**: every turn re-writes, re-reads and re-pins the rows of every
  picture of the conversation (40–80 ms a resident picture a turn). Per-picture files + load only the unreused ones.
- **V8. WebP → BMP instead of PNG** in `Vision.normalize` (no PNG encode + decode): −50…−150 ms for WebP clients.
- Resize: skip an axis whose length does not change (exact identity today, still computed).
- Docs to fix: OPTIMIZATION_NOTES §5 says the slot runs `--prefetch 1` (it runs 0, the ×1.091 row); ENGINE_V3 says
  the shared expert runs before the router syncs (prompt path only); "everything per-step is pinned" (routing
  read-backs on the prompt path, both logits copies and the sampled path are pageable); the vision "36–463 ms" is
  the whole ENC, not GPU time; strata_vision.cpp comment says head size 72 (RADIO-H: 80).
Its unknown worth measuring: with `--prefetch 0` the non-skipped misses are copied only after each layer's sync
(0–4 ms a window); a prefetch filtered by predicted weight (the precise preset's prefetch, gated by `skip_w`) could
hide them and lower PPL at once — an A/B, not a code change.
