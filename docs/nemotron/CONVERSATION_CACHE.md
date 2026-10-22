# strata-nemotron: checkpoints inside a conversation and parked conversations (2026-10-22)

Implementer-facing. Off by default: without the flags below the engine behaves as before (one VRAM snapshot, taken at
the last message boundary). Source: `src/core/nemo_conv_cache.cpp`, `include/strata/core/nemo_conv_cache.hpp`, the
`run_serve` part of `src/program/nemo_main.cpp`, and `checkpoint_*` / `kv_*` in `NemoModel`.

## What it is for

A Mamba state cannot be rolled back, so a request can resume only at a position whose state was set aside earlier.
Before this change the engine kept one such snapshot, at the last message boundary of the last request. Two things
followed in an agent session at 60K+ tokens:

- every turn re-read the previous assistant answer and the template tokens after the snapshot (about 45 tokens in the
  tests below), at short-prompt speed;
- any other request (a title, a summary, a sub-agent) or an edited earlier message threw the snapshot away, and the
  whole context was read again: 7.8 s at 81K tokens (measured, see `REVIEW_2026-10-22.md`).

## Flags

Pass them through `args` of the server config, or on the engine's command line.

| flag | default | meaning |
|---|---|---|
| `--prompt-cache N` | 0 (off) | up to N live checkpoints of the current conversation. A pool of N pinned buffers of ~165 MB each is allocated once at load |
| `--prompt-cache-every M` | 16384 | a checkpoint every M tokens while a long prompt is read (0 = none). Use a multiple of `--chunk` |
| `--conversation-cache-mib X` | 0 (off) | pinned host memory for parked conversations. If X > 0 and N < 2, N is raised to 2 |

The example config uses `--prompt-cache 8 --conversation-cache-mib 3000`: 8 × ~165 MB + 3000 MiB of pinned RAM on top
of the 59 GiB expert arena.

Debugging only: the environment variable `STRATA_NEMO_DUMP_LOGITS=file` appends the logits after each request's
prompt to a file; the request parameter `nocache=1` reads the prompt cold, `nocache=2` uses the old single snapshot.

## What a checkpoint holds

- the Mamba state and conv state of the 40 Mamba blocks (the bulk of the ~165 MB);
- the MTP row `mtp.h`, the pending-token state (`pending`, `pending_pos`, `consumed_tok`) and `pos`;
- the head's logits row, so the first token can be sampled without a forward pass.

K/V cells are positional: a cell is written at its position, so resuming at position p inside the same conversation
needs no K/V copy. The cells past p are simply overwritten. `kv_gen()` counts `reset()` calls; a live checkpoint is valid
only while no `reset()` has happened since it was taken.

## When checkpoints are taken

- at a message boundary (as before);
- at chunk boundaries of a long prompt, on an absolute grid that is a multiple of `--prompt-cache-every`;
- after generation (the "answer" checkpoint). Its ids are the prompt plus the tokens actually fed to the model, so the
  next turn, whose prompt repeats the assistant answer, matches it exactly. With the cache on, the last speculation
  window feeds no tokens beyond `max_new`; otherwise the answer checkpoint would not match the next prompt.

## How a request picks a checkpoint

The engine takes the checkpoint, live or parked, whose ids are the longest prefix of the new prompt. Pictures must
match too: the keys of the pictures the checkpoint covers must equal the request's keys (a picture with an unknown key,
`?`, never matches).

- **Live checkpoint**: its state is copied host to device. Checkpoints deeper than it are dropped, because their K/V
  cells are about to be overwritten. If parking is on and a deeper checkpoint is dropped, it is parked first, so a
  branch from an earlier turn does not lose the other branch.
- **Parked conversation**: the current conversation is parked, then `reset()` runs, then the parked K/V cells and the
  checkpoint are copied back and the parked entry is removed.
- **No match**: the current conversation is parked and `reset()` runs; the prompt is read from the start.

## Parking

Parking copies the conversation's K/V cells `[0, pos)` of every K/V owner (the 8 attention layers and the MTP block)
and up to three checkpoints (the deepest one and the two deepest message boundaries) to pinned host memory. The
checkpoints share the K/V cells up to the deepest position. Space comes from one arena (first-fit, with merging of free
blocks) of `--conversation-cache-mib`.

Eviction:
- live checkpoints: least recently used, except the deepest;
- parked conversations: the oldest first; an entry whose ids are a prefix of a new entry's ids is replaced by it.

## Sizes and times

Measured in one engine process: `--skip-w 0`, `--spec 5`, chunk 16384, `--prompt-cache 8`, 3000 MiB budget (RTX 5090).

| | measured |
|---|---|
| parking a 60K conversation | 21–43 ms, 729–907 MiB |
| unparking 60K | 17–18 ms |
| unparking 22K | ~10 ms |
| restoring a live checkpoint | ~3.8 ms |

## How it was checked

- Resume is bit-identical to the old snapshot (logit difference 0.0, same tokens), for a live checkpoint and for a
  parked one. Against a cold read the logits differ by 2–4: that is the chunk split, not the cache, and the old
  snapshot already had the same difference.
- Tokens taken from a checkpoint: P (60K) then P + 200 tokens: 60,002; the answer checkpoint: 20,267 = prompt + 63 fed
  tokens; a branch P+A then P+B: 60,002, with the prompt time going from 7.4 s to about 1.2 s; P (60K), then an
  unrelated Q, then P + 100: 60,002 after unparking.
- Eviction: 8 conversations of 22K tokens with a 3000 MiB budget. No failures. The two oldest (C1, C3) were evicted
  (0 tokens reused); the three that were asked again afterwards (C5, C7, C8) reused 22,002 tokens each.
- Through the server, old build against this one, same machine: the turn-by-turn tables are in
  `REVIEW_2026-10-22.md`. In short, after a side request in the middle of an 81K session the next turn took 7,803 ms
  without the cache and 204 ms with it.

## Limits

- Parking happens on every branch. In a turn where the answer is re-rendered differently from what was generated, the
  answer checkpoint does not match, so each turn can park a copy of the conversation; entries with non-matching
  prefixes pile up until the budget evicts them.
- Unparking restores only the checkpoints stored with the parked conversation (at most three), not the ones in
  between.
- A resume equals a cold read bit for bit only when the chunk split is the same.
- A warm request does a little more work: in the A/B its prompt step went from 31–35 ms to 41–49 ms (the checkpoint
  restore, plus parking a copy of the conversation, ~23 ms per park).
- The single-snapshot path is still there (`nocache=2`), and without the flags none of the pools are allocated.
