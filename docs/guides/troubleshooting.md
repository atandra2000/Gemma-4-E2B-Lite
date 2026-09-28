# Troubleshooting

Failure modes this codebase is *designed* to raise, what each means, and
the fix. Every row is enforced by a check — these errors are the happy
path of the validation discipline.

## Data

| Symptom | Raised by | Meaning → fix |
|---|---|---|
| `tokenizer mismatch: manifest says 'llama2', this project requires 'gpt2'` | `data/dataset.py:ShardWindows.__init__` | the shards were packed by another project's tokenizer contract → re-run `python3 data/prepare_data.py` (its shim pins the GPT-2 tokenizer) |
| `manifest lists missing shard files: [...]` | `data/dataset.py:ShardWindows.__init__` | manifest and directory disagree → re-pack or restore the shard dir; never hand-edit the manifest |
| `shard shard_003.bin: N tokens on disk, manifest records M` | `data/dataset.py:ShardWindows.__init__` | corrupt or truncated shard → re-pack; the manifest is the source of truth |
| `corpus too small: N tokens < seq_len S` | `data/dataset.py:ShardWindows.__init__` | fewer than one full window → pack more data or lower `seq_len` |
| `window W: token id T >= vocab_size 50257 — wrong tokenizer packed this corpus` | `data/dataset.py:ShardWindows.validate` | same-shape vocab, different tokenizer → the sampled boundary check caught it; re-pack with this repo's shim |
| `FileNotFoundError: ... requires the workspace shared_data package` | `data/prepare_data.py:_require_shared_data` | data prep run outside the CoreProjects workspace → run from a checkout where `../shared_data` exists |

## Loss / training

| Symptom | Raised by | Meaning → fix |
|---|---|---|
| `ValueError: all labels are ignored (-100); refusing to produce a silent NaN` | `training/losses.py:_valid_count` | every target in the batch is padding → fix the batch composition, or make the caller skip the optimizer update (that choice is deliberate and belongs to the caller) |
| `[train] step S: nonfinite loss/grad — update skipped, gradients dropped` | `training/pretrain.py:train` | a bad micro-batch poisoned the window; the update is skipped, the run continues → if this recurs, look at data first (see the rows above), then LR |
| `[train] step S: nonfinite grad norm — update skipped` | `training/pretrain.py:train` | gradients blew up after accumulation → check LR vs `warmup_fraction` and `grad_clip` in the config |
| `ValueError: chunk_size must be >= 1, got 0` | `training/losses.py:chunked_causal_ce` | config `loss_chunk_tokens` must be ≥ 1 |

## Checkpoints / resume

| Symptom | Raised by | Meaning → fix |
|---|---|---|
| `FileNotFoundError: no complete checkpoint under ...` | `utils/checkpoint.py:CheckpointManager.load` | no `pointer.json` (never saved) or only a bare `gen_*` dir (crash mid-save — correctly rejected) → start fresh or resume from an older healthy dir |
| `checkpoint config_hash '...' != run's '...' — refusing to resume an incompatible run` | `utils/checkpoint.py:CheckpointManager.load` | the config changed since the checkpoint → resume only with the original config, or start a new run |

## NaN discipline (attention)

- An all-masked query row (left-padded row, no valid keys) must not NaN.
  The cache path fills skipped slots with `models/cache.py:MASK_FILLER`
  (`finfo.min`) instead of −inf exactly for this — if you bypass the
  cache's `attention_mask` with a hand-rolled −inf mask, you will
  reinvent the bug. See
  [cross-layer KV sharing](../concepts/cross-layer-kv-sharing.md#masks-over-per-row-positions).
- If quality "degrades after N decode steps": check position derivation
  first. Positions must come from validity counts, once per step, before
  any producer appends (`models/transformer.py:Gemma4LiteModel.forward_hidden`)
  — per-block derivation ropes later blocks one step ahead. The cache's
  `models/cache.py:ProducerKVCache.valid_counts` is the clock; slot
  indices after local trimming are *not* positions.

## Environment

| Symptom | Meaning → fix |
|---|---|
| oracle tests skip or fail on `transformers` | the parity oracle needs the pinned `transformers==5.15.1` (+ `safetensors`); an older system copy (e.g. 5.10.2) may shadow it — see the environment record in [sources](../sources.md) |
| GPU-gated tests silently skip | by design: `gpu` / `slow` / `heavy` markers auto-skip on CPU hosts (`pytest.ini`); never delete them because they skip (AGENTS.md rule) |
| doc gate fails: `unresolved citation` | a doc cites a symbol that no longer exists → fix the doc or the rename; the gate exists so this is a build failure, not doc rot |

## What is deliberately NOT a failure mode

- Same-step re-saves of a checkpoint generation: handled in place by
  `utils/checkpoint.py:CheckpointManager.save`.
- A window crossing a shard boundary: served by two mmap reads
  (`data/dataset.py:ShardWindows.__getitem__`).
- A micro-batch with zero valid tokens at a window boundary: skipped
  silently by `finalize_window` (a fully-ignored *batch* inside the loss
  still raises — see the loss row above).
