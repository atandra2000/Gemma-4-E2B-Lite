# References: the training stack — loss, loop, persistence

> Audience: intermediate→expert. Covers every public symbol of
> `training/losses.py`, `training/pretrain.py`, `utils/checkpoint.py`.
> The applied narrative lives in [training.md](../training.md).

## `training/losses.py` — chunked causal cross-entropy

`training/losses.py:chunked_causal_ce(hidden, tied_weight, labels, logit_softcap=30.0, chunk_size=1024)`

Contract: identical math to the oracle path —
`softcap(linear(hidden))[:, :-1]` vs `labels[:, 1:]`, mean over valid
tokens — without materializing `[B, T, V]`.

- **Shift once, on compact tensors**: `hidden[:, :-1]` vs
  `labels[:, 1:]`, flattened before chunking — windows downstream are
  flat slices, and the dataset therefore ships flat windows (see
  [data reference](data.md)).
- **FP32 reductions throughout**: linear, softcap
  (`training/losses.py:_softcap`), CE.
- **Chunking**: `chunk_size` token rows per chunk; each chunk computed
  under non-reentrant `checkpoint(...)` — backward recomputes chunk
  logits instead of retaining any chunk graph; `tied_weight` is passed
  explicitly so its gradient flows through recomputation. Only the
  single-chunk case (≤ `chunk_size` rows) skips the checkpoint wrapper —
  same math, cheaper path.
- **Ignore semantics**: CE default −100 excluded from sum *and* count;
  `training/losses.py:_valid_count` raises `ValueError` on an
  all-ignored batch — the caller decides between failing and skipping
  the optimizer step, never a silent NaN.
- The sum-then-divide shape is what the loop exploits:
  `train` backward-passes `loss × valid` and divides the accumulated
  gradient by the window's actual valid tokens (next section).

## `training/pretrain.py` — the loop

### Schedule and optimizer

- `training/pretrain.py:lr_at` — linear warmup (first step trains:
  `(step+1)/warmup`) then cosine to `lr_final_fraction · lr`. Token
  semantics: a "step" is one optimizer update covering
  `accumulation_tokens` valid tokens. Production: 8B / 65,536 ≈
  122,071 updates, warmup 1%, 3e-4 → 3e-5 (from `configs/pretrain_a100.yaml`).
- `training/pretrain.py:build_optimizer` — AdamW, weight decay on
  matrices only: `p.ndim < 2`, embeddings and the PLE table are
  no-decay (house convention, encoded in the name checks).
- `training/pretrain.py:IGNORE_ID` — the loop-side −100 alias.

### Determinism plumbing

- `training/pretrain.py:capture_rng` / `training/pretrain.py:restore_rng`
  — Python / NumPy / torch RNG states, saved with every checkpoint so a
  resumed run continues bit-identically (`tests/test_resume.py`).
- Model init seeds `torch.manual_seed(seed)` before construction —
  identical configs build identical models; otherwise a resumed run
  diverges from step 1 (`training/pretrain.py:train`).

### `training/pretrain.py:train(cfg, batches, ...)`

Takes any iterable of `(input_ids, labels)` — tests feed fixtures,
production feeds `data/dataset.py:build_dataloader`. Per micro-batch:

1. `loss = chunked_causal_ce(model.forward_hidden(ids), weight, labels, ...)` —
   note `forward_hidden`, not `forward`: the tied weight is applied
   inside the loss, so full logits never exist.
2. `(loss × valid).backward()` — sum-scaled; normalization deferred.
3. Accumulate `window_valid`, `window_loss_sum`, `tokens_seen`.

At each accumulation-window boundary (`window_valid ≥
accumulation_tokens`) → `finalize_window`: zero-valid windows are
skipped silently; **nonfinite loss or any nonfinite grad → update
skipped, gradients zeroed** (message printed, run continues); otherwise
gradients ÷ actual valid tokens, `clip_grad_norm_(grad_clip)`, LR from
`training/pretrain.py:lr_at`, `optimizer.step()`. A checkpoint
(`utils/checkpoint.py:CheckpointManager.save`) lands after every
optimizer step, carrying `step`, `tokens_seen`, RNG states,
`config_hash`, `batches_consumed`. Resume replays-skip consumed batches
(`batches_consumed` in metadata) before the first new batch.

### Entry points

- `training/pretrain.py:main` — CLI (`--config`, `--checkpoint-dir`,
  `--resume`); `data.source: synthetic` builds
  `training/pretrain.py:smoke_batches` (the ONLY synthetic-data
  configuration — `configs/smoke.yaml`), anything else wires
  `data/dataset.py:build_dataloader` with labels = inputs (LM objective).
- Guardrails live in config validation (`models/config.py:ModelConfig.from_yaml`)
  and the dataset adapter — production must fail on absent or
  mismatched data, not fall back (see
  [troubleshooting](../guides/troubleshooting.md)).

## `utils/checkpoint.py` — atomic generations

`utils/checkpoint.py:CheckpointManager` — one generation per save under
`gen_<step>/`:

- `utils/checkpoint.py:CheckpointManager.FILES` — the complete set
  (`model.safetensors`, `optim.pt`, `rng.pt`, `meta.json`); a generation
  counts as complete only when all four exist.
- `utils/checkpoint.py:CheckpointManager.save` — writes a new
  generation, then commits by atomically renaming `pointer.json` to
  point at it. Tied embeddings alias one storage, which safetensors
  rejects — duplicated storages are cloned once at save time. Same-step
  re-saves (the final save after the last window's save) replace the
  generation in place, with the pointer window kept short.
- `utils/checkpoint.py:CheckpointManager.load` — restores the latest
  complete generation (`strict=True` weights, optimizer, RNG); rejects
  a `config_hash` mismatch without touching the checkpoint.
- `utils/checkpoint.py:CheckpointManager.latest_generation` — the
  pointer's target only if every file is present; a bare `gen_*` dir
  without a pointer is an incomplete write and is ignored.
- `utils/checkpoint.py:CheckpointManager._atomic_json` — temp file +
  `os.replace` for both JSONs (the mechanism behind "whole generation or
  previous one").

Crash semantics: a crash before the rename leaves the previous
generation loadable; partial writes are never loadable and never
overwrite the last complete one.
