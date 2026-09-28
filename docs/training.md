# Training — the applied pipeline

> The narrative version of [references/training.md](references/training.md).
> Status discipline: everything here is CPU-measured or analytic; the
> A100 row of the README status table is still pending (no hardware
> session yet), and nothing below predicts hardware results.

## Pipeline at a glance

```
shared_data pipeline → shards + manifest → ShardWindows (validated, mmap)
    → build_dataloader (shuffled, resumable) → train() → chunked_causal_ce
    → accumulation window → finalize (skip-or-step) → atomic checkpoint
```

## 1. Data

`python3 data/prepare_data.py` packs the universal corpus with this
repo's tokenizer contract (GPT-2 BPE, 50,257 — set by
`data/prepare_data.py:GEMMA4_TOKENIZER_NAME` and friends). The adapter
(`data/dataset.py:ShardWindows`) then refuses to train on anything that
doesn't match the manifest and the vocab contract — the five data checks
are listed in [references/data.md](references/data.md). Windows are flat
`seq_len` slices; the loss owns the next-token shift, so there is no
shift in the data path (one shift, one owner — the classic double-shift
bug is designed out).

## 2. Config

`configs/pretrain_a100.yaml` (production) and `configs/smoke.yaml` (the
only synthetic-data configuration) share a schema; `models/config.py:ModelConfig.from_yaml`
validates the architecture section at load. Training-relevant production
values: `seq_len` 4096, `total_tokens` 8e9 (**budget target, not a proven
optimum**), `accumulation_tokens` 65,536, AdamW lr 3e-4 → 3e-5 (cosine,
1% warmup), `grad_clip` 1.0, BF16 autocast plan, activation
checkpointing on, `torch_compile` off until eager parity on hardware.

Derived (analytic): `ceil(8e9 / 65,536)` ≈ **122,071 optimizer updates**;
≈ 1,953,125 windows of 4,096.

## 3. The loop (`training/pretrain.py:train`)

Per micro-batch:

1. `model.forward_hidden(ids)` → final hidden states `[B, T, D]` —
   **not** full logits.
2. `training/losses.py:chunked_causal_ce` applies the tied head, the
   +1 shift, the softcap and CE in `loss_chunk_tokens`-sized chunks
   under non-reentrant checkpointing — only one chunk's logits ever
   live; backward recomputes the rest. FP32 reductions throughout.
3. `(loss × valid).backward()` — sum-scaled; each micro-batch
   contributes proportionally to its actual valid tokens.

At each 65,536-valid-token window, `finalize_window`:

- zero valid tokens → skip silently;
- nonfinite loss **or any nonfinite gradient** → skip the update, zero
  grads, continue (a printed notice, not a crash — a single bad
  micro-batch shouldn't kill an 8B-token run);
- otherwise: grads ÷ actual window valid tokens,
  `clip_grad_norm_`, `training/pretrain.py:lr_at`, `optimizer.step()`.

The LR schedule is token-based (an "step" = one optimizer update):
linear warmup on the first 1% of updates, cosine to 10% of peak.

## 4. The memory stack (derived, not measured)

The conservative design-§3 accounting, produced by
`models/config.py:memory_assumptions` and printed by
`scripts/parameter_budget.py:main`:

| Item | Value | Basis |
|---|---|---|
| trainable state (params + grads + AdamW moments + master, 16 B/param) | ≈ 5.20 GiB | ledger subtotal 348,882,944 × 16 B — analytic |
| full-logit row, B=1, T=4096 | 0.38 GiB BF16 / 0.77 GiB FP32 | `V × T × bytes` — analytic |
| logits actually materialized | one `loss_chunk_tokens` chunk at a time | `training/losses.py:chunked_causal_ce` design |
| activations | recomputed per attention block when `activation_checkpointing` | `models/transformer.py:Gemma4LiteModel.checkpoint_blocks` |

What is **not** in this table: peak VRAM, throughput, and the BF16-vs-
FP32 parity factor on A100 — those are Task-11 measurements and do not
exist yet. The chunked loss plus tied head exist precisely so the
full `[B, T, V]` tensor (0.77 GiB FP32 at B=1 alone, multiplied by batch
in practice) never has to.

## 5. Persistence and resume

Every optimizer step saves an atomic generation
(`utils/checkpoint.py:CheckpointManager.save`): weights (safetensors,
tied storage cloned once), optimizer state, **RNG states**
(`training/pretrain.py:capture_rng`), and metadata including
`config_hash` and `batches_consumed`. `--resume` restores all four and
replay-skips consumed batches; `tests/test_resume.py` proves the
continuation is bit-identical. Crash mid-save cannot corrupt the previous
generation (pointer-rename commit — see `utils/checkpoint.py:CheckpointManager.save`
in [references/training.md](references/training.md)).

## 6. Verification hooks a train run should use

- `python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml`
  before the run — the ledger must reconcile (348,965,184 exact).
- `data/dataset.py:ShardWindows.validate` as a preflight — sampled
  vocab-boundary scan.
- Watch for the two skip notices in the loop log (nonfinite loss/grad) —
  recurring skips are a data or LR problem, not noise.

## 7. What changes on A100 (planned, unmeasured)

The plan (design §5) — not yet exercised on hardware: BF16 autocast
(`precision.autocast`), `microbatch_sizes` benchmarked at 1 then 2,
activation checkpointing on, `torch_compile` only after eager parity
passes on-device. Every number this section will eventually carry is
pending the Task-11 session; until then, the only hardware claim in this
repo is the *absence* of hardware claims.
