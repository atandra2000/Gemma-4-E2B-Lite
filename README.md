# Gemma-4-E2B-Lite

From-scratch, small autoregressive language model in raw PyTorch preserving the
Gemma E2B text mechanisms: per-layer embeddings (PLE), local/global attention,
proportional RoPE and cross-layer KV reuse. Trained from scratch on the
workspace corpus; one A100 80GB target. This is an E2B-derived Lite adaptation,
not a checkpoint-compatible reproduction (~349M parameters from the design's
analytic ledger, exact count established by the instantiated-model gate).

**Status: CPU-correct through Phase 3 (Tasks 1–10 of the execution plan);
Phase 4's A100 hardware boundary (Task 11) is next.**
`models/config.py` holds the validated architecture contract, producer map and
analytic ledger (large-matrix subtotal 348,882,944); `models/ple.py`,
`models/attention.py` + `models/transformer.py` implement PLE, decoder-block
arithmetic, eager/SDPA local/global attention with cross-layer KV sharing, the
full forward (final norm, tied head, softcap), the producer-owned KV cache
with greedy generation, and non-reentrant activation checkpointing with
verified gradient parity (`tests/test_backends.py`). `training/losses.py`
provides the softcapped chunked causal CE with its own +1 shift;
`data/dataset.py:ShardWindows` and `build_dataloader` window the shared
pipeline's manifest-validated uint32 shards (tokenizer-name, length and
sampled vocab-boundary checks); `training/pretrain.py` runs AdamW on a
warmup+cosine token schedule with gradient accumulation and atomic resume
via `utils/checkpoint.py:CheckpointManager`. Tiny-weight parity against the
pinned v5.15.1 upstream oracle is verified in `tests/test_reference.py`, and
the exact instantiated total is **348,965,184** parameters (ledger
reconciled, `scripts/parameter_budget.py`). No training run or hardware
measurement yet; A100 evidence (Task 11) is next.

## Documents

- [Design specification](../../llm-research/DESIGN-gemma-4-e2b-lite.md) — the model contract.
- [Execution plan](../../llm-research/EXECUTION-PLAN-gemma-4-e2b-lite.md) — task order and acceptance gates.
- [Source manifest](docs/sources.md) — pinned upstream revisions and verification status.
- [Diagrams](docs/diagrams/) — system, decoder block, training loop and
  dataflow (standalone HTML, dark/light themes; PNG renders alongside).

## Completion states

Three separate states; missing hardware or checkpoints leaves later states
pending, never "passed":

1. **CPU-correct** — implementation, tests and deterministic recovery pass locally.
2. **A100-measured** — fit, throughput and memory measured on the target hardware.
3. **Trained** — checkpoint trained and quality report published.

## Environment

macOS dev host: CPU/MPS only, raw PyTorch. GPU validation happens on rented
A100 pods via later-phase `scripts/*a100*.py`. See `AGENTS.md` for project rules.
