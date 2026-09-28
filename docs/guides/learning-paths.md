# Learning Paths — How to Read the Gemma-4-E2B-Lite Docs

Four tracks: `concepts/` (theory), `references/` (code-keyed
walkthroughs), `guides/` (this track), `training.md` (applied pipeline).
Start at [docs/README.md](../README.md) for the corpus map and the
file→doc table.

## Beginner — "what is this model?"

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [README](../../README.md) | what Gemma-4-E2B-Lite is, its status discipline (CPU-correct, A100 pending, not trained) |
| 2 | [guides/quickstart.md](quickstart.md) | run the test suite, build the model, reproduce the exact parameter count on your CPU |
| 3 | [concepts/per-layer-embeddings.md](../concepts/per-layer-embeddings.md) | how token identity reaches every layer, and what the norms pin |
| 4 | [concepts/local-global-attention.md](../concepts/local-global-attention.md) | why 16 layers look at 512 tokens and 4 look at everything |
| 5 | [guides/glossary.md](glossary.md) | the vocabulary used by every other doc |

## Intermediate — "how do I train it?"

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [references/data.md](../references/data.md) | how shards become windows, and every validation that protects the run |
| 2 | [training.md](../training.md) | the full applied loop: schedule, accumulation, checkpointing, resume |
| 3 | [references/training.md](../references/training.md) | the loop's public surface: `chunked_causal_ce`, `train`, `CheckpointManager` |
| 4 | [concepts/cross-layer-kv-sharing.md](../concepts/cross-layer-kv-sharing.md) | why 10 layers compute K/V for 20, and what the cache guarantees |
| 5 | [guides/troubleshooting.md](troubleshooting.md) | the failure modes the validations exist to catch |

## Expert — "how is it built and verified?"

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [concepts/proportional-partial-rope.md](../concepts/proportional-partial-rope.md) | the exact frequency construction and why the denominator is the full head dim |
| 2 | [references/models.md](../references/models.md) | every config key, every validation, the analytic ledger, block-by-block shape trace |
| 3 | [references/cache.md](../references/cache.md) | producer/consumer cache invariants, masking over per-row positions, generation |
| 4 | [AUDIT](../AUDIT.md) | the verification record: gates, parity, the 348,965,184 reconciliation |
| 5 | [sources](../sources.md) | the pinned upstream revisions this repo proves parity against |
