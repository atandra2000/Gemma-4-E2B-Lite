# Cross-layer KV sharing — ten layers compute K/V, twenty use them

> Audience: expert. Builds on [local/global attention](local-global-attention.md).
> The cache mechanics here are the engine behind
> [generation](../references/cache.md) and
> [training memory](../training.md#4-the-memory-stack-derived-not-measured).

Every layer still needs *correct* K/V — the sharing claim is about who
**computes and stores** them. Layers 0–9 (the producer prefix) own K/V
projections and norms; layers 10–19 (the shared suffix) have **no K/V
weights at all** and reuse a same-type producer's tensors. With a KV head
per layer and multi-query broadcast, this removes 10 layers' worth of
K/V matrices and norms from the ledger
(`models/config.py:count_large_matrices` →
`kv_projections_prefix_only`, 2,359,296 params in the production config —
versus double that if every layer produced).

## The producer map

`models/config.py:ModelConfig.kv_producer` is the total map: prefix
layers produce for themselves; a consumer inherits the **last same-type
prefix layer** — shared locals (10–19) ← layer 8, shared globals
(10–19) ← layer 9, in the production config
(`models/config.py:ModelConfig.share_boundary` = 10). Validation in
`models/config.py:ModelConfig.__post_init__` refuses configs where a
consumer has no same-type producer below the boundary — this is why the
smoke config (`models/config.py:tiny_config`) interleaves globals into
its prefix.

## Two storage regimes, one map

**Training** — `models/attention.py:Attention.forward` receives a plain
dict (`shared_kv_states`). The *last producer of each type* stores its
full-length, roped, normalized K/V
(`models/attention.py:Attention.store_full_length_kv` flags exactly that
layer); consumers alias `shared_kv_states[layer_type]`. The tensors stay
**autograd-connected** — consumers backprop through the producer's K/V —
which is a stated design invariant, not an implementation accident.
Nothing is detached, nothing is copied.

**Inference** — `models/cache.py:ProducerKVCache` replaces the dict:
each prefix producer appends its chunk **once per token**
(`models/cache.py:ProducerKVCache.producer_append`), and each consumer
reads its producer's current-call states
(`models/cache.py:ProducerKVCache.consumer_states` via
`models/cache.py:ProducerKVCache.stream_for` →
`models/config.py:ModelConfig.kv_producer`). After the *last consumer of
the call* has run, `models/cache.py:ProducerKVCache.finish_step`
commits the call's states to history and trims each **local** stream to
`window − 1` keys (the most a future local query can still attend);
**global streams are never trimmed**, which is why the untrimmed global
stream's length is the absolute token clock
(`models/cache.py:ProducerKVCache.num_tokens`,
`models/cache.py:ProducerKVCache.valid_counts`).

The cache holds each persistent tensor **once**
(`models/cache.py:ProducerKVCache.unique_storages` counts distinct
`data_ptr`s — the "no consumer copies" accounting that
`tests/test_cache.py` asserts).

## Masks over per-row positions

With left padding and trimming, "slot index" ≠ "token position". The
cache builds masks from positions, not indices:
`models/cache.py:ProducerKVCache.attention_mask` derives each key's
position as `dropped + cumsum(valid) − 1` — padding tokens occupy slots
but contribute zero to the cumsum, so valid tokens in a left-padded row
sit at the positions they would occupy unpadded. The keep-set is
`key_pos ≤ query_pos ∧ (window) ∧ valid`, filled with
`models/cache.py:MASK_FILLER` (`finfo.min`, the all-masked-row NaN guard
explained in [local/global attention](local-global-attention.md)).

Query positions come in from the caller, derived **once per step** in
`models/transformer.py:Gemma4LiteModel.forward_hidden` — before any
producer appends, because `valid_counts()` grows the moment the first
global producer runs and per-block derivation would rope later blocks
one step ahead — see the comment in
`models/transformer.py:Gemma4LiteModel.forward_hidden`; the invariant is
tested in `tests/test_cache.py`.

## Gradient checkpointing across the aliasing

`models/transformer.py:Block.forward` may checkpoint its attention call
(`models/transformer.py:Gemma4LiteModel.checkpoint_blocks`). That is safe
precisely because recomputation re-runs the *same call* with the *same
arguments*: producers recompute fresh K/V, consumers re-alias them, and
no mutable dict is read at backward time
(`models/transformer.py:Block.forward` comment). Non-reentrant
checkpointing preserves this.

## What is verified

- Producer/consumer aliasing with no consumer-owned storages:
  `tests/test_cache.py`.
- Eager↔SDPA and checkpoint grad parity across the sharing:
  `tests/test_backends.py`.
- Bit-identical resume through the whole loop:
  `tests/test_resume.py`.
- Structural absence of consumer K/V weights and of any `lm_head` when
  tied: `scripts/parameter_budget.py:reconcile_with_ledger`.
