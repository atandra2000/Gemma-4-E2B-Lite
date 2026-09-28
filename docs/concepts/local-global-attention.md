# Local/global attention — two clocks for two context scales

> Audience: intermediate→expert. Builds its own prerequisites (multi-query
> attention, additive masks). Pairs with
> [proportional partial RoPE](proportional-partial-rope.md) and
> [cross-layer KV sharing](cross-layer-kv-sharing.md).

Recurrent-adjacent mechanisms (sliding windows) and long-range mechanisms
(full attention) have opposite cost/quality curves. Gemma E2B's answer,
preserved here, is to split the 20 decoder layers into **two attention
types on a fixed period** and let each specialize:

| Property | Local layers (16) | Global layers (4) |
|---|---|---|
| Indices (0-indexed) | 0–19 minus {4, 9, 14, 19} | **4, 9, 14, 19** |
| Reach | window 512 (key ≤ query, query − key < window) | all preceding keys |
| Head dim | 128 | 256 (pinned 1:2, `models/config.py:ModelConfig.global_head_dim`) |
| RoPE | full-dim, θ=10⁴ | partial (factor 0.25), θ=10⁶ |
| KV role | producers 0–9 write; 10–19 consume | same split |

The layer map is data, not convention: `models/config.py:ModelConfig.global_layers`
holds the global indices, `models/config.py:ModelConfig.layer_type`
answers "local or global?", and `models/config.py:ModelConfig.__post_init__`
enforces the invariants (sorted/unique globals, **final layer must be
global**, valid head geometry). The smoke config
(`models/config.py:tiny_config`) reuses the same machinery with globals
{2, 5, 8, 11} — proof the pattern, not the numbers, is what's validated.

## Why the head dims differ (the pinned 1:2 relation)

Local heads work on a 512-token neighborhood; global heads must encode
relative position over thousands of tokens, which [partial RoPE with a
large θ](proportional-partial-rope.md) achieves by widening the head
(128 → 256) while rotating only a quarter of it
(`models/config.py:ModelConfig.global_rope_partial_factor`). The 1:2
relation is validated, not assumed
(`models/config.py:ModelConfig.__post_init__` raises unless
`global_head_dim == 2 × local_head_dim`).

## Multi-query attention, one KV head

Every layer has **six query heads and one KV head**
(`models/config.py:ModelConfig.n_heads`, `models/config.py:ModelConfig.head_dim`):
K/V are projected to a single head dim and broadcast over the query heads
via `k.expand(...)` in `models/attention.py:Attention.forward`. With
[cross-layer KV sharing](cross-layer-kv-sharing.md) on top, only the 10
prefix layers own K/V projections at all — the KV parameter line in the
ledger is `kv_projections_prefix_only`
(`models/config.py:count_large_matrices`).

## Scales, norms, and the non-negotiable 1.0

The attention scale is **pinned to 1.0** — never SDPA's default `1/√d`
(`models/attention.py:Attention.scaling`;
`models/config.py:ModelConfig.__post_init__` rejects any other value).
Q and K are RMSNorm'd per head *before* the dot product
(`models/attention.py:Attention.q_norm`,
`models/attention.py:Attention.k_norm`; the V norm has no weight — see
[PLE/norms](per-layer-embeddings.md#the-norm-underneath-and-two-source-conventions-worth-knowing)).
QK-norm plus scale 1.0 is the source recipe; changing the scale changes
the model, which is why it is config-validated.

## Two causality implementations, one contract

**Training** (no cache): `models/attention.py:causal_mask` builds an
additive `[T, T]` mask over **absolute positions** — 0 to keep, −inf to
skip; with a `window` the keep-set becomes `key ≤ query ∧ query − key <
window`. Per layer the mask comes from
`models/transformer.py:Block.forward`, which passes the layer's own
window only when `models/attention.py:Attention.layer_type` is local.

**Inference** (`models/cache.py:ProducerKVCache`): masks are per-row and
per-layer-type, built from tracked key validity and the trim offset —
see [KV sharing](cross-layer-kv-sharing.md#masks-over-per-row-positions).

The eager reference path (`models/attention.py:Attention.forward`,
`attn_backend="eager"`) is the Task-4 oracle shape: explicit
`matmul → +mask → softmax(FP32) → matmul`. The SDPA path calls
`F.scaled_dot_product_attention` with `scale=1.0` and the additive mask
expanded to `[B, 1, T, kv]`. Backend parity is gated by
`tests/test_backends.py` — both paths agree to FP32 tolerance, so "eager"
and "fast" are the same function by test, not by faith.

## All-masked rows must not NaN

The cache path fills skipped slots with `finfo.min`, **not** −inf
(`models/cache.py:MASK_FILLER`): a left-padded query row whose keys are
all masked would otherwise softmax to NaN and poison valid positions in
later layers through `0 · NaN`. This one constant is the difference
between "batch of variable-length requests works" and "mysterious NaN at
step k" — see [troubleshooting](../guides/troubleshooting.md#nan-discipline-attention).
