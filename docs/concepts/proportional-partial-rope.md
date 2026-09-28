# Proportional partial RoPE — the global layers' position code

> Audience: expert. Requires RoPE fluency; builds its own prerequisites.
> Sibling docs: [local/global split](local-global-attention.md),
> [KV sharing](cross-layer-kv-sharing.md).

RoPE encodes position by rotating each *pair* of head coordinates at a
frequency `f_i = θ^(−2i/d)`: small `i` pairs spin fast (fine local
detail), large `i` pairs spin slow (coarse position). Two knobs — θ and
*d* — therefore decide what "distant" looks like to a head. The two
attention types in this model make opposite choices, and
`models/attention.py:rotary_inv_freq` implements both with one function:

| | Local layers | Global layers |
|---|---|---|
| θ (`models/config.py:ModelConfig.local_rope_theta` / `.global_rope_theta`) | 10⁴ | 10⁶ |
| rotated coordinates | all of head dim | **first 64 of 256** (`factor 0.25`) |
| non-rotated tail | none | 192 coordinates, zero frequency → identity |
| frequency denominator | head_dim | **full head_dim (256), not the rotary width (64)** |

## What "proportional" buys

The naive way to do partial RoPE is to compute frequencies over the
rotary width only (`θ^(−2i/64)` for i < 32 pairs). The *proportional*
variant keeps the exponent denominator at the **full** head dim:
`models/attention.py:rotary_inv_freq` computes
`1 / θ^(2i/head_dim)` for the `partial_rotary_factor · head_dim/2` active
pairs, then appends `head_dim/2 − rope_angles` **zeros** for the idle
tail. Zero frequency ⇒ `cos=1, sin=0` ⇒ the tail coordinates pass
through unchanged — `models/attention.py:apply_rope` applies
`x·cos + rotate_half(x)·sin` uniformly and the tail self-cancels.

Why this matters: with denominator = rotary width, the *existing* 64
coordinates would spin **4× faster** than a full-dim RoPE at the same θ —
the frequency spectrum of the head would shift wholesale. With
denominator = head dim, the active subspace keeps *exactly the
frequencies the first quarter of a full-dim θ=10⁶ RoPE would have had*.
The head gains 192 position-blind channels (which the MLP/Norms can use
for content) without re-tuning the position code it already trusts.

## Where the numbers come from

- `global_rotary_partial_factor = 0.25` ⇒ rotary subspace
  `models/config.py:ModelConfig.global_rotary_dim` = 64 (property, None
  if the factor doesn't divide evenly). Validated even, positive, ≤ head
  dim in `models/config.py:ModelConfig.__post_init__`.
- The per-layer θ/factor pair is selected in
  `models/attention.py:Attention.__init__` — globals get
  `(θ=10⁶, factor 0.25)`, locals get `(θ=10⁴, factor 1.0)` — and stored
  as a non-persistent `inv_freq` buffer on the layer.

## Absolute positions only

`models/attention.py:rope_cos_sin` turns `position_ids [B, T]` into
`cos/sin [B, T, head_dim]`. Positions are **absolute token positions**,
never cache-slot indices — under
[cross-layer KV sharing](cross-layer-kv-sharing.md) the cache trims local
history, so slot indices and token positions diverge; positions are
derived from validity counts (`models/cache.py:ProducerKVCache.valid_counts`)
exactly once per step in
`models/transformer.py:Gemma4LiteModel.forward_hidden`. Left-padded rows
get correct positions because padding tokens never count (the
cumsum-of-valid convention). Getting this wrong is the classic
"quality degrades after N steps" bug — see
[troubleshooting](../guides/troubleshooting.md).

## What is asserted where

- The 1:2 head-dim relation that makes the quarter-width an integer:
  `models/config.py:ModelConfig.__post_init__`.
- Eager↔SDPA parity across the window edge (T=33, window 32 in the tiny
  config): `tests/test_backends.py`.
- Upstream numeric parity of the whole path: `tests/test_reference.py`
  (pinned `transformers==5.15.1`, SHA-verified at import — see
  [sources](../sources.md)).
