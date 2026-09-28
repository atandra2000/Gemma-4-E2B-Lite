# Per-Layer Embeddings (PLE) — the token signal that reaches every layer

> Audience: intermediate. Builds its own prerequisites (norms, embeddings).
> Citations use `file.py:Symbol`, verified by `tests/test_doc_refs.py`.

A plain transformer injects token identity exactly once, at the embedding
table. Every deeper layer sees the token only through the residual stream —
by the time gradients and depth have reshaped it, "which token is this?"
is a noisy, indirect quantity. Per-layer embeddings (PLE) fix that: the
model computes a **fresh, normed, token-identity signal for every layer**
and adds it at that layer's input. This is one of the four mechanisms
Gemma-4-E2B-Lite preserves from upstream Gemma E2B (with
[local/global attention](local-global-attention.md),
[proportional partial RoPE](proportional-partial-rope.md) and
[cross-layer KV sharing](cross-layer-kv-sharing.md)).

## The two branches

`models/ple.py:PLE` builds the signal from two independent branches:

1. **Identity branch** — a dedicated token table
   (`models/ple.py:PLE.table`, shape `[V, L·P]`, here V=50,257, L=20
   layers, P=64) scaled by `√P` and reshaped to `[B, T, L, P]`. This is
   pure token identity: no context, same value for the same token at
   every position and every call.
2. **Projected branch** — the *already-scaled* main embedding
   (hidden states × `√D`, done in `models/transformer.py:Gemma4LiteModel.forward_hidden`)
   is projected to `L·P` by `models/ple.py:PLE.input_proj` (factor
   `1/√D`), reshaped, and normalized by a P-dim RMSNorm
   (`models/ple.py:PLE.projection_norm`).

The two are mixed by `models/ple.py:PLE.forward` as
`(projected + identity) · 1/√2` (`models/ple.py:PLE.mix_scale` = `2⁻⁰·⁵`)
and returned as one stack `[B, T, L, P]`.

### The scaling facts (all in `models/ple.py`)

| Quantity | Value | Where |
|---|---|---|
| identity table scale | `√P` (per-branch RMS) | `models/ple.py:PLE.forward` |
| projected-branch scale | `1/√D` | `models/ple.py:PLE.forward` |
| branch mix | `(proj + ident) · 2⁻⁰·⁵` | `models/ple.py:PLE.mix_scale` |

An honesty note the code itself makes: the `1/√D` factor is **inert** —
the RMSNorm that immediately follows is scale-invariant, so removing it
changes nothing numerically. It is kept because the upstream/design
contract (`design §4`) states it, and this repo pins source semantics
(`models/ple.py` module docstring). The `√P` on the identity branch is
*not* inert: there is no norm on that branch before the mix.

## Consumption: one slice per layer

The full model computes the stack **once per forward** in
`models/transformer.py:Gemma4LiteModel.forward_hidden` and hands layer *i*
its slice `ple_signal[:, :, i, :]` — the upstream model-loop contract
(`models/ple.py:PLE` docstring). Nothing else in the network reads the
stack; a layer never sees another layer's signal.

## Injection inside the block

Each `models/transformer.py:Block` injects the signal after the MLP
residual, as a third gated residual:

```
residual = h
h = gelu_tanh(ple_gate(h))      # models/transformer.py:Block — [B,T,D] -> [B,T,P]
h = h * ple_signal              # broadcast over the P dim
h = post_ple_norm(ple_proj(h))  # back to [B,T,D], normed
h = residual + h
```
(# illustrative — the PLE injection lines of `models/transformer.py:Block.forward`, verbatim arithmetic)

So PLE is not an input-side trick only: it is a *learned gate* — the layer
decides, per position and per P-channel, how much of the raw token signal
to let in. `models/transformer.py:Block.ple_gate` / `Block.ple_proj` are
the D→P and P→D matrices; they are the bulk of
`count_large_matrices(config)["ple_projections"]` in the analytic ledger
(`models/config.py:count_large_matrices`).

## The norm underneath (and two source conventions worth knowing)

All norms here are `models/ple.py:RMSNorm`: FP32 compute,
`x · rsqrt(mean(x²) + eps) · weight`, with `eps = 1e-6`
(`models/config.py:ModelConfig.rms_eps`). Two conventions are pinned from
upstream v5.15.1 and are *not* generic transformer folklore:

- The weight is a **plain multiplier** — not the `(1 + weight)` offset of
  older Gemma generations (`models/ple.py:RMSNorm` docstring).
- A norm can exist **with no weight at all**:
  `models/ple.py:RMSNorm.with_scale=False` is how the V-norm normalizes
  on producer layers (`models/attention.py:Attention`), and the parameter
  ledger asserts no `v_norm` weight exists
  (`scripts/parameter_budget.py:reconcile_with_ledger`).

## Size accounting

Upstream uses P=256; this repo reduces to P=64
(`models/config.py:ModelConfig.ple_dim`). The PLE parameter cost is exact
and ledger-reconciled (production config): token table 64,328,960 +
projections 2,949,120 + projection norm 64 — together ~19% of the
348,965,184 exact total (`scripts/parameter_budget.py:main` prints the
full breakdown; `models/config.py:memory_assumptions` carries the memory
side). All counts are measured on the meta device by
`scripts/parameter_budget.py:reconcile_with_ledger`, not estimated.

## Tests

`tests/test_ple.py` covers the branches (15 tests): branch scales, the
mix, the per-layer slice contract, and weight-free normalization.
