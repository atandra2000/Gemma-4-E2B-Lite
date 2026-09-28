# References: the model package — config, PLE, attention, blocks

> Audience: expert. Every public symbol of `models/config.py`,
> `models/ple.py`, `models/attention.py`, `models/transformer.py` is cited
> here; the ledger script closes the loop. Companion references:
> [cache & generation](cache.md), [training stack](training.md),
> [data adapter](data.md).

## `models/config.py` — the architecture contract

### Config keys (`models/config.py:ModelConfig`)

Field names match the `model:` section of `configs/*.yaml`
(`models/config.py:ModelConfig.from_dict` rejects unknown keys;
`models/config.py:ModelConfig.from_yaml` expects a top-level `model:`).

| Key | Value (production) | Meaning |
|---|---|---|
| `vocab_size` | 50,257 | GPT-2 BPE |
| `eos_id` | 50,256 | pinned generation terminator |
| `hidden_dim` | 768 | residual width D |
| `n_layers` | 20 | decoder blocks L |
| `n_heads` | 6 | query heads per layer (one KV head — multi-query) |
| `local_head_dim` | 128 | local head width |
| `global_head_dim` | 256 | global head width; must be exactly 2× local |
| `global_layers` | (4, 9, 14, 19) | zero-indexed global attention layers; final layer global |
| `share_boundary` | 10 | layers < boundary produce K/V; ≥ boundary consume |
| `ple_dim` | 64 | PLE width P (L = n_layers) |
| `mlp_hidden_dim` | 3,072 | base FFN width |
| `mlp_suffix_multiplier` | 2 | FFN width on shared-suffix layers (3,072 → 6,144) |
| `local_window` | 512 | sliding-window reach of local layers |
| `local_rope_theta` | 10⁴ | full-dim RoPE base, local layers |
| `global_rope_theta` | 10⁶ | [proportional partial RoPE](../concepts/proportional-partial-rope.md) base |
| `global_rope_partial_factor` | 0.25 | rotary subspace = 0.25 × 256 = 64 |
| `attention_scale` | 1.0 | pinned; never SDPA's 1/√d |
| `attn_backend` | "eager" | or "sdpa"; parity gated |
| `rms_eps` | 1e-6 | every RMSNorm |
| `logit_softcap` | 30.0 | `30·tanh(logits/30)` |
| `tie_embeddings` | true | head = embedding matrix |

### Validation (`models/config.py:ModelConfig.__post_init__`)

Positive dims; EOS in vocab range; `n_heads · local_head_dim == hidden_dim`;
the 1:2 head-dim relation; even `local_head_dim` (RoPE pairs); global
layers sorted/unique/in-range/**non-empty/final-layer-global**;
`share_boundary` in `[1, n_layers]`; **every consumer has a same-type
producer below the boundary** (via `models/config.py:ModelConfig.shared_layers`
and `models/config.py:ModelConfig.kv_producer`); valid rotary subspace
(`models/config.py:ModelConfig.global_rotary_dim` — even, >0, ≤ head dim);
`attention_scale == 1.0`; backend in {eager, sdpa}.

### Layer maps and per-layer shapes

- `models/config.py:ModelConfig.layer_type` — "local" | "global" (type
  alias `models/config.py:LayerType`).
- `models/config.py:ModelConfig.kv_producer` — self for prefix; last
  same-type prefix layer for consumers (shared locals ← 8, shared
  globals ← 9 here).
- `models/config.py:ModelConfig.head_dim` /
  `models/config.py:ModelConfig.qkv_dim` — 128/768 local, 256/1,536 global.
- `models/config.py:ModelConfig.mlp_hidden` — 3,072 prefix, 6,144 suffix.

### Named configs

`models/config.py:production_config` (the design §3 20-layer model) and
`models/config.py:tiny_config` (12-layer CPU-test model: two full
local/global periods after a 6-layer producer prefix — every consumer
validated).

### The analytic ledger

- `models/config.py:count_large_matrices` — the six large groups; tied
  head counted once; KV projections counted on the producer prefix only.
- `models/config.py:count_norms_and_buffers` — 5 per-layer norms × 20,
  PLE projection norm (64), Q norms (3,072), producer-only K/V norms
  (1,536 — no V-norm weight, upstream convention), final norm (768),
  and the 20 unit `layer_scalar` **buffers** (not parameters).
- `models/config.py:memory_assumptions` — conservative design-§3 GiB:
  16 B/param (weights+grads+AdamW moments+master) on the ledger subtotal,
  plus full-logit rows for B=1, T=4096 (0.38 GiB BF16 / 0.77 GiB FP32).

## `models/ple.py` — signal stack

`models/ple.py:RMSNorm` — FP32 compute, `x·rsqrt(mean(x²)+eps)·weight`;
plain-multiplier weight (not `(1+w)`); `with_scale=False` = weight-free
normalization (the V-norm). `models/ple.py:PLE` — identity table ×√P,
projected (scaled-embedding) branch ×1/√D then P-dim RMSNorm, mixed
×`models/ple.py:PLE.mix_scale` (2⁻⁰·⁵), returns `[B, T, L, P]`. Full
mechanism: [PLE concept](../concepts/per-layer-embeddings.md).

## `models/attention.py` — one layer's attention

RoPE plumbing: `models/attention.py:rotary_inv_freq` (per-pair inverse
frequencies; zero-frequency tail for partial), `models/attention.py:rope_cos_sin`
(absolute positions → cos/sin `[B, T, head_dim]`),
`models/attention.py:rotate_half`, `models/attention.py:apply_rope`
(`x·cos + rotate_half(x)·sin`, broadcast over heads).

Causality: `models/attention.py:causal_mask` — additive `[T, T]`, 0 keep /
−inf skip, optional window (`key ≤ query ∧ query−key < window`), absolute
positions.

`models/attention.py:Attention.__init__` — per-layer geometry from the
config maps; `models/attention.py:Attention.scaling = 1.0`; Q/K/V norms
(V weight-free); producer flag
(`models/attention.py:Attention.store_full_length_kv` — the *last*
same-type prefix layer stores); non-persistent `inv_freq` buffer.

`models/attention.py:Attention.forward` — shape trace
(`[B, T, D]` in):

1. Q: `q_proj` → view `[B, T, 6, hd]` → `q_norm` → rope → `[B, 6, T, hd]`.
2. K/V: consumers alias (`models/cache.py:ProducerKVCache.consumer_states`
   or dict by layer type); producers project one KV head, norm, rope K,
   then `models/cache.py:ProducerKVCache.producer_append` (cache) or store
   into the dict (training, autograd-connected).
3. Broadcast K/V over 6 heads (`expand` — no copy).
4. Backend: eager = explicit softmax in FP32 with the additive mask;
   sdpa = `F.scaled_dot_product_attention(..., scale=1.0)` with the mask
   expanded to `[B, 1, T, kv]`.
5. `o_proj` back to `[B, T, D]`.

Mechanism context: [local/global](../concepts/local-global-attention.md),
[RoPE](../concepts/proportional-partial-rope.md),
[sharing](../concepts/cross-layer-kv-sharing.md).

## `models/transformer.py` — blocks and full model

`models/transformer.py:GatedGeluMlp` — `down(gelu_tanh(gate(x))·up(x))`,
no biases, width from `ModelConfig.mlp_hidden` (3,072 / 6,144).

`models/transformer.py:Block` — five norms (`input_norm`,
`post_attention_norm`, `pre_ffn_norm`, `post_ffn_norm`, `post_ple_norm`),
`models/transformer.py:Block.ple_gate`/`.ple_proj` (D→P, P→D),
`models/transformer.py:Block.layer_scalar` (unit **buffer**, not a
parameter — present so a future learned scalar needs no surgery), and the
residual arithmetic in pinned order: attention → post-norm → add;
FFN → post-norm → add; gated PLE injection → post-norm → add; × layer
scalar. Position ids default to a fresh arange, or — under a cache — are
provided by the caller (see
[cross-layer KV sharing](../concepts/cross-layer-kv-sharing.md));
`models/transformer.py:Block.gradient_checkpointing` switches the
attention call to non-reentrant checkpointing.

`models/transformer.py:Gemma4LiteModel` — scaled embedding (`√D`) + PLE
stack → 20 blocks → `final_norm` → tied head (`F.linear`, no second
storage) → softcap. `models/transformer.py:Gemma4LiteModel.forward_hidden`
returns the normed final hidden `[B, T, D]` (the chunked-CE input);
`models/transformer.py:Gemma4LiteModel.forward` returns softcapped logits
`[B, T, V]`. `models/transformer.py:Gemma4LiteModel.checkpoint_blocks`
toggles recompute for all blocks.

## `scripts/parameter_budget.py` — ledger ↔ instantiation

`scripts/parameter_budget.py:classify_parameter` maps every parameter
name to its ledger group (unknown names raise; `lm_head` and any
`v_norm` weight are hard errors). `scripts/parameter_budget.py:reconcile_with_ledger`
instantiates on the **meta device**, asserts the structural sharing
contract (no consumer K/V weights/norms, exactly `n_layers` unit
scalars), reconciles every group, and returns exact counts — the Task-5
gate behind the 348,965,184 total. `scripts/parameter_budget.py:main`
prints the analytic tables, checks the production config against the
design row (`scripts/parameter_budget.py:DESIGN_SUBTOTAL` = 348,882,944
large-matrix subtotal), then reconciles:

```
$ python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml
  ...
  parameters (unique storages): 348,965,184
  reconciliation: matches the analytic ledger
```
(# verified — exact totals reproduced by the command; full tables elided)
