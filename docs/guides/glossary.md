# Glossary

Notation, then acronyms, then config keys (cross-linked to
[references/models.md](../references/models.md)).

## Notation

| Symbol | Meaning | Set by |
|---|---|---|
| `B, T` | batch, sequence length | runtime tensors |
| `D` | hidden/residual width — 768 | `hidden_dim` |
| `P` | PLE width — 64 | `ple_dim` |
| `L` | layer count — 20 | `n_layers` |
| `V` | vocab size — 50,257 | `vocab_size` |
| `hd` | per-layer head dim — 128 (local) / 256 (global) | `head_dim(i)` |
| `[B, T, L, P]` | PLE signal stack; layer *i* reads slice `[:, :, i, :]` | `models/ple.py:PLE.forward` |
| `θ` | RoPE base — 10⁴ local, 10⁶ global | `local_rope_theta` / `global_rope_theta` |
| `window` | local reach — 512 keys | `local_window` |
| `softcap` | `30·tanh(x/30)` on logits | `logit_softcap` |
| `share_boundary` | producer/consumer split — 10 | `share_boundary` |
| `-100` | ignored-label sentinel (CE convention) | `training/pretrain.py:IGNORE_ID` |

## Acronyms

| Term | Meaning | Where |
|---|---|---|
| PLE | Per-Layer Embeddings — token-identity signal injected at every layer | [concept](../concepts/per-layer-embeddings.md) |
| MQA | Multi-Query Attention — 6 query heads share 1 KV head per layer | [local/global](../concepts/local-global-attention.md) |
| KV sharing | producer prefix computes K/V; consumer suffix aliases it | [concept](../concepts/cross-layer-kv-sharing.md) |
| RoPE | Rotary position embedding | [concept](../concepts/proportional-partial-rope.md) |
| partial RoPE | only a fraction of head coordinates rotate (0.25 on globals) | same |
| SDPA | `torch.nn.functional.scaled_dot_product_attention` backend | `models/attention.py:Attention.forward` |
| CE | cross-entropy (chunked, causal, softcapped here) | [references/training.md](../references/training.md) |
| BF16 | bfloat16 autocast precision (A100 plan) | `configs/pretrain_a100.yaml` |
| EOS | end-of-sequence token — 50,256, pinned terminator | `eos_id` |
| LM head | vocabulary projection (tied to the embedding matrix here) | `models/transformer.py:Gemma4LiteModel.forward` |
| oracle | pinned upstream implementation used for numeric parity | [sources](../sources.md) |

## Config keys (model section)

Every key with its validation: [references/models.md](../references/models.md)
§"Config keys". The ones with non-obvious semantics:

- `share_boundary` — layers `< 10` produce K/V, layers `≥ 10` consume a
  same-type producer's (`models/config.py:ModelConfig.kv_producer`).
- `global_layers` — which layers are global; final layer must be global.
- `mlp_suffix_multiplier` — FFN width doubles on layers `≥ share_boundary`
  (3,072 → 6,144), independent of layer type.
- `global_rope_partial_factor` — rotary subspace = factor × 256 = 64;
  frequencies use the **full** head dim as denominator.
- `attention_scale` — pinned 1.0; any other value is rejected.
- `attn_backend` — `"eager"` (reference) or `"sdpa"` (fast, parity-gated).

Training-section keys (`configs/*.yaml` → `training:`): `seq_len`,
`total_tokens` (budget target, 8B), `accumulation_tokens` (65,536 valid
tokens per update), `warmup_fraction` / `lr_final_fraction` (schedule
shape), `microbatch_sizes`, `precision.params` / `precision.autocast`,
`activation_checkpointing`, `torch_compile`, `loss_chunk_tokens`,
`config_hash` (resume-compatibility pin).
