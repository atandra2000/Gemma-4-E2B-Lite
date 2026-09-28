# References: the producer cache and greedy generation

> Audience: expert. Covers every public symbol of `models/cache.py` and
> `inference/generate.py`. Theory: [cross-layer KV sharing](../concepts/cross-layer-kv-sharing.md).

## `models/cache.py`

### One constant with a job

`models/cache.py:MASK_FILLER` — additive-mask filler is `float32 finfo.min`,
not −inf. An all-masked row (a padding query with no valid keys) must
softmax to uniform garbage, never NaN: NaN keys would reach valid queries
as `0 · NaN` in later layers. See
[troubleshooting](../guides/troubleshooting.md#nan-discipline-attention).

### `models/cache.py:ProducerKVCache`

Per-request, resettable, never module-global. State per producer layer:

| field | contents |
|---|---|
| `_history` | committed K/V per producer layer (post-`finish_step`) |
| `_valid` | per-layer key-validity vector `[B, 1, kv, 1]` |
| `_inflight` | current-call states (history + this call's chunk) |
| `_dropped` | per-row count of valid keys lost to local trimming (position offset) |

- `models/cache.py:ProducerKVCache.stream_for` — which producer's stream
  a layer reads (`models/config.py:ModelConfig.kv_producer`; producers
  read themselves).
- `models/cache.py:ProducerKVCache.producer_append` — append chunk K/V
  `[B, 1, C, hd]` once; first call stores the chunk as history, later
  calls concat onto it; validity grows in lockstep (real `key_valid` or
  all-ones on padding-free decode steps). Returns the full current-call
  states for the producer's own queries.
- `models/cache.py:ProducerKVCache.consumer_states` — the consumer's
  read: exactly the producer's inflight tensors (aliasing, no copies —
  asserted by `models/cache.py:ProducerKVCache.unique_storages`).
- `models/cache.py:ProducerKVCache.attention_mask` — additive
  `[B, 1, q_len, kv_len]` mask per layer type. Key positions =
  `dropped + cumsum(validity) − 1` (padding occupies slots but not
  positions); query positions arrive from the caller. Keep-set:
  `key_pos ≤ query_pos`, plus the local window when the layer is local,
  plus validity; skipped entries filled with `models/cache.py:MASK_FILLER`.
  The producer's own mask is built *before* it appends, so the stored
  validity is padded with the current chunk on the fly.
- `models/cache.py:ProducerKVCache.finish_step` — called once per step by
  `models/transformer.py:Gemma4LiteModel.forward_hidden` after the last
  block: drops inflight views, trims each **local** stream to
  `window − 1` keys (accounting dropped validity into `_dropped` so
  positions stay absolute), commits to history. **Global streams are
  never trimmed.**
- `models/cache.py:ProducerKVCache.valid_counts` — per-row count of valid
  tokens ever appended, read from the untrimmed global stream: each row's
  next absolute position.
- `models/cache.py:ProducerKVCache.history_length` — committed length for
  a producer index or a layer-type name (all local streams share length).
- `models/cache.py:ProducerKVCache.num_tokens` — the absolute token clock
  (untrimmed global history length).
- `models/cache.py:ProducerKVCache.unique_storages` — distinct `data_ptr`s
  held persistently: the no-consumer-copies invariant
  (`tests/test_cache.py`).
- `models/cache.py:ProducerKVCache.reset` — clear everything (new request).

## `inference/generate.py`

`inference/generate.py:generate` — greedy decoding over the producer
cache. Inputs: `input_ids [B, T]` prompts (or `[T]`), `max_new_tokens`,
optional `chunk_size` (prefill in chunks), optional `key_valid [B, T]`
marking left-padding. Returns `(tokens [B, steps], finished [B])`; rows
that hit EOS carry trailing `eos_id` entries.

Flow:

1. One `models/cache.py:ProducerKVCache` for the whole batch; every row
   is a stream consumer of the same producers (independent requests share
   the call, masked by validity).
2. **Chunked prefill**: feed `[start:stop)` slices; each chunk's forward
   masks against history + chunk. The row's *trigger* logits — at its
   last valid prompt token (`last_valid`, computed from `key_valid`) —
   are captured when their chunk passes.
3. **Decode loop**: feed every row each step (finished rows emit
   `eos_id`); stop early when all rows finish.
4. Convention is pinned: raw GPT-2 BPE IDs, no scaffolding,
   EOS 50,256 terminates (`models/config.py:ModelConfig.eos_id`) —
   identical in every ablation arm and eval script (AGENTS.md rule).

Position ids are derived inside the model from validity counts —
`generate` never passes them, so left-padded rows decode at correct
absolute positions (see
[cross-layer KV sharing](../concepts/cross-layer-kv-sharing.md#masks-over-per-row-positions)).

Minimal use:

```python
# illustrative — untrained weights: correctness demo, not quality
from models.config import tiny_config
from models.transformer import Gemma4LiteModel
from inference.generate import generate
model = Gemma4LiteModel(tiny_config())
tokens, finished = generate(model, input_ids, max_new_tokens=16)
```
