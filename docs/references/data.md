# References: the data adapter and the pipeline shim

> Audience: intermediate. Covers every public symbol of `data/dataset.py`
> and `data/prepare_data.py`. The shared pipeline itself is documented in
> `shared_data/documentation/` (workspace, not this repo).

## `data/prepare_data.py` — the GPT-2 contract shim

`data/prepare_data.py:main` is a thin CLI over the workspace
`shared_data` pipeline (download → clean → dedup → tokenize → pack); the
shim pattern mirrors DiffusionGemma-Lite. What this repo adds:

- The tokenizer contract, as module constants:
  `data/prepare_data.py:GEMMA4_TOKENIZER_NAME` (`"gpt2"`),
  `data/prepare_data.py:GEMMA4_VOCAB_SIZE` (50,257),
  `data/prepare_data.py:GEMMA4_EOS_TOKEN_ID` (50,256),
  `data/prepare_data.py:GEMMA4_PAD_TOKEN_ID` (50,256 — EOS doubles as
  pad; packing marks padding via the document stream, and attention
  padding is a *runtime* concern via `key_valid`).
- `_apply_gemma4_defaults` materializes a project-local
  `data/data_config.yaml` (universal config + this tokenizer block) so
  the pipeline packs to the model's vocab — a different tokenizer with a
  same-shaped vocab file is the classic silent-corruption failure, caught
  downstream by `data/dataset.py:ShardWindows`.
- `DEFAULT_DATA_ROOT` = `<repo>/data/pretrain_corpus`; shards land in
  `<data_root>/shards/`. `LLM_DATA_ROOT` is exported because the pack
  stage re-resolves the root from the environment in its subprocess.
- `_require_shared_data` — a clean `FileNotFoundError` when the workspace
  package is missing instead of a confusing import error.

## `data/dataset.py` — windows over packed shards

### `data/dataset.py:ShardWindows`

A `torch.utils.data.Dataset` of **flat `seq_len`-token windows** — no
next-token shift here: the loss owns the +1 shift
(`training/losses.py:chunked_causal_ce`), so windows are plain slices of
the logical stream.

Construction validates before anything trains
(`data/dataset.py:ShardWindows.__init__`):

1. Wraps `shared_data.dataset.ShardDataset` — the pipeline keeps
   manifest validation, checksums, dtype, and the mmap; nothing is
   re-implemented here.
2. **Tokenizer-name check** — manifest must say `gpt2`; a manifest from
   another tokenizer family is silently-wrong data (same file shape,
   different token semantics) and is rejected by name.
3. **Existence check** — every manifest-listed shard exists.
4. **Length check** — on-disk token count equals the manifest record.
5. **Size check** — at least one full window.

`data/dataset.py:ShardWindows.validate` — a *sampled* vocab-boundary scan
(~16 windows): packed IDs must stay under the contract vocab; catches a
same-shape/different-tokenizer corpus. Cheap enough for preflight, not
an exhaustive scan — the docstring says so plainly.

`data/dataset.py:ShardWindows.__getitem__` — window `w * seq_len ..
+seq_len` is assembled from consecutive memmap reads; a window crossing a
shard boundary is served by two reads; only the final partial window of
the whole stream is dropped. No corpus copy ever lands in host memory —
reads go through the mmap (`ShardWindows` docstring).

### `data/dataset.py:ShuffledRangeSampler`

Deterministic, resumable window order: permutation fixed by
`(seed, n_windows)` (`np.random.default_rng(seed).permutation`);
`offset` restarts mid-order after a checkpoint resume without
regenerating any draws. House contract shared with DiffusionGemma-Lite
and `shared_data.loader`.

### `data/dataset.py:build_dataloader`

`ShardWindows` + `ShuffledRangeSampler` + `DataLoader(drop_last=True)`;
`offset_windows` resumes the shuffle order (pairs with
`batches_consumed` from the checkpoint metadata). The training loop
consumes `(ids, ids)` — labels are the inputs, the +1 shift happens in
the loss (`training/pretrain.py:main`).

## Data flow summary

```
shared_data pipeline (workspace)          this repo
download/clean/dedup/tokenize/pack   →   shards/*.bin + manifest.json
                                          data/prepare_data.py:main   (contract shim)
                                          data/dataset.py:ShardWindows (validate + mmap windows)
                                          data/dataset.py:build_dataloader (shuffled, resumable)
                                          training/pretrain.py:train    (loss owns the +1 shift)
```
