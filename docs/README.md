# Gemma-4-E2B-Lite — Documentation Index

Four tracks: `concepts/` (theory from first principles, one doc per
conceptual cluster), `references/` (code-keyed walkthroughs), `guides/`
(how-to), and `training.md` (the applied pipeline). Every code citation
uses `file.py:Symbol`, verified by `tests/test_doc_refs.py`; the
standalone checker is `scripts/check_docs.py`. Honesty rule: headline
numbers are measured in this repo or tagged `[INFERENCE]` — see the
README status table and [AUDIT.md](AUDIT.md).

## Corpus size

Measured 2026-09-21 (`wc -w`, content docs; excludes this nav table's own
row arithmetic, no session artifacts exist in `docs/`):

| Track | Files | Words |
|---|---:|---:|
| concepts/ | 4 | 2,374 |
| references/ | 4 | 2,807 |
| guides/ | 4 | 1,867 |
| training.md + AUDIT.md + sources.md | 3 | 2,149 |
| nav (this file) | 1 | 532 |
| Top-level trio (README + AGENTS + SKILLS) | 3 | 3,576 |
| **Total** | **19** | **13,305** |

## Learning paths

- **Beginner:** [quickstart](guides/quickstart.md) →
  [PLE concept](concepts/per-layer-embeddings.md) →
  [references/models.md](references/models.md) → [glossary](guides/glossary.md)
- **Intermediate:** [references/data.md](references/data.md) →
  [training.md](training.md) → [references/training.md](references/training.md) →
  [cross-layer KV sharing](concepts/cross-layer-kv-sharing.md)
- **Expert:** [proportional partial RoPE](concepts/proportional-partial-rope.md) →
  [references/cache.md](references/cache.md) →
  [local/global attention](concepts/local-global-attention.md) →
  [AUDIT](AUDIT.md)

(Step-table versions with "what you will know after" columns:
[guides/learning-paths.md](guides/learning-paths.md).)

## Concepts track

| Doc | Audience | Core topics |
|---|---|---|
| [per-layer-embeddings.md](concepts/per-layer-embeddings.md) | intermediate | identity vs projected branches, √P and 1/√D scaling, the inert-factor honesty note, P-slice consumption, weight-free norms |
| [local-global-attention.md](concepts/local-global-attention.md) | intermediate→expert | 5-layer period, window 512 vs full reach, pinned 1:2 head dims, MQA, scale 1.0, eager vs SDPA, the MASK_FILLER NaN guard |
| [proportional-partial-rope.md](concepts/proportional-partial-rope.md) | expert | θ=10⁶, factor 0.25, full-head-dim denominator, zero-frequency tail, absolute positions |
| [cross-layer-kv-sharing.md](concepts/cross-layer-kv-sharing.md) | expert | producer map, training dict vs ProducerKVCache, trimming and the global clock, per-row position masks, checkpoint-safe aliasing |

## References track

| Doc | Walks through |
|---|---|
| [references/models.md](references/models.md) | `models/config.py` (every key + validation + ledger), `models/ple.py`, `models/attention.py`, `models/transformer.py`, `scripts/parameter_budget.py` |
| [references/cache.md](references/cache.md) | `models/cache.py` (all methods), `inference/generate.py` |
| [references/training.md](references/training.md) | `training/losses.py`, `training/pretrain.py`, `utils/checkpoint.py` |
| [references/data.md](references/data.md) | `data/prepare_data.py`, `data/dataset.py` |

## Guides track

| Doc | Contents |
|---|---|
| [guides/learning-paths.md](guides/learning-paths.md) | three tier tables (beginner / intermediate / expert) |
| [guides/quickstart.md](guides/quickstart.md) | install → tests → ledger → generate → data → smoke train |
| [guides/troubleshooting.md](guides/troubleshooting.md) | every designed-to-raise failure, symptom → meaning → fix |
| [guides/glossary.md](guides/glossary.md) | notation, acronyms, config-key glossary |

## File→doc map

| Module | Documented in |
|---|---|
| `models/config.py` | [references/models.md](references/models.md); validation + ledger |
| `models/ple.py` | [concepts/per-layer-embeddings.md](concepts/per-layer-embeddings.md); [references/models.md](references/models.md) |
| `models/attention.py` | [concepts/local-global-attention.md](concepts/local-global-attention.md); [concepts/proportional-partial-rope.md](concepts/proportional-partial-rope.md); [references/models.md](references/models.md) |
| `models/transformer.py` | [references/models.md](references/models.md); [concepts/per-layer-embeddings.md](concepts/per-layer-embeddings.md) (injection) |
| `models/cache.py` | [concepts/cross-layer-kv-sharing.md](concepts/cross-layer-kv-sharing.md); [references/cache.md](references/cache.md) |
| `training/losses.py` | [references/training.md](references/training.md); [training.md](training.md) |
| `training/pretrain.py` | [training.md](training.md); [references/training.md](references/training.md) |
| `inference/generate.py` | [references/cache.md](references/cache.md) |
| `data/prepare_data.py` | [references/data.md](references/data.md) |
| `data/dataset.py` | [references/data.md](references/data.md) |
| `utils/checkpoint.py` | [references/training.md](references/training.md) |
| `scripts/parameter_budget.py` | [references/models.md](references/models.md) (ledger reconciliation) |
| `tests/` (111 tests) | [AUDIT.md](AUDIT.md) (verification record); per-mechanism notes in each concept doc |

## Top-level docs

- [README](../README.md) — public overview, status table, quick start
- [AGENTS](../AGENTS.md) — pins, implementation rules, doc-gate hard rules
- [SKILLS](../SKILLS.md) — measured developer workflows
- [sources](sources.md) — pinned upstream revisions + oracle environment
- [diagrams/](diagrams/) — Archify HTML diagrams + PNG renders
