# Gemma-4-E2B-Lite — Documentation & Codebase Audit

> Scope: the whole repo (1,796 lines of Python across 14 modules + the
> test suite) against the docs taxonomy introduced 2026-09-21. This audit
> is dated and reproducible — every verification run below names its
> command and result, and the two doc gates are wired into the default
> `pytest` run so the audit's claims cannot silently rot.

## Verification runs

| # | Command | Result |
|---|---|---|
| 1 | `python3 -m pytest -q` | **115 passed** (111 model/repo tests + 4 doc-gate tests), 14 warnings, 31.45 s, CPU, 2026-09-21 |
| 2 | `python3 scripts/check_docs.py --coverage --links` | **all PASS** — 19 docs, 246 anchors; resolution PASS, coverage PASS (every public symbol of the 12 coverage modules cited), line anchors PASS, links PASS |
| 3 | `python3 -m pytest tests/test_doc_refs.py -q` | **4 passed** in 1.17 s — the gate under pytest, collected by the default run |
| 4 | `python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml` | `parameters (unique storages): 348,965,184` · `unit layer-scalar buffers: 20` · `reconciliation: matches the analytic ledger` (design-row lines all `ok`) |
| 5 | `wc -w` corpus recount | 13,305 words across the 19 content docs — matches the nav-map table (`docs/README.md`, dated 2026-09-21) |

## 1. State of the docs — what was already excellent

- **README status discipline.** The README's three-completion-states
  table (CPU-correct ✅ / A100-measured ❌ / trained ❌) with per-item
  evidence is the portfolio's cleanest honesty device; this audit adopts
  it rather than duplicating it.
- **`docs/sources.md`.** Pinned upstream revisions with SHA-256
  verification and the oracle environment record — the parity claims in
  the docs trace to a verifiable pin.
- **`AGENTS.md`.** The no-copied-status and separate-completion-states
  rules are exactly what keeps future docs honest; the doc-gate rules
  added 2026-09-21 make the gates binding.
- **Diagrams.** Four Archify HTML diagrams with PNG renders (System
  render pending — README updated to say so rather than link a
  nonexistent file).
- **Tests as documentation.** 111 tests encode the semantics the docs
  describe (window-edge parity, aliasing invariants, bit-identical
  resume); the doc gate binds the prose to those symbols.

## 2. Findings — docs↔code misalignment

| ID | Severity | Finding | Evidence | Resolution |
|---|---|---|---|---|
| G1 | minor | README linked a nonexistent System-diagram PNG | `docs/diagrams/` holds only `gemma4-system.{html,architecture.json}`; the gate flagged the link when first run | README atlas row now says "PNG pending"; no dead link |
| G2 | minor | No doc-gate coverage at all before this pass | Wave-2 gap-matrix row: 1 doc file / 64 lines / 7 anchors / no gates | Both gates landed; 235 anchors across 18 docs, coverage PASS |
| G3 | minor | Nav map, guides, AUDIT, concepts, references did not exist | same | Landed 2026-09-21 (this pass) |
| G4 | open | No hardware measurements anywhere (A100 session pending) | README status table; `docs/training.md` §7 | By design — stays open until Task 11; all throughput/VRAM/quality cells remain empty, numbers tagged `[INFERENCE]` when estimated |
| G5 | open | 8B-token run, ablations, quality report not started | Tasks 12–13 | Not started by design; unblocked by hardware |

G4/G5 are status findings, not doc defects: the docs deliberately carry
empty result cells rather than predictions.

## 3. From-scratch explanation of the codebase

The repo is a from-scratch, raw-PyTorch replica of the **Gemma E2B text
mechanisms** at 349M scale, built to pretrain on one A100 80GB.

- **Contract layer** — `models/config.py:ModelConfig` is a frozen,
  validated dataclass whose `__post_init__` encodes every architecture
  invariant (1:2 head dims, final layer global, every consumer has a
  same-type KV producer below `share_boundary`, rotary subspace even,
  scale pinned 1.0). Named configs (`production_config`, `tiny_config`)
  share the validation; the YAML loaders reject unknown keys.
- **Signal layer** — `models/ple.py:PLE` builds a `[B, T, L, P]`
  token-identity stack (identity table ×√P + projected scaled-embedding
  branch ×1/√D, RMSNormed, mixed ×2⁻⁰·⁵); each of the 20
  `models/transformer.py:Block`s gates its own slice into the residual
  stream (`ple_gate` → `ple_proj`, post-normed) after the MLP.
- **Attention layer** — 16 local layers (window 512, full-dim RoPE
  θ=10⁴, head dim 128) alternate with 4 global layers (full reach,
  partial RoPE θ=10⁶ factor 0.25, head dim 256) at indices 4, 9, 14, 19.
  Six query heads share one KV head; Q/K are RMSNormed; scale is 1.0
  (`models/attention.py:Attention`). Both backends (eager reference and
  SDPA) are parity-gated.
- **Sharing layer** — layers 0–9 produce K/V, 10–19 alias a same-type
  producer (`models/config.py:ModelConfig.kv_producer`). Training stores
  autograd-connected producer states in a plain dict; inference uses
  `models/cache.py:ProducerKVCache` (append-once producers, aliasing
  consumers, local streams trimmed to `window−1`, global stream as the
  absolute clock, per-row position masks with the `MASK_FILLER`
  all-masked-row NaN guard).
- **Objective layer** — `training/losses.py:chunked_causal_ce` applies
  the tied head, softcap, +1 shift and CE in chunks under non-reentrant
  checkpointing; full `[B, T, V]` logits never materialize. The data
  path ships flat windows because the loss owns the shift.
- **Persistence layer** — `training/pretrain.py:train` accumulates by
  actual valid tokens, skips nonfinite updates, and saves an atomic
  generation per step (`utils/checkpoint.py:CheckpointManager`: weights
  + optimizer + RNG + metadata behind a pointer rename); resume is
  bit-identical (`tests/test_resume.py`).
- **Verification layer** — `scripts/parameter_budget.py:reconcile_with_ledger`
  reconciles the analytic ledger against the meta-device instantiation
  (348,965,184 exact); `tests/test_reference.py` proves parity against
  the pinned upstream source; the doc gates prove the docs against the
  code.

## 4. Modification plan (priority order)

1. **Keep the gates green as code evolves** — any public-symbol change
   must update a doc in the same change (AGENTS.md hard rule now).
2. **Task 11 (A100)** — when hardware lands, fill the README status
   table and `docs/training.md` §7 with measured receipts; retire no
   `[INFERENCE]` tag until its number is measured.
3. **Task 12–13 (training + quality report)** — the AUDIT verification
   table gains the run receipts; `docs/AUDIT.md` gets re-dated, not
   rewritten.
4. **System-diagram PNG** — render and swap the "PNG pending" cell
   (G1).
5. **Depth growth rule** — doc depth scales with measurements (G4), not
   padding: new sections appear when new evidence does.

## 5. Acceptance criteria for "audit complete"

- [x] Both doc gates pass and are included in the default `pytest` run
  (RUN1/RUN2/RUN3 outputs above).
- [x] Every public symbol of the 12 coverage modules is cited ≥ 1×
  (RUN2 coverage line).
- [x] Nav map corpus table measured (`wc -w`) and dated (RUN5;
  `docs/README.md`).
- [x] Every finding G1–G3 resolved; G4/G5 recorded as status-open with
  owner tasks, not doc defects.
- [ ] Re-audit after the A100 session (Task 11) — the only remaining
  trigger; this audit otherwise stands until code changes.
