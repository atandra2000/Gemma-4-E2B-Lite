# AGENTS.md — Gemma-4-E2B-Lite

The `LLM/AGENTS.md` workspace rules and `CoreProjects` instructions apply.
This file adds project-specific rules.

## Source contract

The design (llm-research/DESIGN-gemma-4-e2b-lite.md §2) and its pinned
revisions are authoritative. Pins are recorded and verified in
`docs/sources.md`:

- HF `google/gemma-4-E2B` config revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`.
- Transformers tag `v5.15.1`, `modeling_gemma4.py` SHA-256
  `4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`.

Never silently substitute a sibling model's block or a newer upstream
version. Any source disagreement is resolved before numerical-parity gates.

## Implementation rules

- **Raw PyTorch only.** No HF Trainer, no Lightning. Transformers is an
  optional test oracle (tiny random models), never the implementation.
- **Test-only deps must stay test-only.** Runtime imports must not require
  the oracle extra.
- **Markers:** `gpu` / `slow` / `heavy` gate non-CPU tests; they auto-skip
  on this macOS host. Never delete GPU-gated tests because they skip.
- **Packages are created only when a task needs them.** No speculative dirs.
- **No checkpoints, corpora or token caches in git.**
- **No copied status.** Do not import other projects' test counts, CI
  workflows, metrics or trained-model claims. Headline numbers are measured
  here or tagged `[INFERENCE]`.
- **Completion states are separate:** CPU-correct, A100-measured, trained.
  Never report a later state as passed because an earlier one is.
- **Generation convention (pinned):** GPT-2 BPE, raw token IDs, no system or
  turn scaffolding, EOS 50,256 terminates. Identical in every ablation arm
  and evaluation script.

## Documentation gates (hard rules)

- **Both doc gates must pass before any change lands:**
  `python3 scripts/check_docs.py --coverage --links` and
  `python3 -m pytest tests/test_doc_refs.py` (collected by the default
  `pytest` run — `pytest.ini` testpaths include it).
- **Citation style:** every code reference in prose is `file.py:Symbol`;
  line-number anchors fail the gate. Python snippets carry
  `# illustrative` or `# verified`.
- **Coverage:** every public symbol in `models/`, `training/`, `data/`,
  `inference/`, `utils/`, `scripts/` must be cited in the docs. Renaming
  or adding a public symbol means updating the docs in the same change.
- **Doc truth:** headline numbers in docs carry a measured source;
  estimates are tagged `[INFERENCE]`. The docs live under `docs/`
  (nav map: `docs/README.md`; audit: `docs/AUDIT.md`).
