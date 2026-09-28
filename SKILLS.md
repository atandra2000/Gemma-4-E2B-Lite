# SKILLS.md — Gemma-4-E2B-Lite

> Companion to `AGENTS.md` (pins + rules). Day-to-day developer workflows
> with their last measured results.

## Skill 1: Run the CPU test suite

```bash
cd Gemma-4-E2B-Lite
python3 -m pytest -q
```

Covers attention (masks, RoPE, window edge), cache (aliasing, trimming,
masking), config validation, data adapter, loss, PLE, backend parity,
oracle reference parity, deterministic resume, the training loop, and the
doc gate (`tests/test_doc_refs.py`). Last measured: **111 passed / 0
skipped, 39.15 s** on CPU (2026-09-20, README verification table).

## Skill 2: Reproduce the parameter ledger

```bash
python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml
```

Expected: every design-row line `ok`, then
`parameters (unique storages): 348,965,184` and
`reconciliation: matches the analytic ledger`. Any mismatch **raises** —
the script cannot print a wrong total (Task-5 gate).

## Skill 3: Run the upstream oracle parity gate

```bash
python3 -m pytest tests/test_reference.py -q
```

Needs the pinned oracle (`transformers==5.15.1` + `safetensors`; SHA-256
verified at import — see `docs/sources.md`). Parity covers logits, PLE
signals, per-layer hidden states, loss and mapped gradients at
atol 1e-5 / rtol 1e-4, including a forward crossing the local-window edge.
If an older system `transformers` shadows the pin, follow the environment
record in `docs/sources.md`.

## Skill 4: Docs gates

```bash
python3 scripts/check_docs.py --coverage --links
python3 -m pytest tests/test_doc_refs.py -q
```

Every doc citation is `file.py:Symbol`; line-number anchors fail the
gate; every public symbol in `models/`, `training/`, `data/`,
`inference/`, `utils/`, `scripts/` must be cited at least once. If you
add or rename a public symbol, update its doc in the same change — the
gate fails the build otherwise. Dated runs live in `docs/AUDIT.md`.

## Skill 5: Data prep and preflight

```bash
python3 data/prepare_data.py --skip-download   # offline smoke of the pipeline path
```

Then before any training run, validate the packed corpus against the
model contract:

```python
# illustrative — preflight boundary scan
from data.dataset import ShardWindows
ds = ShardWindows("data/pretrain_corpus", seq_len=4096, vocab_size=50257)
ds.validate()
```

The constructor already rejects tokenizer mismatch, missing shards and
length drift; `validate()` adds the sampled vocab-boundary scan. See
[docs/references/data.md](docs/references/data.md).

## Skill 6: Smoke-train and resume determinism

```bash
python3 training/pretrain.py --config configs/smoke.yaml --checkpoint-dir /tmp/smoke
python3 training/pretrain.py --config configs/smoke.yaml --checkpoint-dir /tmp/smoke --resume
```

The smoke config is the only synthetic-data configuration (design §5).
Resume must continue bit-identically (`tests/test_resume.py`); if a
resumed run diverges, suspect config changes (the `config_hash` gate
should have refused) or non-deterministic init (seed is set before model
construction).

## Pitfalls

- **Local Python version:** PEP-604 annotations (`int | None`) need
  3.10+; the macOS default `python3` (3.9) fails collection. Use 3.10+
  (README badge; `requirements.txt` has no version pin — the interpreter
  is the pin).
- **`LLM_DATA_ROOT`:** `data/prepare_data.py` exports it because the pack
  subprocess re-resolves the data root from the environment — in-process
  overrides don't reach it.
- **EOS convention (pinned):** GPT-2 BPE, raw IDs, no scaffolding,
  EOS 50,256 terminates. Same in every eval script — don't "fix" a
  generator to add chat scaffolding.
- **No copied status:** never quote sibling projects' test counts or
  hardware numbers; headline numbers are measured in this repo or tagged
  `[INFERENCE]` (AGENTS.md).
- **GPU-gated tests:** `gpu`/`slow`/`heavy` markers auto-skip on CPU
  hosts — never delete them because they skip.
- **Synthetic data:** legal only under `configs/smoke.yaml`
  (`data.source: synthetic`). The production config must fail on absent
  or mismatched data, not fall back.
