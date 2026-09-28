# Quickstart

Everything below runs on a laptop CPU; no GPU is required until the
Phase-4 A100 session (not yet run — see the README status table).

## Install

```bash
git clone https://github.com/atandra2000/Gemma-4-E2B-Lite.git
cd Gemma-4-E2B-Lite
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runtime deps are `torch` + `pyyaml`; `pytest` and the pinned
`transformers==5.15.1` oracle are test-only (runtime never imports
`transformers` — see [sources](../sources.md)).

## Verify the implementation (2 minutes)

```bash
python3 -m pytest -q            # 111 tests, ~40 s on CPU (measured 2026-09-20)
```

Included by default: the doc gate (`tests/test_doc_refs.py`) — every
`file.py:Symbol` citation in these docs must resolve; run it standalone
as `python3 scripts/check_docs.py --coverage --links`.

## Reproduce the exact parameter count

```bash
python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml
```

Prints the analytic ledger groups, the norm/buffer split, and the
meta-device reconciliation. Expected: `348,965,184` exact,
`reconciliation: matches the analytic ledger`. Any mismatch raises — the
script never prints a wrong total (see
[references/models.md](../references/models.md)).

## Generate from untrained weights

```python
# illustrative — correctness demo; quality claims require a trained model
import torch
from models.config import production_config
from models.transformer import Gemma4LiteModel
from inference.generate import generate

model = Gemma4LiteModel(production_config())
ids = torch.randint(0, 50257, (1, 8))
tokens, finished = generate(model, ids, max_new_tokens=8)
```

Untrained output is noise — this exercises the producer cache, masking
and the pinned EOS convention (`eos_id` 50,256).

## Prepare data (workspace pipeline)

```bash
python3 data/prepare_data.py                  # download → clean → tokenize → pack
python3 data/prepare_data.py --skip-download  # offline smoke of the same path
```

Writes `data/pretrain_corpus/shards/` + manifest, packed with the GPT-2
contract (see [references/data.md](../references/data.md)). Requires the
workspace `shared_data` package beside this repo.

## Smoke-train (deterministic, synthetic)

```bash
python3 training/pretrain.py --config configs/smoke.yaml --checkpoint-dir /tmp/smoke
```

The smoke config is the ONLY synthetic-data configuration (design §5):
tiny model, 32,768 tokens, CPU, FP32. Resume it:

```bash
python3 training/pretrain.py --config configs/smoke.yaml \
    --checkpoint-dir /tmp/smoke --resume
```

## The production command (documented, not yet run)

```bash
python3 training/pretrain.py --config configs/pretrain_a100.yaml
```

Single A100 80GB, BF16 autocast, 8B-token budget. No hardware
measurement exists yet — peak-VRAM and throughput numbers will come from
the Task-11 session and appear with receipts, not predictions.
