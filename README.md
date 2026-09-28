<div align="center">

# Gemma-4-E2B-Lite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3DDC84?logo=apache&logoColor=white)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-111%20passing-brightgreen?logo=pytest&logoColor=white)](#-verification)
[![GPU: A100 80GB](https://img.shields.io/badge/GPU-A100%2080GB%20target-76B900?logo=nvidia&logoColor=white)](#-status-measured-vs-pending)

> **Status:** CPU-correct through Phase 3 (Tasks 1–10 of the execution plan) —
> implementation, tests, upstream-oracle parity and deterministic recovery all
> verified locally. The Phase 4 A100 hardware boundary (Task 11) is next:
> **no hardware measurements and no training run yet.**

A from-scratch, raw-PyTorch autoregressive language model preserving the
**Gemma E2B text mechanisms** — per-layer embeddings (PLE), local/global
attention, proportional partial RoPE, and cross-layer KV reuse — sized for
pretraining on a **single NVIDIA A100 80GB**.

**~349M parameters (exactly 348,965,184) · 20 layers · 768 hidden · GPT-2 BPE 50,257 · 8B-token budget · 1× A100 80GB target**

[**Architecture**](#-architecture) · [**Status**](#-status-measured-vs-pending) · [**Quick start**](#-quick-start) · [**Verification**](#-verification) · [**Docs**](#-documentation)

</div>

```bash
python3 -m pytest -q   # 111 passed in ~40 s on a laptop CPU — no GPU required
```

---

## 📖 Overview

**Gemma-4-E2B-Lite** is an E2B-derived **Lite adaptation, not a
checkpoint-compatible reproduction**: vocabulary, width, depth, head sizes and
sharing depth are all reduced from upstream. No vision/audio encoders, MoE,
MTP, diffusion, instruction tuning, distributed training, custom kernels, or
128K-context claims belong to v1. Upstream Transformers is an optional **test
oracle** (tiny random models), never the implementation.

Three mechanisms carry the design:

1. **Per-layer embeddings (PLE).** Every decoder layer receives its own
   token-derived signal, computed once from IDs + scaled embeddings as
   `[B, T, L, P]` with the upstream reciprocal-√ scaling
   (`models/ple.py:PLE`) and injected after each block's MLP
   (`models/transformer.py:Block`).
2. **Local/global attention with cross-layer KV reuse.** Sixteen local layers
   (window 512) alternate with four global layers at 4, 9, 14, 19. Layers 0–9
   **produce** K/V; layers 10–19 **consume** their producer's — half the model
   stores no K/V of its own (`models/attention.py:Attention`,
   `models/cache.py:ProducerKVCache`).
3. **Pinned numerics.** Attention scale 1.0 (never SDPA's default 1/√d), Q/K
   norms (V normalizes without a weight), RMSNorm ε=1e-6, logit softcap 30,
   proportional **partial** RoPE on global layers (θ=10⁶, rotary subspace 64).
   Tiny-model logits, PLE signals, per-layer hidden states, loss and every
   mapped parameter gradient match the pinned `transformers==5.15.1` oracle at
   CPU FP32 atol 1e-5 / rtol 1e-4 (`tests/test_reference.py`).

### How it compares to the rest of the portfolio

| Project | Attention | Position encoding | KV strategy | Generation |
|---|---|---|---|---|
| [LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite) | GQA (8Q/4KV) | RoPE θ=500K | per-layer cache | token AR |
| [DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) | MLA (latent KV) | YaRN (decode only) | latent compression | token AR |
| [GPT-OSS-Lite](https://github.com/atandra2000/GPT-OSS-Lite) | GQA + sliding(128)/full alt + sinks | YaRN 128K | sliding/full split | token AR |
| [Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite) | — (complex SSM) | — | constant state | token AR |
| [DiffusionGemma-Lite](https://github.com/atandra2000/DiffusionGemma-Lite) | block-causal GQA | — | canvas-level | block diffusion |
| **Gemma-4-E2B-Lite** | **local(512)/global 4:1, multi-query (6Q/1KV)** | **proportional partial RoPE θ=10⁶ (global)** | **cross-layer sharing: 10 producers → 20 layers** | **token AR** |

---

## 🗺️ Visual Architecture Atlas

Four standalone Archify diagrams (dark/light themes, no server needed).
Three carry PNG renders; the System diagram's render is pending.

| Diagram | What it shows | Interactive HTML | Visual preview |
|---|---|:---:|:---:|
| **System** | Component ownership: config contract → model → training → data → utils | [Open ↗](docs/diagrams/gemma4-system.html) | PNG pending |
| **Decoder block** | One layer's exact residual arithmetic incl. PLE injection and the layer scalar | [Open ↗](docs/diagrams/gemma4-block.html) | [PNG](docs/diagrams/gemma4-block.visual-check.1440x900.dark.png) |
| **Training loop** | AdamW + warmup/cosine token schedule, accumulation, atomic resume | [Open ↗](docs/diagrams/gemma4-training-loop.html) | [PNG](docs/diagrams/gemma4-training-loop.visual-check.1440x900.dark.png) |
| **Dataflow** | shared_data pipeline → manifest validation → memmapped windows → loss | [Open ↗](docs/diagrams/gemma4-dataflow.html) | [PNG](docs/diagrams/gemma4-dataflow.visual-check.1440x900.dark.png) |

## 🏗️ Architecture

### End-to-end forward pass

```
Input token IDs [B, T]  (GPT-2 BPE, raw IDs — no scaffolding; EOS 50,256 terminates)
      │
      ▼
Main embedding  50,257 → 768, scaled by √768   ──┐
      │                                          ├─► PLE signal [B, T, L=20, P=64]
      ▼                                          │   (reciprocal-√ scaling, once per forward)
20× Decoder Blocks (local 0–19 minus {4,9,14,19} global):
  ┌────────────────────────────────────────────────┐
  │ residual = h                                   │
  │ h = input_norm(h)                              │
  │ h = Attention(h)          local(window 512)    │
  │        or Global          partial RoPE θ=10⁶   │
  │     scale 1.0 · Q-norm · K/V-norm (producers)  │
  │     KV: producers 0–9 write, consumers 10–19   │
  │     alias their same-type producer             │
  │ h = residual + post_attention_norm(h)          │
  │ h = mlp(pre_ffn_norm(h))  gated GELU-tanh      │
  │     768→3072 (local) / 768→6144 (shared suffix)│
  │ h = residual + post_ffn_norm(h)                │
  │ h = ple_proj(gelu_tanh(ple_gate(h))·ple_signal)│
  │ h = residual + post_ple_norm(h)                │
  │ h *= layer_scalar          (unit buffer)       │
  └────────────────────────────────────────────────┘
      │
      ▼
Final RMSNorm → tied head (embedding matrix, F.linear)
      │
      ▼
Logit softcap  30 · tanh(logits / 30)
      │
      ▼
Chunked causal CE (training) — owns its own +1 shift; windows are flat x0 slices
```

### Model specifications

| Parameter | Value | Upstream / adaptation |
|---|---:|---|
| **Total parameters** | **348,965,184** (exact, instantiated & ledger-reconciled) | Reduced from 2.3B effective / 5.1B with embeddings |
| Layers / hidden width | 20 / 768 | Reduced from 35 / 1,536 |
| Query / KV heads | 6 / 1 per layer (multi-query) | Reduced query count, grouping retained |
| Local / global head dim | 128 / 256 (pinned 1:2; global rotary subspace 64) | Preserved |
| Global layers (0-indexed) | 4, 9, 14, 19 | Four local, then one global; final layer global |
| KV sharing boundary | 10 (prefix 0–9 produces; 10–19 consume) | Preserved |
| Local window | 512 | Preserved |
| RoPE | local θ=10,000 full-dim · global θ=1,000,000 partial (factor 0.25) | Source semantics preserved |
| MLP | gated GELU-tanh, 3,072 base; shared suffix doubles to 6,144 | — |
| PLE width | 64 | Reduced from 256 |
| Vocabulary / tokenizer | 50,257 / GPT-2 BPE, EOS 50,256 | Explicit reduction from 262,144 |
| Attention scale / softcap | 1.0 (never SDPA's 1/√d) / 30 | Match source |
| Norm ε / dropout / bias | 1e-6 / 0 / false | Match source |
| Embedding ↔ LM head | tied (single storage) | — |
| Sequence length / token budget | 4,096 / 8B (budget target, not proven optimum) | No length curriculum in v1 |

### Parameter ledger

Reproduce digit-for-digit with `python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml`:

| Large matrix | Parameters |
|---|---:|
| Main embedding (tied, counted once) | 38,597,376 |
| PLE token table (V×L×P) | 64,328,960 |
| PLE projections | 2,949,120 |
| MLP matrices | 212,336,640 |
| Q and output projections | 28,311,552 |
| KV projections (producers only) | 2,359,296 |
| **Subtotal (large matrices)** | **348,882,944** |
| Norms (per-layer ×5, PLE-projection, Q, K/V producer-only, final) | 82,240 |
| **Total (exact, instantiated on meta device and reconciled)** | **348,965,184** |

### Training memory assumptions (conservative, design §3)

| Item | Value |
|---|---:|
| Trainable bytes @ 16 B/param (weights + grads + AdamW moments + master copy) | 5.20 GiB |
| Full logits, B=1 T=4096, BF16 / FP32 | 0.38 / 0.77 GiB |
| Precision plan | FP32 params + BF16 autocast on A100, no GradScaler |
| Chunked CE | chunked over the vocabulary; full logits never materialized in training |

---

## 🚀 Quick start

### Install

```bash
git clone https://github.com/atandra2000/Gemma-4-E2B-Lite.git
cd Gemma-4-E2B-Lite

python3 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt                       # torch + pyyaml + pytest + oracle pin
```

`requirements.txt` is deliberately tiny — raw PyTorch only. The pinned
`transformers==5.15.1` is a **test-only** oracle (runtime code imports no
`transformers`; verified by keeping `sys.modules` clean after importing
`models`).

### Verify on CPU

```bash
python3 -m pytest -q
# 111 passed, 39.15 s (macOS arm64, CPU) — measured 2026-09-20
```

### Generate (untrained weights — correctness demo, not quality)

```bash
python3 -c "
import torch
from models.config import ModelConfig
from models.transformer import Gemma4LiteModel
from inference.generate import generate

cfg = ModelConfig.from_yaml('configs/smoke.yaml')
model = Gemma4LiteModel(cfg).eval()
ids, _ = generate(model, torch.tensor([[464, 3139, 286, 4881]]), max_new_tokens=8)
print(ids.shape)   # torch.Size([1, 8]) = generated steps (prompt not included);
                   # (ids, finished) is returned — greedy, EOS 50256 terminates
"
```

### Data & training

```bash
# 1. Data — the workspace shared_data pipeline (download→clean→tokenize→pack);
#    this repo's shim supplies the GPT-2 tokenizer contract.
python3 data/prepare_data.py            # writes data/pretrain_corpus/shards/ + manifest
python3 data/prepare_data.py --skip-download   # offline smoke of the same path

# 2. Production training (single A100 80GB — CPU is smoke-test only)
python3 training/pretrain.py --config configs/pretrain_a100.yaml

# 3. Deterministic smoke run (the ONLY synthetic-data configuration, design §5)
python3 training/pretrain.py --config configs/smoke.yaml --checkpoint-dir /tmp/smoke

# 4. Parameter budget / ledger reproduction
python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml
```

**Atomic recovery.** `utils/checkpoint.py:CheckpointManager` writes each
generation ({model, optimizer, RNG, metadata}) through sibling temp files
committed by a final atomic pointer rename — a reader sees the whole generation
or the previous one; partial writes are never loadable. `--resume` restores
weights, optimizer and RNG state for bit-identical continuation
(`tests/test_resume.py` verifies deterministic recovery).

---

## 📊 Status: measured vs pending

The three completion states are kept **separate** — a later state is never
reported as passed because an earlier one is:

| State | Status |
|---|---|
| 1. CPU-correct — implementation, tests, oracle parity, deterministic recovery | ✅ **measured here** |
| 2. A100-measured — fit, BF16 parity, throughput, peak memory (Task 11) | ❌ pending — no hardware session yet |
| 3. Trained — checkpoint + published quality report (Tasks 12–13) | ❌ not started |

| Item | Status | Evidence |
|---|---|---|
| Test suite | ✅ **115 passed / 0 skipped** (CPU, 2026-09-28) | `uv run pytest tests/ -q` |
| Exact parameter count | ✅ 348,965,184, instantiated on meta device, reconciles the analytic ledger | `scripts/parameter_budget.py` |
| Upstream oracle parity | ✅ logits, PLE signals, per-layer hidden states, loss, mapped grads @ atol 1e-5 / rtol 1e-4 | `tests/test_reference.py` (pinned `transformers==5.15.1`, SHA-256 verified at import) |
| Eager ↔ SDPA backend parity + checkpoint grad parity | ✅ | `tests/test_backends.py` |
| Atomic resume / deterministic recovery | ✅ bit-identical continuation | `tests/test_resume.py` |
| Manifest-validated data adapter | ✅ tokenizer-name, length and sampled vocab-boundary checks; no corpus copied into host memory | `data/dataset.py:ShardWindows` |
| **Peak VRAM < 72 GiB, tokens/s, BF16 vs FP32 parity** | ❌ **pending — Task 11 A100 pod** | `scripts/*a100*.py` (to be added by Task 11) |
| **8B-token run, ablation pilots, quality report** | ❌ **not started** | Tasks 12–13 |

No headline performance number exists yet — any future wall-clock, memory or
quality claim will be measured in this repo or explicitly tagged `[INFERENCE]`.

---

## 🔬 Verification

```bash
# Full suite (CPU-friendly, ~40 s)
python3 -m pytest -q
# 111 passed, 14 warnings (torch.jit deprecation on Python 3.14) in 39.15 s

# GPU-gated tests auto-skip on CPU hosts and are never deleted
python3 -m pytest -m gpu -v          # skipped on this macOS host

# Ledger reconciliation (digit-for-digit vs the instantiated meta-device model)
python3 scripts/parameter_budget.py --config configs/pretrain_a100.yaml

# Docs that cite code symbols are anchor-verified; diagrams have visual checks
ls docs/diagrams/*.visual-check.html
```

---

## 📚 Documentation

| doc | contents |
|---|---|
| [`docs/README.md`](docs/README.md) | **Nav map** — corpus table (measured, dated), learning paths, per-track tables, file→doc map |
| [`docs/concepts/`](docs/concepts/) | Theory from first principles: PLE, local/global attention, proportional partial RoPE, cross-layer KV sharing |
| [`docs/references/`](docs/references/) | Code-keyed walkthroughs: model package, cache & generation, training stack, data adapter |
| [`docs/guides/`](docs/guides/) | learning-paths, quickstart, troubleshooting, glossary |
| [`docs/training.md`](docs/training.md) | The applied training pipeline (loop, memory stack, persistence) |
| [`docs/AUDIT.md`](docs/AUDIT.md) | Dated audit: verification runs, findings, from-scratch explanation, modification plan |
| [`docs/sources.md`](docs/sources.md) | **Pinned source manifest** — upstream revisions, SHA-256 verification, oracle environment record |
| [`docs/diagrams/`](docs/diagrams/) | System, decoder block, training-loop and dataflow diagrams (standalone HTML + dark/light PNGs) |
| [`AGENTS.md`](AGENTS.md) | Coding-agent contract: source pins, implementation rules, completion-state discipline, doc-gate rules |
| [`SKILLS.md`](SKILLS.md) | Measured developer workflows (tests, ledger, oracle, gates, smoke/resume) |
| [Design specification](https://github.com/atandra2000/CoreProjects/blob/main/llm-research/DESIGN-gemma-4-e2b-lite.md) | The authoritative model contract (§2 sources, §3 configuration, §4 model, §5 recipe) |
| [Execution plan](https://github.com/atandra2000/CoreProjects/blob/main/llm-research/EXECUTION-PLAN-gemma-4-e2b-lite.md) | Task order and acceptance gates (Tasks 1–13 across five phases) |

> The design and execution-plan docs live one level above this repo in the
> `CoreProjects` workspace (`llm-research/`), so they are not part of this
> repository. Use the workspace links in the table above.

### Pinned upstream sources (verified in `docs/sources.md`)

| Artifact | Pin |
|---|---|
| [E2B config.json](https://huggingface.co/google/gemma-4-E2B/blob/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/config.json) | HF revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| [modeling_gemma4.py](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4/modeling_gemma4.py) | Transformers tag `v5.15.1` · SHA-256 `4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3` (recomputed at oracle import) |
| Tokenizer contract | GPT-2 BPE, vocab 50,257, EOS 50,256 — enforced at data-load time, not by a downloaded file |

Never silently substitute a sibling model's block or a newer upstream version;
any source disagreement is resolved before numerical-parity gates.

---

## 📁 Project structure

```
Gemma-4-E2B-Lite/
├── models/
│   ├── config.py              # validated ModelConfig, layer/KV producer maps, analytic ledger
│   ├── ple.py                 # RMSNorm + per-layer embeddings (reciprocal-√ scaling)
│   ├── attention.py           # RoPE (default + proportional partial), causal/local masks,
│   │                          #   eager reference + SDPA backend with producer/consumer sharing
│   ├── transformer.py         # Block (post-norm residual arithmetic + PLE injection + layer
│   │                          #   scalar) and Gemma4LiteModel (tied head, softcap, checkpointing)
│   └── cache.py               # ProducerKVCache: producer-owned KV, per-layer-type masking,
│                              #   chunked prefill, left-padding validity tracking
├── training/
│   ├── losses.py              # softcapped chunked causal CE with its own +1 shift
│   └── pretrain.py            # AdamW, warmup+cosine token schedule, grad accumulation,
│                              #   BF16 autocast plan, atomic resume
├── inference/
│   └── generate.py            # greedy decoding over the producer cache (pinned EOS convention)
├── data/
│   ├── prepare_data.py        # shim over the workspace shared_data pipeline (GPT-2 contract)
│   └── dataset.py             # ShardWindows + build_dataloader over memmapped uint32 shards
├── utils/
│   └── checkpoint.py          # CheckpointManager: atomic generations, RNG state, safetensors
├── configs/
│   ├── pretrain_a100.yaml     # production configuration (§3 / §5 of the design)
│   └── smoke.yaml             # the ONLY synthetic-data configuration
├── scripts/
│   └── parameter_budget.py    # ledger ↔ instantiated-model reconciliation
├── tests/                     # 111 tests: attention, cache, config, data, loss, PLE,
│                              #   oracle reference, backends, resume, training
├── docs/
│   ├── sources.md             # pinned upstream revisions + verification records
│   └── diagrams/              # Archify HTML diagrams + dark/light PNG visual checks
├── AGENTS.md                  # coding-agent contract (pins, rules, states)
├── LICENSE                    # Apache 2.0
├── requirements.txt           # runtime: torch + pyyaml; test-only: pytest + oracle pin
└── pytest.ini                 # markers: gpu / slow / heavy (auto-skip on CPU hosts)
```

---

## ⚠️ Known caveats

- **No hardware measurements yet.** Every memory/throughput figure in this
  README is a conservative *analytic assumption* from design §3, not a
  measurement. Task 11 (A100 pod) must confirm peak VRAM < 72 GiB, tokens/s
  and BF16-vs-FP32 parity before any rental commitment.
- **No trained checkpoint.** The generate snippet above runs untrained
  weights; it demonstrates the cache/generation machinery, not quality.
- **Synthetic data is legal only in `configs/smoke.yaml`** (design §5) —
  production training fails on absent or mismatched shared-pipeline data
  rather than substituting synthetic tokens.
- **The 8B-token budget is a target, not a proven optimum**, and the run
  duration band (30–60 h at 40–50% MFU) is analytic until Task 11 measures it.
- **Upstream claims are upstream's.** The Gemma E2B model-card numbers are not
  Lite measurements and are never reported as such.

---

## 🤝 Contributing

1. Read [`AGENTS.md`](AGENTS.md) first — it pins the source contract, the
   test-only-oracle rule and the completion-state discipline.
2. Raw PyTorch only: no HF Trainer/Lightning in the implementation; no new
   runtime dependency without updating the source contract.
3. Run `python3 -m pytest -q` — the suite must stay green (currently
   **111 passed / 0 skipped**).
4. GPU-gated tests (`gpu` / `slow` / `heavy` markers) auto-skip on CPU hosts —
   never delete them because they skip.
5. Preserve the pinned numerics (attention scale 1.0, partial-RoPE semantics,
   softcap 30, PLE scaling, sharing boundary 10) — parity tests fail by design
   if these drift.

---

## 🔖 References

- **Gemma E2B model card** — Google, [huggingface.co/google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) (2.3B effective / 5.1B with embeddings; multimodal source model)
- **Transformers reference implementation** — [`transformers` v5.15.1, `modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4/modeling_gemma4.py) (pinned by SHA-256)
- **Gemma technical report** — [arXiv:2607.02770](https://arxiv.org/abs/2607.02770)
- **Per-Layer Embeddings (PLE)** — [Transformers Gemma docs](https://huggingface.co/docs/transformers/en/model_doc/gemma4) (token + projected-input signals, reciprocal-√ scaling)
- **RoPE** — Su et al., [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **RMSNorm** — Zhang & Sennrich, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- **GPT-2 BPE tokenizer** — Radford et al., [arXiv:1901.00530](https://arxiv.org/abs/1901.00530) (vocab 50,257; EOS 50,256 pinned)
- Sibling portfolio: [LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite) · [DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) · [GPT-OSS-Lite](https://github.com/atandra2000/GPT-OSS-Lite) · [Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite) · [DiffusionGemma-Lite](https://github.com/atandra2000/DiffusionGemma-Lite)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE). This matches the pinned upstream
`transformers` reference (Apache 2.0) whose numerics this project adapts.
Upstream model weights and model-card claims remain subject to Google's Gemma
terms and are not redistributed here.






