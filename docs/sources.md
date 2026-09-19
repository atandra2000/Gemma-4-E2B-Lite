# Source manifest

Pinned upstream artifacts for the Gemma-4-E2B-Lite source contract
(design §2). Verified 2026-09-19. Preserve attribution/license notices for
any adapted code.

## Verified pins

| Artifact | Pin | Verification (2026-09-19) |
|---|---|---|
| [E2B config.json](https://huggingface.co/google/gemma-4-E2B/blob/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/config.json) | HF revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | Resolves, HTTP 200 |
| [modeling_gemma4.py](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4/modeling_gemma4.py) | Transformers tag `v5.15.1`, SHA-256 `4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3` | Recomputed from a fresh download of the tag; exact match |

The configuration reference (`Gemma4TextConfig`) and the proportional-RoPE
helpers both live in the pinned `modeling_gemma4.py` file — the SHA-256
above covers them. If a parity gate ever needs a finer pin (specific
helper, its own dependencies), record it here before use; do not pull a
newer upstream version.

## Context artifacts (not code pins)

| Artifact | Role |
|---|---|
| [Model card](https://huggingface.co/google/gemma-4-E2B) | 2.3B effective / 5.1B with embeddings; multimodal source model. Claims are upstream's, not Lite measurements |
| [PLE documentation](https://huggingface.co/docs/transformers/en/model_doc/gemma4) | Token and projected-input signals, reciprocal-square-root scaling |
| [Technical report](https://arxiv.org/abs/2607.02770) | Architecture context |

## Pending pins

- **Tokenizer assets** (GPT-2 BPE, EOS 50,256): hash pinned in the data
  manifest at Task 9, not here.
- **Weights:** none downloaded. Task 1 explicitly excludes training data and
  checkpoint downloads.

## Recorded environment (2026-09-19)

- macOS dev host (arm64), CPU/MPS only — no CUDA, no Triton.
- `uv` 0.12.12, Python 3.14.7, torch 2.12.0 importable system-wide.
- GPU-dependent verification (BF16 parity, A100 fit) is deferred to Task 11.
