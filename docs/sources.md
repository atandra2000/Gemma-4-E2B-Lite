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

## Oracle environment (recorded at Task 5, 2026-09-19)

The numerical-parity gate (`tests/test_reference.py`) ran against the pinned
upstream source in this environment — not skipped:

- `transformers` **5.15.1** installed from PyPI into the user site-packages
  (`~/Library/Python/3.14/lib/python/site-packages`), shadowing an older
  5.10.2 system copy. Test-only dependency; runtime code imports no
  transformers (checked: `sys.modules` stays clean after importing `models`).
- `safetensors` 0.8.0 (5.15.1's floor) installed alongside.
- `modeling_gemma4.py` SHA-256 **recomputed at import time and verified
  against the pin** (`4f874549…0cc3`): exact match. The 5.10.2→5.15.1 diff in
  the text path is registration style and vision/audio plumbing only; the
  norm, decoder-layer, PLE, softcap and eager-attention semantics used here
  are unchanged.
- Parity result: tiny-config logits, PLE signals, per-layer hidden states,
  loss and every mapped parameter gradient agree at CPU FP32
  atol=1e-5/rtol=1e-4, including a forward crossing the local-window edge
  (T=33, window 32).
