#!/usr/bin/env python3
"""Held-out validation score for Gemma-4-E2B-Lite. Used by tools/kloop/kloop.py.

Prints one line for the harness to read:

    VAL_BPB=<number>

Bits per byte, lower is better. Not raw cross-entropy. Cross-entropy moves
when the vocabulary moves, so a config that shrinks the vocab would look like
progress while doing no work. Bits per byte divides by the byte length of the
text instead, so architectural changes stay comparable.

Also prints `peak_vram_mb=<number>` on stderr. tools/kloop/kloop.py records it
as a soft cost of the change.

This script is the ground truth. tools/kloop freezes it by hash and refuses to
run a trial if it changes. Do not edit it to make a score look better; if the
metric is genuinely wrong, re-freeze deliberately with `kloop.py freeze`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

from models.config import ModelConfig
from models.transformer import Gemma4LiteModel
from training.losses import chunked_causal_ce
from utils.checkpoint import CheckpointManager

IGNORE_ID = -100
LN2 = math.log(2.0)


def byte_count(token_ids: list[int], encoder) -> int:
    """UTF-8 bytes of the text these token ids stand for.

    BPE decode is lossless for the bytes the tokens carry, so this is the true
    denominator for bits per byte.
    """
    return len(encoder.decode(token_ids).encode("utf-8"))


def score(model: Gemma4LiteModel, loader, *, chunk_size: int, device: str,
          encoder) -> float:
    """Bits per byte over the loader. Lower is better."""
    weight = model.embed.weight if model.config.tie_embeddings else model.lm_head.weight
    cap = model.config.logit_softcap
    nats_total, bytes_total = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            ids = ids.to(device)
            labels = ids.clone()
            hidden = model.forward_hidden(ids)
            loss = chunked_causal_ce(hidden, weight, labels, cap, chunk_size=chunk_size)
            valid = int((labels[:, 1:] != IGNORE_ID).sum())
            if valid == 0:
                continue
            nats_total += float(loss.detach()) * valid
            bytes_total += byte_count(labels[:, 1:].reshape(-1).tolist(), encoder)

    if bytes_total == 0:
        raise ValueError("val set produced no bytes to score")
    return nats_total / LN2 / bytes_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out val score (bits per byte)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--val-data", required=True,
                        help="Held-out shard directory. Never the training data path.")
    parser.add_argument("--tokenizer", default="gpt2",
                        help="tiktoken encoding the val ids were built with")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    val_root = Path(args.val_data)
    if not val_root.exists():
        sys.exit(f"[eval_val] val data not found: {val_root}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("data", {}).get("source") == "synthetic":
        sys.exit("[eval_val] refusing synthetic data; the score must come from real tokens")

    try:
        import tiktoken
        encoder = tiktoken.get_encoding(args.tokenizer)
    except Exception as exc:
        sys.exit(f"[eval_val] cannot load tokenizer {args.tokenizer!r}: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = Gemma4LiteModel(ModelConfig.from_dict(cfg["model"])).to(device)
    try:
        meta = CheckpointManager(args.checkpoint_dir).load(model, device=device)
    except FileNotFoundError as exc:
        sys.exit(f"[eval_val] no checkpoint to score: {exc}")
    model.eval()
    print(f"[eval_val] generation {meta.get('generation_dir')} on {device}", file=sys.stderr)

    from data.dataset import build_dataloader
    loader = build_dataloader(val_root, seq_len=cfg["training"]["seq_len"],
                              batch_size=args.batch_size, seed=0)
    loader = (b for _, b in zip(range(args.max_batches), loader))

    val_bpb = score(model, loader,
                    chunk_size=cfg["training"].get("loss_chunk_tokens", 4096),
                    device=device, encoder=encoder)

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"peak_vram_mb={peak_mb:.1f}", file=sys.stderr)

    print(f"VAL_BPB={val_bpb:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
