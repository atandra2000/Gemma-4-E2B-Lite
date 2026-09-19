"""Task 10: CPU pretrain loop — AdamW, token schedule, accumulation, atomic resume.

Synthetic data is only legal in the smoke configuration (design §5); the
loop takes a batch iterator, so tests feed deterministic fixtures and
production feeds data/dataset.py:build_dataloader over shared shards.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_LLM_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.config import ModelConfig
from models.transformer import Gemma4LiteModel
from training.losses import chunked_causal_ce
from utils.checkpoint import CheckpointManager

IGNORE_ID = -100


def lr_at(step: int, *, total_steps: int, warmup_fraction: float,
          lr: float, lr_final_fraction: float) -> float:
    """Linear warmup then cosine to lr_final_fraction * lr (token-schedule
    semantics: step = optimizer steps, each covering accumulation_tokens)."""
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return lr * (step + 1) / warmup  # (step+1): the first step trains
    progress = (step - warmup) / max(1, total_steps - warmup)
    floor = lr * lr_final_fraction
    return floor + (lr - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_optimizer(model: Gemma4LiteModel, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only (norm weights, unit scalars,
    embeddings and PLE table excluded — house convention)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "embed" in name or "table" in name else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def restore_rng(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])


def train(cfg: dict, batches, *, model: Gemma4LiteModel | None = None,
          checkpoint_dir: str | None = None, resume: bool = False,
          log_every: int = 10) -> dict:
    """Run the loop over ``batches`` (iterable of (input_ids, labels)).

    Accumulation normalizes gradients by the ACTUAL valid tokens in the
    window; a nonfinite loss or grad norm skips the optimizer update (never
    a silent NaN step). Checkpoints are atomic generations; ``resume=True``
    restores model/optimizer/RNG/data position from the latest complete one.
    """
    t = cfg["training"]
    if model is None:
        # Deterministic init: identical configs must build identical models,
        # otherwise interrupted/resumed runs diverge from the first step.
        torch.manual_seed(int(t.get("seed", 42)))
        model = Gemma4LiteModel(ModelConfig.from_dict(cfg["model"]))
    optimizer = build_optimizer(model, t)
    manager = CheckpointManager(checkpoint_dir) if checkpoint_dir else None

    step, tokens_seen, batches_consumed = 0, 0, 0
    if resume and manager is not None:
        meta = manager.load(model, optimizer, device="cpu",
                            expect_config_hash=t.get("config_hash"))
        restore_rng(meta["rng_states"])
        step, tokens_seen, batches_consumed = (
            meta["step"], meta["tokens_seen"], meta["batches_consumed"])
        print(f"[train] resumed at step {step} ({tokens_seen:,} tokens, "
              f"{batches_consumed} batches)")

    accum_tokens = t["accumulation_tokens"]
    total_tokens = t["total_tokens"]
    total_steps = math.ceil(total_tokens / accum_tokens)
    loss_chunk = t.get("loss_chunk_tokens", 4096)
    weight = (model.embed.weight if model.config.tie_embeddings
              else model.lm_head.weight)

    batch_list = batches if isinstance(batches, list) else None

    def finalize_window(window_valid: int, window_loss_sum: float) -> None:
        nonlocal step
        if window_valid == 0:
            return
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None)
        if not math.isfinite(window_loss_sum) or not grads_finite:
            print(f"[train] step {step}: nonfinite loss/grad — update skipped, "
                  f"gradients dropped")
            optimizer.zero_grad(set_to_none=True)
            return
        lr = lr_at(step, total_steps=total_steps,
                   warmup_fraction=t["warmup_fraction"], lr=t["lr"],
                   lr_final_fraction=t["lr_final_fraction"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= window_valid  # valid-token normalization
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        if not math.isfinite(float(grad_norm)):
            print(f"[train] step {step}: nonfinite grad norm — update skipped")
            optimizer.zero_grad(set_to_none=True)
            return
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_valid, window_loss_sum = 0, 0.0
    started = time.time()

    it = iter(batch_list if batch_list is not None else batches)
    for index in range(batches_consumed):
        next(it)  # resume: replay-skip already-consumed batches
    while tokens_seen < total_tokens:
        try:
            ids, labels = next(it)
        except StopIteration:
            break
        batches_consumed += 1
        loss = chunked_causal_ce(model.forward_hidden(ids), weight, labels,
                                 model.config.logit_softcap, chunk_size=loss_chunk)
        valid = int((labels[:, 1:] != IGNORE_ID).sum())
        (loss * valid).backward()  # sum-scaled; normalized at update time
        window_loss_sum += float(loss.detach()) * valid
        window_valid += valid
        tokens_seen += ids.numel()
        if window_valid >= accum_tokens:
            finalize_window(window_valid, window_loss_sum)
            logged_loss = window_loss_sum / window_valid
            window_valid, window_loss_sum = 0, 0.0
            if manager is not None:
                manager.save(model, optimizer, step=step, tokens_seen=tokens_seen,
                             rng_states=capture_rng(),
                             extra_meta={"config_hash": t.get("config_hash"),
                                         "batches_consumed": batches_consumed})
            if log_every and step % log_every == 0:
                rate = tokens_seen / max(1e-9, time.time() - started)
                print(f"[train] step {step} loss {logged_loss:.4f} "
                      f"tokens {tokens_seen:,} rate {rate:,.0f} tok/s")

    finalize_window(window_valid, window_loss_sum)
    if manager is not None:
        manager.save(model, optimizer, step=step, tokens_seen=tokens_seen,
                     rng_states=capture_rng(),
                     extra_meta={"config_hash": t.get("config_hash"),
                                 "batches_consumed": batches_consumed})
    return {"model": model, "optimizer": optimizer, "step": step,
            "tokens_seen": tokens_seen, "manager": manager}


def smoke_batches(vocab: int, eos: int, seq: int, n: int, seed: int = 0):
    """Deterministic synthetic fixture — ONLY for the smoke configuration."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ids = torch.randint(0, vocab, (1, seq), generator=generator)
        yield ids, ids.clone()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma-4-E2B-Lite pretrain (CPU)")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, t = cfg["data"], cfg["training"]
    if d.get("source") == "synthetic":
        model_cfg = ModelConfig.from_dict(cfg["model"])
        n = math.ceil(t["total_tokens"] / (t["seq_len"] * t["microbatch_sizes"][0])) + 1
        batches = list(smoke_batches(model_cfg.vocab_size, model_cfg.eos_id,
                                     t["seq_len"], n, seed=d.get("seed", 0)))
    else:
        from data.dataset import build_dataloader
        root = Path(d.get("data_root", "data/pretrain_corpus"))
        loader = build_dataloader(root, seq_len=t["seq_len"],
                                  batch_size=t["microbatch_sizes"][0],
                                  seed=d.get("seed", 0))
        batches = ((b, b.clone()) for b in loader)  # labels = inputs (LM objective)
    train(cfg, batches, checkpoint_dir=args.checkpoint_dir or "checkpoints/smoke",
          resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
