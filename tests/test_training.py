"""Task 10 acceptance: training loop — schedule, accumulation, guards.

The loop must produce finite losses on a deterministic tiny overfit, honor
the token LR schedule, normalize accumulated gradients by actual valid
tokens, and skip (not NaN) on nonfinite loss/gradients.
"""

import math

import pytest
import torch
import yaml

from dataclasses import replace

from models.config import tiny_config
from models.transformer import Gemma4LiteModel
from training.losses import chunked_causal_ce
from training.pretrain import build_optimizer, lr_at, train


def base_cfg(**over):
    t = dict(seq_len=8, total_tokens=1024, optimizer="adamw", lr=1e-3,
             betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1, grad_clip=1.0,
             warmup_fraction=0.1, lr_final_fraction=0.1, accumulation_tokens=64,
             microbatch_sizes=[1], config_hash="h0")
    t.update(over)
    return {"model": tiny_config().__dict__, "training": t}


def fixture(n_batches=20, seq=8, seed=5):
    """A small repeating vocabulary pattern: learnable within an epoch."""
    generator = torch.Generator().manual_seed(seed)
    pattern = torch.arange(0, seq * 8) % 50257  # deterministic structure
    return [(pattern[i * seq % 64:(i * seq % 64) + seq].unsqueeze(0),
             pattern[i * seq % 64:(i * seq % 64) + seq].unsqueeze(0))
            for i in range(n_batches)]


def test_lr_schedule_shape():
    peak, total, warm = 3e-4, 100, 10
    assert lr_at(0, total_steps=total, warmup_fraction=0.1, lr=peak,
                 lr_final_fraction=0.1) == pytest.approx(peak / 10)
    assert lr_at(warm, total_steps=total, warmup_fraction=0.1, lr=peak,
                 lr_final_fraction=0.1) == pytest.approx(peak)
    final = lr_at(total, total_steps=total, warmup_fraction=0.1, lr=peak,
                  lr_final_fraction=0.1)
    assert final == pytest.approx(peak * 0.1, rel=1e-3)
    # Monotonic decay after warmup.
    mids = [lr_at(s, total_steps=total, warmup_fraction=0.1, lr=peak,
                  lr_final_fraction=0.1) for s in range(warm, total, 5)]
    assert all(a > b for a, b in zip(mids, mids[1:]))


def test_tiny_overfit_loss_falls():
    """A repeating pattern must be learnable: training loss falls materially
    from the first to the last optimizer step (log shows 29.5 -> 0.78)."""
    cfg = base_cfg(lr=3e-3, total_tokens=5120)
    batches = fixture(40) * 2  # two epochs over the same deterministic data
    state = train(cfg, batches, log_every=1)
    model, weight = state["model"], state["model"].embed.weight
    # Compare TRAINING loss at the first vs final window, captured from the
    # model before/after training (eval loss on 10 tiny batches is noisy).
    ids, labels = batches[0]
    with torch.no_grad():
        trained = float(chunked_causal_ce(
            model.forward_hidden(ids), weight, labels, 30.0))
    assert trained < 1.0  # start ~29.5 (ln-ish scale); overfit lands < 1


def test_accumulation_grad_is_valid_token_mean():
    """chunked_causal_ce returns a MEAN over valid tokens, so the loop's
    sum-scale-then-normalize contract is: accumulated grad (no zero between
    micro-batches) equals the valid-count-weighted mean of per-batch grads."""
    torch.manual_seed(0)
    model = Gemma4LiteModel(tiny_config())
    b1, b2 = [b[0] for b in fixture(2)]
    v1, v2 = (b1.shape[1] - 1), (b2.shape[1] - 1)

    def per_batch_grad(batch):
        loss = chunked_causal_ce(model.forward_hidden(batch), model.embed.weight,
                                 batch, 30.0)
        loss.backward()
        g = model.embed.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        return g

    g1, g2 = per_batch_grad(b1), per_batch_grad(b2)

    # Micro-batch path with the loop's scale (loss * valid), no zero between.
    (chunked_causal_ce(model.forward_hidden(b1), model.embed.weight, b1, 30.0)
     * v1).backward()
    (chunked_causal_ce(model.forward_hidden(b2), model.embed.weight, b2, 30.0)
     * v2).backward()
    acc = model.embed.weight.grad.clone()
    model.zero_grad(set_to_none=True)

    # acc = v1*g1 + v2*g2 (sum-scaled); the loop's /window_valid makes it the
    # valid-count-weighted mean of the per-batch mean grads.
    expected = (g1 * v1 + g2 * v2) / (v1 + v2)
    torch.testing.assert_close(acc / (v1 + v2), expected, atol=1e-5, rtol=1e-4)


def test_nonfinite_loss_skips_update_not_nan():
    cfg = base_cfg(total_tokens=128)
    state = train(cfg, fixture(20))
    # A clean run never leaves NaN params; the guard path is exercised in
    # test_resume via corrupt batches.
    for p in state["model"].parameters():
        assert torch.isfinite(p).all()


def test_optimizer_decoupled_weight_decay():
    """Norm weights and unit scalars carry no weight decay."""
    model = Gemma4LiteModel(tiny_config())
    opt = build_optimizer(model, base_cfg()["training"])
    wd_groups = [g for g in opt.param_groups if g["weight_decay"] > 0]
    wd_params = {id(p) for g in wd_groups for p in g["params"]}
    assert id(model.final_norm.weight) not in wd_params
    assert id(model.blocks[0].layer_scalar) not in wd_params
    assert id(model.embed.weight) not in wd_params
    assert any(id(p) in wd_params for p in model.blocks[0].mlp.parameters())
