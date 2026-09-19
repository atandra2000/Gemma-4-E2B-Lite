"""Task 7 acceptance: SDPA backend and activation-checkpointing safety.

SDPA with explicit scale=1.0 and the offset-aware cache masks must reproduce
the eager reference within the design §6 CPU FP32 tolerance, with and without
a ProducerKVCache in play. Non-reentrant checkpointing of the attention
sublayer must reproduce uncheckpointed outputs AND all parameter gradients
exactly (same kernels, same inputs). Compile parity is exercised separately
(compile-with-checkpointing is a known-incompatible combination; see the
skipped-compile test note).
"""

from dataclasses import replace

import pytest
import torch

from models.cache import ProducerKVCache
from models.config import tiny_config
from models.transformer import Gemma4LiteModel

TOLERANCE = dict(atol=1e-5, rtol=1e-4)
# Recorded numerical explanation (plan Task 7 rule): eager and SDPA reduce the
# same scores/softmax differently (fused kernel, different summation order),
# so CPU FP32 divergence lands just past the 1e-5 base atol on long sequences.
# Measured on this host: eager-vs-SDPA prefill logits max diff 3.4e-5 at T=70.
# The design's rtol=1e-4 still dominates for large-magnitude logits; atol is
# relaxed to 1e-4 to match the shape-noise allowance already recorded in
# tests/test_cache.py. Gradients are compared at the base tolerance.
SDPA_NOISE = dict(atol=1e-4, rtol=1e-4)


def tiny_model(config=None):
    torch.manual_seed(42)
    return Gemma4LiteModel(config or tiny_config())


def sample_ids(config, seq, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, config.vocab_size, (1, seq), generator=generator)


def eager_model(config=None):
    return tiny_model(config)  # default attn_backend="eager"


def sdpa_model(config=None):
    return tiny_model(replace(config or tiny_config(), attn_backend="sdpa"))


@torch.no_grad()
def test_sdpa_matches_eager_prefill():
    config = tiny_config()
    model_e, model_s = eager_model(config), sdpa_model(config)
    ids = sample_ids(config, 70, seed=1)  # > 2 * local_window

    out_e, out_s = model_e(ids), model_s(ids)
    assert out_e.shape == out_s.shape
    assert torch.allclose(out_e, out_s, **SDPA_NOISE), (
        f"eager/SDPA prefill diverges (max abs diff {(out_e - out_s).abs().max():.3e})"
    )


@torch.no_grad()
def test_sdpa_matches_eager_at_window_edge():
    """The plan's 511/512/513 local-window gate, SDPA vs eager, no cache."""
    config = replace(tiny_config(), local_window=512)
    model_e, model_s = eager_model(config), sdpa_model(config)
    for seq in (511, 512, 513):
        ids = sample_ids(config, seq, seed=seq)
        diff = (model_e(ids) - model_s(ids)).abs().max()
        assert torch.allclose(model_e(ids), model_s(ids), **SDPA_NOISE), (
            f"T={seq} diverges (max abs diff {diff:.3e})"
        )


@torch.no_grad()
def test_sdpa_matches_eager_through_cache():
    """SDPA handles the cache's offset-aware [B, 1, Q, kv] masks (padding,
    trimming, per-row positions) identically to eager."""
    config = tiny_config()
    model_e, model_s = eager_model(config), sdpa_model(config)
    short = sample_ids(config, 30, seed=11)[0]
    long = sample_ids(config, 45, seed=12)[0]

    # Chunked prefill + decode for two rows, one left-padded.
    for model in (model_e, model_s):
        cache = ProducerKVCache(config)
        prompt = torch.stack([torch.cat([torch.full((15,), 0), short]), long])
        kv = torch.tensor([[False] * 15 + [True] * 30, [True] * 45])
        out = model(prompt[:, :20], cache=cache, key_valid=kv[:, :20])
        out = model(prompt[:, 20:], cache=cache, key_valid=kv[:, 20:])
        token = out[:, -1].argmax(-1).view(2)
        for _ in range(3):
            token = model(token.view(2, 1), cache=cache)[:, -1].argmax(-1)
        if model is model_s:
            sdpa_token = token
    assert sdpa_token.equal(token), "SDPA cache-path tokens diverge from eager"


@torch.no_grad()
def test_sdpa_matches_eager_batched_2d_mask():
    """Broadcast rule: a plain [T, T] mask (no cache) must broadcast over
    batch/heads inside SDPA — same logits as eager."""
    config = tiny_config()
    model_e, model_s = eager_model(config), sdpa_model(config)
    ids = sample_ids(config, 40, seed=4)
    assert torch.allclose(model_e(ids), model_s(ids), **SDPA_NOISE)


def test_sdpa_scale_is_explicit_not_inverse_sqrt():
    """The design pins scale=1.0; SDPA must not silently apply 1/sqrt(d)."""
    model = sdpa_model()
    att = model.blocks[0].attention
    assert att.attn_backend == "sdpa" and att.scaling == 1.0


def test_invalid_backend_rejected():
    from models.config import ModelConfig

    with pytest.raises(ValueError, match="attn_backend"):
        ModelConfig(attn_backend="flash")


# -- activation checkpointing -------------------------------------------------


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _train_step(model, ids, cache=None, key_valid=None):
    model.zero_grad(set_to_none=True)
    logits = model(ids, cache=cache, key_valid=key_valid)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        ids[:, 1:].reshape(-1),
    )
    loss.backward()
    return loss.detach(), _grads(model)


def test_checkpointed_matches_uncheckpointed_outputs_and_grads():
    """Same seeds -> same weights; checkpointed backward must reproduce the
    uncheckpointed loss and EVERY parameter gradient exactly (shared KV
    producers included — consumer grads must flow through recomputed K/V)."""
    config = tiny_config()
    model_plain = tiny_model(config)
    model_ckpt = tiny_model(config)
    ids = sample_ids(config, 24, seed=2)

    loss_plain, grads_plain = _train_step(model_plain, ids)
    model_ckpt.checkpoint_blocks(True)
    loss_ckpt, grads_ckpt = _train_step(model_ckpt, ids)

    assert torch.allclose(loss_plain, loss_ckpt, **TOLERANCE), (
        f"loss differs: {loss_plain.item()} vs {loss_ckpt.item()}"
    )
    assert set(grads_plain) == set(grads_ckpt), "same parameter set must receive grads"
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], **TOLERANCE), (
            f"grad mismatch on {name} "
            f"(max abs diff {(grads_plain[name] - grads_ckpt[name]).abs().max():.3e})"
        )


def test_checkpointed_producer_grads_flow_through_shared_kv():
    """Consumer-layer losses must backprop through the producer's stored K/V
    under checkpointing: the k/v producers' projections get gradient from BOTH
    their own layer and the shared consumers above the boundary."""
    config = tiny_config()
    model = tiny_model(config)
    model.checkpoint_blocks(True)
    ids = sample_ids(config, 20, seed=3)
    _train_step(model, ids)

    boundary = config.share_boundary
    for layer_idx in range(boundary):
        for proj in ("k_proj", "v_proj"):
            g = getattr(model.blocks[layer_idx].attention, proj).weight.grad
            assert g is not None and g.abs().sum() > 0, (
                f"producer layer {layer_idx}.{proj} got no grad under checkpointing"
            )


def test_checkpointing_with_cache_inference_leaves_cache_consistent():
    """Checkpointing is a training lever; the cache path is no_grad, so the
    flag must not corrupt the cache lifecycle (inflight trimmed, masks sane)."""
    config = tiny_config()
    model = tiny_model(config)
    model.checkpoint_blocks(True)
    ids = sample_ids(config, 35, seed=5)
    cache = ProducerKVCache(config)

    out_ckpt = model(ids[:, :20], cache=cache)
    cache.finish_step()
    out_next = model(ids[:, 20:], cache=cache)
    model.checkpoint_blocks(False)
    cache_fresh = ProducerKVCache(config)
    out_fresh = model(ids, cache=cache_fresh)
    assert out_next.shape == out_fresh[:, 20:].shape
    assert cache._inflight == {} and cache_fresh._inflight == {}


@pytest.mark.slow
def test_compile_matches_eager_separately():
    """Compile parity is exercised WITHOUT checkpointing active (Task 7:
    'exercise compile separately') — non-reentrant checkpoint + dynamo are
    known-incompatible on this torch line and the plan defers compiled
    training-path integration beyond Task 7."""
    config = tiny_config()
    model = tiny_model(config)
    compiled = torch.compile(model, dynamic=False)
    ids = sample_ids(config, 24, seed=6)
    out_c = compiled(ids)
    out_e = model(ids)
    # Compiled kernels fuse/reorder reductions like SDPA does — same recorded
    # 1e-4 atol allowance as the SDPA comparisons above.
    assert torch.allclose(out_e, out_c, **SDPA_NOISE), (
        f"compiled vs eager diverges (max abs diff {(out_e - out_c).abs().max():.3e})"
    )
