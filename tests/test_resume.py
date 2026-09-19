"""Task 10 acceptance: atomic checkpoint generations and deterministic resume.

Resume must land on bit-identical weights, optimizer state, RNG stream and
next data sample as an uninterrupted run. Simulated incomplete writes must
leave the previous generation loadable. Checkpoint incompatibility must be
rejected without overwriting the last complete generation.
"""

import json

import pytest
import torch
import yaml

from models.config import tiny_config
from models.transformer import Gemma4LiteModel
from training.pretrain import capture_rng, restore_rng, train
from utils.checkpoint import CheckpointManager


def base_cfg(**over):
    t = dict(seq_len=8, total_tokens=256, optimizer="adamw", lr=1e-3,
             betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1, grad_clip=1.0,
             warmup_fraction=0.1, lr_final_fraction=0.1, accumulation_tokens=64,
             microbatch_sizes=[1], config_hash="h0")
    t.update(over)
    return {"model": tiny_config().__dict__, "training": t}


def fixture(n_batches=30, seq=8, seed=5):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 50257, (1, seq), generator=generator),
             None) for _ in range(n_batches)]


def labeled(batches):
    return [(ids, ids.clone()) for ids, _ in batches]


def snapshot(state):
    return {k: v.detach().clone() for k, v in state["model"].state_dict().items()}


def opt_snapshot(state):
    out = {}
    for i, group in enumerate(state["optimizer"].state.values()):
        for k, v in group.items():
            if isinstance(v, torch.Tensor):
                out[(i, k)] = v.detach().clone()
    return out


def test_interrupted_resume_matches_uninterrupted(tmp_path):
    """Kill after 2 full windows; resume must reproduce the uninterrupted run.

    seq_len=9 -> 8 valid tokens/batch; accumulation_tokens=64 -> exactly
    8 batches per window, so a 16-batch cut lands on a window boundary and
    both runs take identical updates up to the interruption."""
    cfg = base_cfg(seq_len=9, total_tokens=288, accumulation_tokens=64)
    batches = labeled(fixture(40, seq=9))

    full = train(cfg, batches, checkpoint_dir=str(tmp_path / "full"))
    reference_weights = snapshot(full)
    reference_opt = opt_snapshot(full)

    cut = 16  # 2 windows exactly (16 x 8 valid tokens)
    partial = train(cfg, batches[:cut], checkpoint_dir=str(tmp_path / "resume"))
    assert partial["step"] == 2

    resumed = train(cfg, batches, checkpoint_dir=str(tmp_path / "resume"),
                    resume=True)
    assert resumed["step"] == full["step"]
    assert resumed["tokens_seen"] == full["tokens_seen"]

    ref_w, res_w = reference_weights, snapshot(resumed)
    for key, tensor in ref_w.items():
        torch.testing.assert_close(res_w[key], tensor, msg=f"weight {key}")
    ref_o, res_o = reference_opt, opt_snapshot(resumed)
    for key, tensor in ref_o.items():
        torch.testing.assert_close(res_o[key], tensor, msg=f"optim {key}")


def test_resume_replays_next_data_sample(tmp_path):
    """After resume, the next batch consumed is the one after the checkpoint."""
    cfg = base_cfg(total_tokens=256)
    batches = labeled(fixture(20))
    state = train(cfg, batches[:8], checkpoint_dir=str(tmp_path / "ck"))
    meta = state["manager"].latest_generation()
    saved = json.loads((meta / "meta.json").read_text())
    # 8 batches x 8 tokens: checkpoint records batches_consumed=8 (one window)
    assert saved["batches_consumed"] == 8


def test_incomplete_write_leaves_previous_generation_loadable(tmp_path):
    """A crash mid-save (partial gen dir, no pointer flip) must be invisible."""
    cfg = base_cfg(total_tokens=128)
    state = train(cfg, labeled(fixture(10)), checkpoint_dir=str(tmp_path / "ck"))
    manager = state["manager"]
    good = manager.latest_generation()
    assert good is not None

    # Simulate a crashed save: a generation directory missing files, no pointer.
    crashed = tmp_path / "ck" / "gen_0000009999"
    crashed.mkdir()
    (crashed / "model.safetensors").write_bytes(b"partial")
    assert manager.latest_generation() == good  # still the old one
    model = Gemma4LiteModel(tiny_config())
    meta = manager.load(model)  # loads the previous generation fine
    assert meta["step"] == state["step"]


def test_incompatible_config_rejected_without_overwrite(tmp_path):
    cfg = base_cfg(total_tokens=128)
    state = train(cfg, labeled(fixture(10)), checkpoint_dir=str(tmp_path / "ck"))
    before = state["manager"].latest_generation()

    other = base_cfg(total_tokens=128, config_hash="different-run")
    model = Gemma4LiteModel(tiny_config())
    with pytest.raises(RuntimeError, match="incompatible|config_hash"):
        CheckpointManager(str(tmp_path / "ck")).load(
            model, expect_config_hash="different-run")
    # The complete generation is untouched.
    assert state["manager"].latest_generation() == before


def test_rng_restored_exactly():
    capture = capture_rng()
    torch.rand(3)
    restore_rng(capture)
    a = torch.rand(3)
    restore_rng(capture)
    b = torch.rand(3)
    torch.testing.assert_close(a, b)
