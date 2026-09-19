"""Task 5 acceptance: full forward and upstream oracle parity.

Identical tiny weights are loaded into the local model and the pinned upstream
`Gemma4ForCausalLM` (transformers v5.15.1, SHA-verified per docs/sources.md)
through an explicit every-parameter map, then logits, intermediate states,
loss and gradients are compared. The instantiated model is also reconciled
against the analytic ledger on the meta device.

Tolerance follows design §6: CPU FP32 atol=1e-5 / rtol=1e-4.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from models.config import ModelConfig, tiny_config
from models.transformer import Gemma4LiteModel

transformers = pytest.importorskip("transformers", reason="oracle extra (test-only)")

TORLERANCE = dict(atol=1e-5, rtol=1e-4)


def assert_close(ours, theirs, label):
    assert ours.shape == theirs.shape, f"{label}: {ours.shape} vs {theirs.shape}"
    assert torch.allclose(ours, theirs, **TORLERANCE), (
        f"{label} diverges (max abs diff {(ours - theirs).abs().max():.3e})"
    )


# -- the explicit weight map -------------------------------------------------

BASE_MAP = {
    "embed.weight": ["model.embed_tokens.weight", "lm_head.weight"],  # tied head
    "ple.table.weight": ["model.embed_tokens_per_layer.weight"],
    "ple.input_proj.weight": ["model.per_layer_model_projection.weight"],
    "ple.projection_norm.weight": ["model.per_layer_projection_norm.weight"],
    "final_norm.weight": ["model.norm.weight"],
}

PER_LAYER_MAP = {
    "input_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "pre_ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
    "post_ple_norm.weight": "post_per_layer_input_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    "ple_gate.weight": "per_layer_input_gate.weight",
    "ple_proj.weight": "per_layer_projection.weight",
    "layer_scalar": "layer_scalar",  # unit buffer, persisted upstream
    "attention.q_proj.weight": "self_attn.q_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.o_proj.weight": "self_attn.o_proj.weight",
}

PRODUCER_EXTRA_MAP = {  # consumers have none of these (design §4)
    "attention.k_proj.weight": "self_attn.k_proj.weight",
    "attention.v_proj.weight": "self_attn.v_proj.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
}


def parameter_map(config) -> dict[str, list[str]]:
    """Every local parameter/buffer name -> every upstream state_dict key."""
    mapping = {name: list(targets) for name, targets in BASE_MAP.items()}
    for i in range(config.n_layers):
        targets = dict(PER_LAYER_MAP)
        if i < config.share_boundary:
            targets |= PRODUCER_EXTRA_MAP
        for local, upstream in targets.items():
            mapping[f"blocks.{i}.{local}"] = [f"model.layers.{i}.{upstream}"]
    return mapping


# -- oracle fixtures ---------------------------------------------------------

def upstream_config(cfg: ModelConfig):
    """Our ModelConfig -> pinned upstream Gemma4TextConfig."""
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=cfg.vocab_size,
        vocab_size_per_layer_input=cfg.vocab_size,  # same vocabulary for the PLE table
        hidden_size=cfg.hidden_dim,
        intermediate_size=cfg.mlp_hidden_dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=1,
        head_dim=cfg.local_head_dim,
        global_head_dim=cfg.global_head_dim,
        hidden_size_per_layer_input=cfg.ple_dim,
        num_kv_shared_layers=cfg.n_layers - cfg.share_boundary,
        layer_types=[
            "full_attention" if i in cfg.global_layers else "sliding_attention"
            for i in range(cfg.n_layers)
        ],
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=4096,
        rms_norm_eps=cfg.rms_eps,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=cfg.local_window,
        final_logit_softcapping=cfg.logit_softcap,
        use_double_wide_mlp=True,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": cfg.local_rope_theta},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": cfg.global_rope_partial_factor,
                "rope_theta": cfg.global_rope_theta,
            },
        },
        pad_token_id=None,  # no padding_idx: row 0 must train like any other
        attn_implementation="eager",
    )


def paired_models(config: ModelConfig):
    """(local model, upstream model) with identical weights, mapped explicitly."""
    from transformers import Gemma4ForCausalLM

    torch.manual_seed(42)
    ours = Gemma4LiteModel(config)
    theirs = Gemma4ForCausalLM(upstream_config(config))

    mapping = parameter_map(config)
    local_names = {name for name, _ in ours.named_parameters()}
    local_names |= {f"blocks.{i}.layer_scalar" for i in range(config.n_layers)}
    assert set(mapping) == local_names, (
        f"map does not cover local parameters: {set(mapping) ^ local_names}"
    )
    named = dict(ours.named_parameters())
    scalars = dict(ours.named_buffers())
    state = {}
    for source, targets in mapping.items():
        tensor = scalars[source] if source.endswith("layer_scalar") else named[source]
        for target in targets:
            state[target] = tensor
    theirs.load_state_dict(state, strict=True)
    assert theirs.config._attn_implementation == "eager"
    return ours, theirs


def sample_ids(config, batch=2, seq=9, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, config.vocab_size, (batch, seq), generator=generator)


def ours_loss(logits, labels):
    shifted = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
    )
    return shifted


# -- parity tests ------------------------------------------------------------

def test_parameter_map_covers_upstream_exactly():
    config = tiny_config()
    from transformers import Gemma4ForCausalLM

    theirs_keys = set(Gemma4ForCausalLM(upstream_config(config)).state_dict())
    mapped = {target for targets in parameter_map(config).values() for target in targets}
    assert mapped == theirs_keys, f"unmapped upstream keys: {theirs_keys - mapped}"


def test_scaled_embedding_parity():
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config)

    their_embeds = theirs.model.embed_tokens(ids)
    assert_close(ours.embed(ids) * ours.embed_scale, their_embeds, "scaled embeddings")


def test_ple_signal_parity():
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config)

    their_embeds = theirs.model.embed_tokens(ids)
    identity = theirs.model.get_per_layer_inputs(ids, their_embeds)
    their_signal = theirs.model.project_per_layer_inputs(their_embeds, identity)
    assert_close(ours.ple(ids, ours.embed(ids) * ours.embed_scale), their_signal, "PLE signal")


def test_per_layer_hidden_parity():
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config)

    captured: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(lambda m, inp, out, idx=i: captured.__setitem__(idx, out))
        for i, layer in enumerate(theirs.model.layers)
    ]
    try:
        theirs.model(input_ids=ids)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == config.n_layers

    hidden = ours.embed(ids) * ours.embed_scale
    signal = ours.ple(ids, hidden)
    shared_kv_states: dict = {}
    for i, block in enumerate(ours.blocks):
        hidden = block(hidden, signal[:, :, i, :], None, shared_kv_states)
        assert_close(hidden, captured[i], f"layer {i} output")


def test_final_hidden_and_logits_parity():
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config)

    their_out = theirs.model(input_ids=ids).last_hidden_state
    assert_close(ours.forward_hidden(ids), their_out, "final hidden states")

    their_logits = theirs(input_ids=ids).logits
    assert_close(ours(ids), their_logits, "softcapped logits")
    assert ours(ids).abs().max() <= config.logit_softcap + 1e-4


def test_loss_and_gradient_parity():
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config)
    labels = sample_ids(config, seed=2)

    their_loss = theirs(input_ids=ids, labels=labels).loss
    our_hidden = ours.forward_hidden(ids)
    weight = ours.embed.weight if config.tie_embeddings else ours.lm_head.weight
    our_logits = config.logit_softcap * torch.tanh(F.linear(our_hidden, weight) / config.logit_softcap)
    our_loss = ours_loss(our_logits, labels)
    assert_close(our_loss.detach(), their_loss.detach(), "loss")

    our_loss.backward()
    their_loss.backward()

    mapping = parameter_map(config)
    # remove_duplicate=False: the tied lm_head alias is the embed storage itself,
    # which is exactly where its gradient accumulates.
    their_named = dict(theirs.named_parameters(remove_duplicate=False))
    compared = 0
    for source, targets in mapping.items():
        if source.endswith("layer_scalar"):
            continue  # buffer
        our_param = dict(ours.named_parameters())[source]
        for target in targets:
            their_param = their_named[target]
            assert our_param.grad is not None and their_param.grad is not None, target
            assert_close(our_param.grad, their_param.grad, f"grad of {source} ({target})")
            compared += 1
    param_entries = len(mapping) - config.n_layers + 1  # scalars are buffers; embed maps to 2 targets
    assert compared == param_entries, "gradient comparison did not traverse the full map"


def test_local_window_boundary_parity():
    """T=33 crosses the 32-token window: absolute-position mask semantics must
    agree with upstream for queries past the window edge."""
    config = tiny_config()
    ours, theirs = paired_models(config)
    ids = sample_ids(config, batch=1, seq=33, seed=3)
    assert_close(ours(ids), theirs(input_ids=ids).logits, "logits across the window edge")


# -- instantiated ledger (Task 5 gate) ----------------------------------------

def load_budget_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "parameter_budget.py"
    spec = importlib.util.spec_from_file_location("parameter_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instantiated_ledger_reconciles():
    """Exact meta-device instantiation matches the analytic ledger, including
    tying, absent consumer K/V and the norm/buffer split."""
    budget = load_budget_script()
    budget.reconcile_with_ledger(tiny_config())
    budget.reconcile_with_ledger(ModelConfig())  # production config


def test_production_exact_total():
    budget = load_budget_script()
    counts = budget.reconcile_with_ledger(ModelConfig())
    from models.config import count_large_matrices, count_norms_and_buffers

    analytic = count_large_matrices(ModelConfig())["subtotal_excluding_norms"] + sum(
        v for k, v in count_norms_and_buffers(ModelConfig()).items() if "buffers" not in k
    )
    assert sum(counts.values()) == analytic == 348_965_184
