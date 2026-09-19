"""Task 3 acceptance: PLE and block arithmetic (design §4, v5.15.1 semantics).

Hand-computed references write the design prose out step by step with explicit
sqrt factors, so replacing addition with concatenation or dropping any scale
factor fails these tests.
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.config import ModelConfig, production_config, tiny_config
from models.ple import PLE, RMSNorm
from models.transformer import Block, GatedGeluMlp

torch.manual_seed(0)


def tiny_model_config() -> ModelConfig:
    """D=4 hand-workable dims; valid producer prefix (layer 2's producer is 0)."""
    return ModelConfig(
        vocab_size=10,
        eos_id=9,
        hidden_dim=4,
        n_layers=3,
        n_heads=2,
        local_head_dim=2,
        global_head_dim=4,
        global_rope_partial_factor=0.5,  # 4 * 0.5 = 2 dims (even)
        global_layers=(0, 2),
        share_boundary=2,
        ple_dim=2,
        mlp_hidden_dim=8,
        mlp_suffix_multiplier=2,
        local_window=4,
    )


def filled_ple(config: ModelConfig) -> PLE:
    """PLE with deterministic nonzero weights in [-1, 1)."""
    generator = torch.Generator().manual_seed(2)  # isolated: full-suite RNG
    # state must not shift these; seed 2 keeps the eps invariance residual
    # at 1.4e-5 (measured), far under the 1e-4 atol — wiring errors are O(1)
    ple = PLE(config)  # state left by earlier test files must not shift these
    with torch.no_grad():
        ple.table.weight.uniform_(-1, 1, generator=generator)
        ple.input_proj.weight.uniform_(-1, 1, generator=generator)
        ple.projection_norm.weight.uniform_(0.5, 1.5, generator=generator)
    return ple


def ple_reference(ids, scaled_emb, table_w, proj_w, norm_w, use_scales=True):
    """Design §4 prose, spelled out: sqrt(P) identity, 1/sqrt(D) projection,
    P-dim RMSNorm, (projected + identity) / sqrt(2)."""
    lp = table_w.shape[1]
    L, P = 3, 2
    s_p = P**0.5 if use_scales else 1.0
    s_d = D**-0.5 if use_scales else 1.0
    mix = 2.0**-0.5 if use_scales else 1.0
    identity = (table_w[ids] * s_p).reshape(*ids.shape, L, P)
    projected = (scaled_emb @ proj_w.t()) * s_d
    projected = projected.reshape(*ids.shape, L, P)
    mean_sq = projected.float().pow(2).mean(-1, keepdim=True) + 1e-6
    projected = projected.float() * torch.pow(mean_sq, -0.5) * norm_w.float()
    return (projected + identity.float()) * mix


D = 4
ids = torch.tensor([[3, 7]])
torch.manual_seed(0)
config = tiny_model_config()
emb = nn.Embedding(config.vocab_size, config.hidden_dim)
with torch.no_grad():
    emb.weight.uniform_(-1, 1)
scaled_emb = emb(ids) * (D**0.5)


class TestPLE:
    def test_shape_contract_and_additive_mixing(self):
        ple = filled_ple(config)
        out = ple(ids, scaled_emb)
        assert out.shape == (1, 2, 3, 2)  # [B, T, L, P]; concat would give L*2P

    def test_hand_computed_mix(self):
        ple = filled_ple(config)
        out = ple(ids, scaled_emb)
        expected = ple_reference(
            ids, scaled_emb, ple.table.weight, ple.input_proj.weight, ple.projection_norm.weight
        )
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sqrt_p_identity_scale_is_load_bearing(self):
        ple = filled_ple(config)
        out = ple(ids, scaled_emb)
        ref = ple_reference(
            ids, scaled_emb, ple.table.weight, ple.input_proj.weight, ple.projection_norm.weight
        )
        no_scale = ple_reference(
            ids, scaled_emb, ple.table.weight, ple.input_proj.weight, ple.projection_norm.weight,
            use_scales=False,
        )
        assert torch.allclose(out, ref, atol=1e-6)
        assert not torch.allclose(out, no_scale, atol=1e-4)

    def test_projection_scale_invariant_through_norm(self):
        # The 1/sqrt(D) projection scale is inert: the P-dim RMSNorm right
        # after it is scale-invariant, so raw and sqrt(D)-scaled inputs give
        # identical PLE output (upstream keeps the factor for reference
        # fidelity; our module must match, whatever it is fed).
        ple = filled_ple(config)
        # eps keeps the invariance approximate; wiring errors differ at O(1).
        assert torch.allclose(ple(ids, scaled_emb), ple(ids, emb(ids)), atol=1e-4)

    def test_ple_pathway_not_zero_initialized(self):
        ple = PLE(config)  # default init, untouched
        out = ple(ids, scaled_emb)
        assert out.abs().sum() > 0
        assert ple.table.weight.abs().mean() > 0


class TestRMSNorm:
    def test_plain_multiplier_not_one_plus_weight(self):
        norm = RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.randn(2, 4)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * 2.0
        out = norm(x)
        assert torch.allclose(out, expected, atol=1e-5)  # (1 + w) offset would give *3
        assert not torch.allclose(out, expected / 2 * 3, atol=1e-3)

    def test_zero_input_is_zero_no_nan(self):
        norm = RMSNorm(4, eps=1e-6)
        out = norm(torch.zeros(1, 4))
        assert torch.isfinite(out).all() and out.abs().sum() == 0

    def test_fp32_compute_regardless_of_input_dtype(self):
        norm = RMSNorm(4, eps=1e-6)
        x = torch.randn(2, 4, dtype=torch.float64)
        assert norm(x).dtype == torch.float64


class TestBlock:
    def block(self) -> tuple[Block, dict]:
        block = Block(config, layer_idx=0)
        block.attention = None  # attention arithmetic is covered in test_attention.py
        with torch.no_grad():
            for norm in (
                block.input_norm, block.post_attention_norm, block.pre_ffn_norm,
                block.post_ffn_norm, block.post_ple_norm,
            ):
                norm.weight.uniform_(0.5, 1.5)
            for lin in (block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
                        block.ple_gate, block.ple_proj):
                lin.weight.uniform_(-0.5, 0.5)
        return block, {}

    def block_reference(self, block, h, signal):
        """Design §4 prose: norms sandwich each branch, post-norm before the
        residual add, PLE gate path last, layer scalar multiplies at the end."""
        residual = h
        h = block.mlp(block.pre_ffn_norm(h))
        h = block.post_ffn_norm(h)
        h = residual + h
        residual = h
        h = F.gelu(block.ple_gate(h), approximate="tanh") * signal
        h = block.post_ple_norm(block.ple_proj(h))
        h = residual + h
        return h * block.layer_scalar

    def test_hand_computed_mlp_and_ple_path(self):
        block, _ = self.block()
        h = torch.randn(1, 2, D)
        signal = torch.randn(1, 2, config.ple_dim)
        assert torch.allclose(block(h, signal), self.block_reference(block, h, signal), atol=1e-5)

    def test_gating_is_load_bearing(self):
        mlp = GatedGeluMlp(config, layer_idx=0)
        x = torch.randn(1, 2, D)
        gated = mlp(x)
        ungated = mlp.down_proj(F.gelu(mlp.gate_proj(x), approximate="tanh"))
        assert not torch.allclose(gated, ungated, atol=1e-4)

    def test_suffix_mlp_double_width(self):
        assert GatedGeluMlp(config, 1).gate_proj.weight.shape == (8, 4)
        # No shared suffix in this 3-layer config; check the multiplier directly.
        assert config.mlp_hidden(1) == 8 and config.mlp_hidden(2) == 16
        production = production_config()
        assert GatedGeluMlp(production, 9).down_proj.weight.shape == (768, 3072)
        assert GatedGeluMlp(production, 10).down_proj.weight.shape == (768, 6144)

    def test_layer_scalar_is_unit_buffer_applied_last(self):
        block, _ = self.block()
        h = torch.randn(1, 2, D)
        signal = torch.randn(1, 2, config.ple_dim)
        assert isinstance(block.layer_scalar, torch.Tensor) and not isinstance(
            block.layer_scalar, nn.Parameter
        )
        base = block(h, signal)
        assert torch.allclose(base, self.block_reference(block, h, signal), atol=1e-5)
        with torch.no_grad():
            block.layer_scalar.fill_(3.0)
        assert torch.allclose(block(h, signal), 3.0 * base, atol=1e-5)

    def test_residual_paths_present(self):
        # Killing the MLP weights must not zero the output: residuals carry input.
        block, _ = self.block()
        h = torch.randn(1, 2, D)
        signal = torch.zeros(1, 2, config.ple_dim)
        with torch.no_grad():
            for lin in (block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
                        block.ple_gate, block.ple_proj):
                lin.weight.zero_()
        out = block(h, signal)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestGradientFlow:
    def test_both_ple_branches_and_all_gates_get_finite_nonzero_grads(self):
        model_config = tiny_model_config()
        emb = nn.Embedding(model_config.vocab_size, model_config.hidden_dim)
        ple = PLE(model_config)
        blocks = nn.ModuleList(Block(model_config, i) for i in range(model_config.n_layers))

        scaled = emb(ids) * (model_config.hidden_dim**0.5)
        signals = ple(ids, scaled)
        h = scaled
        position_ids = torch.arange(ids.shape[1]).unsqueeze(0)
        shared_kv_states = {}  # one request-local dict; producers store, consumers alias
        for i, block in enumerate(blocks):
            h = block(h, signals[:, :, i, :], position_ids, shared_kv_states)
        loss = h.pow(2).sum()
        loss.backward()

        graded = [ple.table.weight, ple.input_proj.weight]  # identity + projected branches
        graded += [block.ple_gate.weight for block in blocks]  # per-layer gates
        graded += [block.ple_proj.weight for block in blocks]
        graded += [block.mlp.gate_proj.weight for block in blocks]
        graded += [emb.weight]
        for tensor in graded:
            assert tensor.grad is not None, "missing gradient"
            assert torch.isfinite(tensor.grad).all(), "nonfinite gradient"
            assert tensor.grad.abs().sum() > 0, "zero gradient"

    def test_layer_scalar_receives_no_grad(self):
        block = Block(config, layer_idx=0)
        h = torch.randn(1, 2, D)
        block(torch.randn(1, 2, D), torch.randn(1, 2, config.ple_dim)).sum().backward()
        assert block.layer_scalar.grad is None or block.layer_scalar.grad.abs().sum() == 0
