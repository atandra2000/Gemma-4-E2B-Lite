"""Task 2 acceptance: config validation, producer map and the analytic ledger."""

import pytest

from models.config import (
    ModelConfig,
    count_large_matrices,
    count_norms_and_buffers,
    memory_assumptions,
    production_config,
    tiny_config,
)

DESIGN_ROWS = {
    "main_embedding_tied": 38_597_376,  # 50257 * 768, tied head counted once
    "ple_token_table": 64_328_960,  # 50257 * 20 * 64
    "ple_projections": 2_949_120,  # D->L*P input + per-block D->P gate and P->D projection
    "mlp_matrices": 212_336_640,  # 3*D*3072 prefix + 3*D*6144 shared suffix
    "q_and_output_projections": 28_311_552,
    "kv_projections_prefix_only": 2_359_296,
}
DESIGN_SUBTOTAL = 348_882_944


class TestProductionConfig:
    def test_layer_types_match_design(self):
        config = production_config()
        assert config.global_layers == (4, 9, 14, 19)
        assert config.layer_type(19) == "global"  # final layer global
        assert sum(config.layer_type(i) == "global" for i in range(20)) == 4
        assert config.layer_type(0) == "local"

    def test_producer_map_matches_design(self):
        # The acceptance gate: shared locals consume producer 8, shared globals 9.
        config = production_config()
        for i in range(10):
            assert config.kv_producer(i) == i, "prefix layers self-produce"
        for i in (10, 11, 12, 13, 15, 16, 17, 18):
            assert config.kv_producer(i) == 8
        for i in (14, 19):
            assert config.kv_producer(i) == 9

    def test_ledger_rows_match_design(self):
        counts = count_large_matrices(production_config())
        for key, expected in DESIGN_ROWS.items():
            assert counts[key] == expected, key

    def test_large_matrix_subtotal(self):
        counts = count_large_matrices(production_config())
        assert counts["subtotal_excluding_norms"] == DESIGN_SUBTOTAL

    def test_norms_and_buffers_are_distinguished(self):
        norms = count_norms_and_buffers(production_config())
        assert norms["unit_layer_scalar_buffers"] == 20
        # Subtotal deliberately excludes norms; they are reported separately.
        assert all(k not in DESIGN_ROWS for k in norms)
        total = norms["layer_norms"] + norms["ple_norms"] + norms["q_norms"] + norms["kv_norms_producer_only"] + norms["final_norm"]
        assert total > 0

    def test_tying_and_softcap_pinned(self):
        config = production_config()
        assert config.tie_embeddings
        assert config.logit_softcap == 30.0
        assert config.attention_scale == 1.0
        assert config.rms_eps == 1e-6

    def test_global_rotary_subspace_is_64(self):
        config = production_config()
        assert config.global_rotary_dim == 64  # 0.25 * 256
        assert config.local_rope_theta < config.global_rope_theta

    def test_mlp_suffix_doubles(self):
        config = production_config()
        assert config.mlp_hidden(9) == 3072
        assert config.mlp_hidden(10) == 6144
        assert config.mlp_hidden(19) == 6144

    def test_kv_dims_inherit_per_type(self):
        config = production_config()
        assert config.head_dim(0) == 128 and config.head_dim(4) == 256
        assert config.qkv_dim(4) == 1536 and config.qkv_dim(0) == 768


class TestTinyConfig:
    def test_two_local_global_periods_and_valid_prefix(self):
        config = tiny_config()
        assert config.global_layers == (2, 5, 8, 11)
        suffix_types = [config.layer_type(i) for i in config.shared_layers()]
        # two full local->global periods after the producer prefix
        assert "".join(t[0] for t in suffix_types) == "llgllg"
        for i in config.shared_layers():
            producer = config.kv_producer(i)
            assert producer is not None and producer < config.share_boundary
            assert config.layer_type(producer) == config.layer_type(i)

    def test_tiny_config_fits_smoke_yaml(self):
        assert ModelConfig.from_yaml("configs/smoke.yaml") == tiny_config()


class TestValidation:
    def test_no_producer_prefix(self):
        with pytest.raises(ValueError, match="producer prefix"):
            ModelConfig(global_layers=(4, 9, 14, 19), share_boundary=0)

    def test_consumer_without_same_type_producer(self):
        # No global layer in the prefix, but a global consumer exists.
        with pytest.raises(ValueError, match="no global producer"):
            ModelConfig(
                hidden_dim=64,
                n_layers=12,
                n_heads=2,
                local_head_dim=32,
                global_head_dim=64,
                global_layers=(8, 11),
                share_boundary=6,
                mlp_hidden_dim=128,
            )

    def test_head_grouping_mismatch(self):
        with pytest.raises(ValueError, match="head grouping"):
            ModelConfig(n_heads=5, hidden_dim=768, local_head_dim=128)

    def test_global_head_dim_breaks_1_to_2_relation(self):
        with pytest.raises(ValueError, match="1:2"):
            ModelConfig(global_head_dim=192)

    def test_odd_head_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            ModelConfig(local_head_dim=127, n_heads=6, hidden_dim=762, global_head_dim=254)

    def test_global_rotary_subspace_odd_rejected(self):
        with pytest.raises(ValueError, match="rotary subspace"):
            ModelConfig(global_rope_partial_factor=9 / 256)  # -> 9 dims, odd

    def test_global_rotary_subspace_nonintegral_rejected(self):
        with pytest.raises(ValueError, match="rotary subspace"):
            ModelConfig(global_rope_partial_factor=0.05)  # 12.8 dims

    def test_global_rotary_subspace_too_large_rejected(self):
        with pytest.raises(ValueError, match="rotary subspace"):
            ModelConfig(global_rope_partial_factor=1.5)

    def test_invalid_vocab(self):
        with pytest.raises(ValueError, match="vocab_size"):
            ModelConfig(vocab_size=0)

    def test_eos_outside_token_range(self):
        with pytest.raises(ValueError, match="token range"):
            ModelConfig(vocab_size=1000, eos_id=1000)

    def test_final_layer_must_be_global(self):
        with pytest.raises(ValueError, match="final layer must be global"):
            ModelConfig(global_layers=(4, 9, 14, 18))

    def test_kv_producer_out_of_range(self):
        with pytest.raises(IndexError):
            production_config().kv_producer(20)


class TestYamls:
    def test_pretrain_yaml_matches_production(self):
        config = ModelConfig.from_yaml("configs/pretrain_a100.yaml")
        assert config == production_config()

    def test_unknown_model_key_fails_clearly(self):
        with pytest.raises(ValueError, match="unknown model config keys"):
            ModelConfig.from_dict({"n_layer": 20})  # typo -> rejected

    def test_memory_assumptions_match_design(self):
        memory = memory_assumptions(production_config())
        # 16 bytes/param over the subtotal is about 5.2 GiB (design §3).
        assert memory["trainable_bytes_16_per_param_gib"] == pytest.approx(5.2, abs=0.05)
        assert memory["full_logits_b1_t4096_bf16_gib"] == pytest.approx(0.38, abs=0.01)
        assert memory["full_logits_b1_t4096_fp32_gib"] == pytest.approx(0.77, abs=0.01)
