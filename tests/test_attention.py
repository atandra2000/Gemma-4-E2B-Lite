"""Task 4 acceptance: reference attention and shared producers.

Reference math in this file re-derives attention from module weights with
explicit ops (scale 1.0, fp32 softmax, absolute-position masks) so a wrong
scale, mask offset or detached producer fails loudly.
"""

import torch
from torch.nn import functional as F

from models.attention import Attention, causal_mask, rotary_inv_freq
from models.config import ModelConfig
from models.transformer import Block

torch.manual_seed(0)


def windowed_config(**overrides) -> ModelConfig:
    """Tiny dims, production window: local/global boundary behavior at T~512."""
    return ModelConfig(
        vocab_size=10,
        eos_id=9,
        hidden_dim=8,
        n_layers=7,
        n_heads=2,
        local_head_dim=4,
        global_head_dim=8,
        global_layers=(2, 5, 6),
        share_boundary=4,
        ple_dim=2,
        mlp_hidden_dim=16,
        mlp_suffix_multiplier=2,
        local_window=512,
        **overrides,
    )


def make_block(config, layer_idx) -> Block:
    torch.manual_seed(layer_idx)
    return Block(config, layer_idx)


def rope_ops(attention, position_ids):
    """Shared rope helpers for the reference math."""
    freqs = attention.inv_freq[None, None, :] * position_ids.float()[:, :, None]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    hd_half = cos.shape[-1] // 2

    def rope(x):  # x [B, T, *, hd]
        x = x.float()
        rot = torch.cat((-x[..., hd_half:], x[..., :hd_half]), dim=-1)
        return x * cos.unsqueeze(2) + rot * sin.unsqueeze(2)

    def norm(x, weight=None):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x if weight is None else x * weight.float()

    return rope, norm


def attention_reference(attention, hidden, position_ids, mask, scale=1.0):
    """Independent eager-attention math for a producer layer, fp32 throughout."""
    hd, H = attention.head_dim, attention.num_heads
    B, T, _ = hidden.shape
    rope, norm = rope_ops(attention, position_ids)

    q = norm(attention.q_proj(hidden).view(B, T, H, hd), attention.q_norm.weight)
    q = rope(q).transpose(1, 2)
    k = norm(attention.k_proj(hidden).view(B, T, 1, hd), attention.k_norm.weight)
    k = rope(k).transpose(1, 2).expand(B, H, T, hd)
    v = norm(attention.v_proj(hidden).view(B, T, 1, hd))  # no weight: with_scale=False
    v = v.transpose(1, 2).expand(B, H, T, hd)

    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, -1)
    return attention.o_proj(out)


class TestKernels:
    def test_local_attention_matches_reference_at_511_512_513(self):
        config = windowed_config()
        for T in (511, 512, 513):
            attention = make_block(config, 0).attention  # layer 0: local producer
            hidden = torch.randn(1, T, config.hidden_dim)
            position_ids = torch.arange(T).unsqueeze(0)
            mask = causal_mask(T, config.local_window)
            out = attention(hidden, position_ids, {}, mask)
            ref = attention_reference(attention, hidden, position_ids, mask)
            assert out.shape == (1, T, config.hidden_dim)
            assert torch.allclose(out, ref, atol=1e-4), f"T={T}"

    def test_global_attention_matches_reference_at_511_512_513(self):
        config = windowed_config()
        for T in (511, 512, 513):
            attention = make_block(config, 2).attention  # layer 2: global producer
            hidden = torch.randn(1, T, config.hidden_dim)
            position_ids = torch.arange(T).unsqueeze(0)
            mask = causal_mask(T)  # all preceding keys, including self
            out = attention(hidden, position_ids, {}, mask)
            ref = attention_reference(attention, hidden, position_ids, mask)
            assert torch.allclose(out, ref, atol=1e-4), f"T={T}"

    def test_scale_is_explicit_one_not_inverse_sqrt(self):
        config = windowed_config()
        attention = make_block(config, 0).attention
        assert attention.scaling == 1.0
        hidden = torch.randn(1, 8, config.hidden_dim)
        position_ids = torch.arange(8).unsqueeze(0)
        mask = causal_mask(8, config.local_window)
        out = attention(hidden, position_ids, {}, mask)
        ref = attention_reference(attention, hidden, position_ids, mask, scale=1.0)
        wrong = attention_reference(
            attention, hidden, position_ids, mask, scale=attention.head_dim**-0.5
        )
        assert torch.allclose(out, ref, atol=1e-4)
        assert not torch.allclose(out, wrong, atol=1e-5)

    def test_v_norm_has_no_weight(self):
        config = windowed_config()
        attention = make_block(config, 0).attention
        assert attention.v_norm.with_scale is False and attention.v_norm.weight is None
        assert attention.k_norm.weight is not None  # K norm keeps its weight

    def test_proportional_rope_zero_frequency_tail(self):
        # 0.25 * 256 -> 32 rotated pairs (64 dims) + 96 zero-frequency pairs.
        inv_freq = rotary_inv_freq(256, 1_000_000.0, 0.25)
        assert inv_freq.shape == (128,)
        assert (inv_freq[:32] > 0).all() and (inv_freq[32:] == 0).all()
        assert inv_freq[0] == 1.0  # exponent denominator is the FULL head dim
        assert torch.isclose(
            inv_freq[1], torch.tensor(1.0 / (1_000_000.0 ** (2 / 256))), atol=1e-12
        )
        local = rotary_inv_freq(4, 10_000.0, 1.0)  # local default: full dim
        assert (local > 0).all() and local.shape == (2,)


class TestCausalityAndWindows:
    def test_future_tokens_do_not_affect_past(self):
        config = windowed_config()
        T = 6
        position_ids = torch.arange(T).unsqueeze(0)
        for layer_idx in (0, 2):  # local, global
            block = make_block(config, layer_idx)
            base = torch.randn(1, T, config.hidden_dim)
            perturbed = base.clone()
            perturbed[:, 3:] = torch.randn(1, T - 3, config.hidden_dim)
            mask = causal_mask(T, config.local_window)
            args = (torch.zeros(1, T, config.ple_dim), position_ids, {}, mask)
            out_base = block(base, *args)
            out_pert = block(perturbed, *args)
            assert torch.allclose(out_base[:, :3], out_pert[:, :3], atol=1e-5), f"layer {layer_idx}"

    def test_local_window_hides_key_at_exact_distance(self):
        # T=513: query 512 must not see key 0 (distance 512), must see key 1
        # (distance 511). Masks obey absolute positions, window half-open.
        config = windowed_config()
        T = 513
        block = make_block(config, 0)  # local producer layer
        hidden = torch.randn(1, T, config.hidden_dim)
        position_ids = torch.arange(T).unsqueeze(0)
        zeros = torch.zeros(1, T, config.ple_dim)
        mask = causal_mask(T, config.local_window)

        out = block(hidden, zeros, position_ids, {}, mask)
        changed_start = hidden.clone()
        changed_start[:, 0] += 10.0
        out_start = block(changed_start, zeros, position_ids, {}, mask)
        changed_near = hidden.clone()
        changed_near[:, 1] += 10.0
        out_near = block(changed_near, zeros, position_ids, {}, mask)

        assert torch.allclose(out[:, 512], out_start[:, 512], atol=1e-5), "key 0 leaked into query 512"
        assert not torch.allclose(out[:, 512], out_near[:, 512], atol=1e-4), "key 1 invisible to query 512"
        # At T=511 nothing is hidden: query 510 sees key 0 at distance 510.
        assert (causal_mask(511, config.local_window)[510, 0] == 0.0)
        assert (causal_mask(513, config.local_window)[512, 0].isinf())

    def test_global_sees_all_previous_including_self(self):
        config = windowed_config()
        T = 513
        block = make_block(config, 2)  # global producer layer
        hidden = torch.randn(1, T, config.hidden_dim)
        position_ids = torch.arange(T).unsqueeze(0)
        zeros = torch.zeros(1, T, config.ple_dim)
        mask = causal_mask(T)
        out = block(hidden, zeros, position_ids, {}, mask)
        for key in (0, 1, 256, 511):
            perturbed = hidden.clone()
            perturbed[:, key] += 10.0
            out_pert = block(perturbed, zeros, position_ids, {}, mask)
            assert not torch.allclose(out[:, 512], out_pert[:, 512], atol=1e-4), f"key {key} invisible"


class TestSharedProducers:
    def test_consumers_have_no_kv_projections_or_norms(self):
        config = windowed_config()
        assert config.kv_producer(4) == 3 and config.kv_producer(6) == 2
        for layer_idx, shared in ((3, False), (4, True)):
            attention = make_block(config, layer_idx).attention
            assert hasattr(attention, "k_proj") is not shared, f"layer {layer_idx}"
            if shared:
                assert not hasattr(attention, "k_norm") and not hasattr(attention, "v_norm")

    def consumer_reference(self, consumer, producer_kv, hidden, position_ids, mask):
        """Full block math for a consumer: aliased K/V, then MLP/PLE/scalar."""
        hd, H = consumer.attention.head_dim, consumer.attention.num_heads
        B, T, _ = hidden.shape
        rope, norm = rope_ops(consumer.attention, position_ids)
        k, v = producer_kv
        q = norm(consumer.attention.q_proj(hidden).view(B, T, H, hd), consumer.attention.q_norm.weight)
        q = rope(q).transpose(1, 2)
        scores = torch.matmul(q, k.expand(B, H, T, hd).transpose(2, 3))
        weights = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
        o = (weights @ v.expand(B, H, T, hd)).transpose(1, 2).reshape(B, T, -1)
        o = consumer.attention.o_proj(o)

        h = hidden + consumer.post_attention_norm(o)
        residual = h
        h = consumer.mlp(consumer.pre_ffn_norm(h))
        h = residual + consumer.post_ffn_norm(h)
        residual = h
        signal = torch.zeros(B, T, consumer.ple_proj.in_features)
        h = F.gelu(consumer.ple_gate(h), approximate="tanh") * signal
        h = consumer.post_ple_norm(consumer.ple_proj(h))
        return (residual + h) * consumer.layer_scalar

    def test_consumer_aliases_producer_states_exactly(self):
        config = windowed_config()
        producer = make_block(config, 3)  # last prefix local producer
        consumer = make_block(config, 4)  # first shared local consumer
        T = 5
        hidden = torch.randn(1, T, config.hidden_dim)
        position_ids = torch.arange(T).unsqueeze(0)
        mask = causal_mask(T, config.local_window)
        shared = {}
        producer(hidden, torch.zeros(1, T, config.ple_dim), position_ids, shared, mask)
        out = consumer(hidden, torch.zeros(1, T, config.ple_dim), position_ids, shared, mask)
        expected = self.consumer_reference(consumer, shared["local"], hidden, position_ids, mask)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_consumer_grads_flow_to_producer_kv(self):
        config = windowed_config()
        producer = make_block(config, 3)
        consumer = make_block(config, 4)
        T = 5
        hidden = torch.randn(1, T, config.hidden_dim)
        position_ids = torch.arange(T).unsqueeze(0)
        mask = causal_mask(T, config.local_window)
        shared = {}
        producer(hidden, torch.zeros(1, T, config.ple_dim), position_ids, shared, mask)
        # Consumer-ONLY loss: the only producer path into it is the shared K/V.
        consumer(hidden, torch.zeros(1, T, config.ple_dim), position_ids, shared, mask).sum().backward()
        for parameter in (producer.attention.k_proj.weight, producer.attention.v_proj.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0, "producer K/V gradients detached"

    def test_only_last_same_type_prefix_layer_stores(self):
        config = windowed_config()
        assert config.kv_producer(4) == 3 and config.kv_producer(6) == 2
        for layer_idx in range(config.share_boundary):
            store = make_block(config, layer_idx).attention.store_full_length_kv
            assert store == (layer_idx in (3, 2)), f"layer {layer_idx} store={store}"

    def test_training_store_is_request_local(self):
        config = windowed_config()
        producer = make_block(config, 3)
        hidden = torch.randn(1, 4, config.hidden_dim)
        position_ids = torch.arange(4).unsqueeze(0)
        mask = causal_mask(4, config.local_window)
        zeros = torch.zeros(1, 4, config.ple_dim)
        for _ in range(2):  # fresh dict per forward: no persistent cache during training
            shared = {}
            producer(hidden, zeros, position_ids, shared, mask)
            assert set(shared) == {"local"}
