"""Decoder blocks: source norm conventions, gated GELU-tanh MLP, suffix double
width, residual ordering, PLE gate application, the unit layer scalar and
eager reference attention with cross-layer KV sharing (design §4). The full
model wires scaled main embedding + PLE through the blocks, final norm, tied
output projection and logit softcap (design §4 forward pipeline).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.attention import Attention, causal_mask
from models.cache import ProducerKVCache
from models.ple import PLE, RMSNorm


class GatedGeluMlp(nn.Module):
    """down(gelu_tanh(gate(x)) * up(x)); intermediate width doubles on the
    shared suffix (config.mlp_hidden). No biases (source convention)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden = config.mlp_hidden(layer_idx)
        dim = config.hidden_dim
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Block(nn.Module):
    """One decoder layer (upstream Gemma4TextDecoderLayer, non-MoE):

        residual = h
        h = input_norm(h); h = attention(h)            # Task 4
        h = post_attention_norm(h); h = residual + h   # post-norm before residual add
        residual = h
        h = mlp(pre_ffn_norm(h)); h = post_ffn_norm(h); h = residual + h
        residual = h
        h = ple_proj(gelu_tanh(ple_gate(h)) * ple_signal)
        h = post_ple_norm(h); h = residual + h
        h *= layer_scalar                              # unit buffer, not a parameter
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        dim, eps = config.hidden_dim, config.rms_eps
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(dim, eps)
        self.post_attention_norm = RMSNorm(dim, eps)
        self.pre_ffn_norm = RMSNorm(dim, eps)
        self.post_ffn_norm = RMSNorm(dim, eps)
        self.post_ple_norm = RMSNorm(dim, eps)
        self.mlp = GatedGeluMlp(config, layer_idx)
        self.ple_gate = nn.Linear(dim, config.ple_dim, bias=False)
        self.ple_proj = nn.Linear(config.ple_dim, dim, bias=False)
        self.layer_scalar = nn.Buffer(torch.ones(1))
        self.attention = Attention(config, layer_idx)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        ple_signal: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        shared_kv_states: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            start = shared_kv_states.num_tokens if isinstance(shared_kv_states, ProducerKVCache) else 0
            position_ids = torch.arange(start, start + seq_len).unsqueeze(0).expand(hidden_states.shape[0], -1)
        if shared_kv_states is None:
            shared_kv_states = {}
        if isinstance(shared_kv_states, ProducerKVCache):
            # Per-row-position causal mask per layer type; padding keys are
            # masked through the cache-tracked `key_valid` validity vector.
            attention_mask = shared_kv_states.attention_mask(
                self.layer_idx, position_ids, key_valid, hidden_states.device
            )
        elif attention_mask is None and self.attention is not None:
            window = self.attention.local_window
            attention_mask = causal_mask(seq_len, window)

        if self.attention is not None:
            residual = hidden_states
            normed = self.input_norm(hidden_states)
            if self.gradient_checkpointing:
                # Recomputation re-runs this exact call with the same tensor
                # arguments (RNG state preserved by checkpoint's default). The
                # producer/consumer split is inside the re-run too, so producer
                # K/V are recomputed fresh and consumers re-alias them — nothing
                # is read from or written to a mutable dict at backward time.
                normed = checkpoint(
                    self.attention, normed, position_ids, shared_kv_states,
                    attention_mask, key_valid, use_reentrant=False,
                )
            else:
                normed = self.attention(
                    normed, position_ids, shared_kv_states, attention_mask, key_valid
                )
            hidden_states = residual + self.post_attention_norm(normed)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_ffn_norm(hidden_states))
        hidden_states = residual + self.post_ffn_norm(hidden_states)

        residual = hidden_states
        hidden_states = F.gelu(self.ple_gate(hidden_states), approximate="tanh")
        hidden_states = hidden_states * ple_signal
        hidden_states = self.post_ple_norm(self.ple_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states * self.layer_scalar


class Gemma4LiteModel(nn.Module):
    """Full model (design §4): IDs -> scaled main embedding + PLE -> blocks ->
    final RMSNorm -> tied vocabulary projection -> softcap.

    forward_hidden() returns the normalized final hidden states [B, T, D] for
    the chunked training loss (Task 8); forward() returns softcapped logits
    [B, T, V]. With tie_embeddings the head is the embedding matrix applied by
    F.linear — no second vocabulary storage exists to reconcile.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_scale = config.hidden_dim**0.5
        self.ple = PLE(config)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_eps)
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.checkpoint_blocks(False)

    def checkpoint_blocks(self, enabled: bool) -> None:
        for block in self.blocks:
            block.gradient_checkpointing = enabled

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: ProducerKVCache | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids) * self.embed_scale
        ple_signal = self.ple(input_ids, x)  # [B, T, L, P]
        if position_ids is None and cache is not None and key_valid is not None:
            # Derive ONCE, before any producer appends this call's tokens:
            # valid_counts() grows as soon as the first global producer runs,
            # so per-block derivation would rope later blocks one step ahead.
            position_ids = cache.valid_counts(input_ids.shape[0])[:, None] + (
                key_valid.cumsum(1) - 1
            ).clamp(min=0)
        # One dict across the loop for training; with a cache, producers append
        # and consumers alias per type. finish_step() commits/trims after the
        # last consumer has run.
        shared_kv_states: dict | ProducerKVCache = cache if cache is not None else {}
        for i, block in enumerate(self.blocks):
            x = block(x, ple_signal[:, :, i, :], position_ids, shared_kv_states, None, key_valid)
        if cache is not None:
            cache.finish_step()
        return self.final_norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: ProducerKVCache | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, position_ids, cache, key_valid)
        weight = self.embed.weight if self.config.tie_embeddings else self.lm_head.weight
        logits = F.linear(hidden, weight)
        cap = self.config.logit_softcap
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        return logits
