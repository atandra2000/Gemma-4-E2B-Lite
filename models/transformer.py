"""Decoder blocks: source norm conventions, gated GELU-tanh MLP, suffix double
width, residual ordering, PLE gate application, the unit layer scalar and
eager reference attention with cross-layer KV sharing (design §4).
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.attention import Attention, causal_mask
from models.ple import RMSNorm


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        ple_signal: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        shared_kv_states: dict | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(hidden_states.shape[0], -1)
        if shared_kv_states is None:
            shared_kv_states = {}
        if attention_mask is None and self.attention is not None:
            window = self.attention.local_window
            attention_mask = causal_mask(seq_len, window)

        if self.attention is not None:
            residual = hidden_states
            hidden_states = self.attention(
                self.input_norm(hidden_states), position_ids, shared_kv_states, attention_mask
            )
            hidden_states = residual + self.post_attention_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_ffn_norm(hidden_states))
        hidden_states = residual + self.post_ffn_norm(hidden_states)

        residual = hidden_states
        hidden_states = F.gelu(self.ple_gate(hidden_states), approximate="tanh")
        hidden_states = hidden_states * ple_signal
        hidden_states = self.post_ple_norm(self.ple_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states * self.layer_scalar
