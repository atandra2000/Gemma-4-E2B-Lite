"""Reference eager attention: local/global causality, source Q/K/V norms,
proportional/default RoPE, explicit scale 1.0, and cross-layer KV sharing
(design §4; upstream Gemma4TextAttention / _compute_proportional_rope_parameters).

Training uses no persistent cache: producer layers store their full-length K/V
in the request-local `shared_kv_states` dict (autograd-connected, never
detached) and same-type consumer layers alias them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.ple import RMSNorm


def rotary_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float = 1.0) -> torch.Tensor:
    """Inverse frequencies over head_dim/2 angle pairs. Proportional RoPE keeps
    the encoding head_dim-sized: the non-rotated tail gets zero frequencies
    (identity rotation), and the exponent denominator is the FULL head_dim."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim)
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles, dtype=torch.float32)])
    return inv_freq


def rope_cos_sin(inv_freq: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [B, T, head_dim] from absolute positions [B, T]."""
    freqs = inv_freq.to(position_ids.device)[None, None, :] * position_ids.float()[:, :, None]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is [B, T, heads, head_dim]; cos/sin [B, T, head_dim] (unsqueeze dim 2)."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


def causal_mask(seq_len: int, window: int | None = None) -> torch.Tensor:
    """Additive mask [T, T] (0 keep, -inf skip) over ABSOLUTE positions.
    Global: all preceding keys including self. Local: key <= query and
    query - key < window."""
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    keep = k <= q
    if window is not None:
        keep = keep & (q - k < window)
    return torch.zeros(seq_len, seq_len).masked_fill(~keep, float("-inf"))


class Attention(nn.Module):
    """One layer's attention. Producers compute K/V and store full-length
    states for their type; consumers (shared suffix) have no K/V projections
    or norms and alias the producer's tensors."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)  # "local" | "global"
        self.head_dim = config.head_dim(layer_idx)
        self.num_heads = config.n_heads
        self.scaling = 1.0  # never SDPA's default inverse-square-root scale
        self.is_kv_shared_layer = layer_idx >= config.share_boundary
        self.local_window = config.local_window if self.layer_type == "local" else None

        dim = config.hidden_dim
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(dim, self.head_dim, bias=False)  # one KV head
            self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, config.rms_eps)
            self.v_norm = RMSNorm(self.head_dim, config.rms_eps, with_scale=False)
            prefix_types = [config.layer_type(i) for i in range(config.share_boundary)]
            self.store_full_length_kv = layer_idx == len(prefix_types) - 1 - prefix_types[::-1].index(
                self.layer_type
            )
        else:
            self.store_full_length_kv = False

        if self.layer_type == "global":
            theta, partial = config.global_rope_theta, config.global_rope_partial_factor
        else:
            theta, partial = config.local_rope_theta, 1.0
        self.register_buffer("inv_freq", rotary_inv_freq(self.head_dim, theta, partial), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv_states: dict,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cos, sin = rope_cos_sin(self.inv_freq, position_ids)

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(q, cos, sin).transpose(1, 2)  # [B, H, T, hd]

        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]  # producer's roped/normalized states
        else:
            k = self.k_proj(hidden_states).view(batch, seq_len, 1, self.head_dim)
            k = apply_rope(self.k_norm(k), cos, sin).transpose(1, 2)
            v = self.v_norm(self.v_proj(hidden_states).view(batch, seq_len, 1, self.head_dim))
            v = v.transpose(1, 2)
            if self.store_full_length_kv:
                # autograd-connected storage: consumers must backprop through us
                shared_kv_states[self.layer_type] = (k, v)

        k = k.expand(batch, self.num_heads, seq_len, self.head_dim)  # repeat_kv
        v = v.expand(batch, self.num_heads, seq_len, self.head_dim)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(out)
