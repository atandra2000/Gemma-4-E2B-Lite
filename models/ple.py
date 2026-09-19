"""Per-Layer Embeddings (PLE) and the source RMS norm.

Semantics pinned to the upstream reference (v5.15.1 Gemma4RMSNorm,
get_per_layer_inputs / project_per_layer_inputs) and the design spec §4.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Source norm convention: FP32 compute, x * rsqrt(mean(x^2) + eps) * weight.

    The scale is a plain multiplier in Gemma 4 — not the (1 + weight) offset
    of older Gemma generations. with_scale=False (the source V-norm) normalizes
    with no learnable weight at all.
    """

    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(dtype)


class PLE(nn.Module):
    """Per-layer embeddings: token-identity table plus projected-input branch.

    forward() returns the mixed signal stack [B, T, L, P]; each decoder layer
    consumes its own slice [:, :, layer_idx, :] (upstream model-loop contract).
    """

    def __init__(self, config):
        super().__init__()
        L, P = config.n_layers, config.ple_dim
        self.n_layers = L
        self.ple_dim = P
        self.hidden_dim = config.hidden_dim
        self.table = nn.Embedding(config.vocab_size, L * P)  # scaled by sqrt(P)
        self.input_proj = nn.Linear(config.hidden_dim, L * P, bias=False)  # scaled by 1/sqrt(D)
        self.projection_norm = RMSNorm(P, config.rms_eps)
        self.mix_scale = 2.0**-0.5  # (identity + projected) / sqrt(2)

    def forward(self, input_ids: torch.Tensor, scaled_embeddings: torch.Tensor) -> torch.Tensor:
        # Identity branch: token table scaled by sqrt(P), reshaped to [B,T,L,P].
        identity = self.table(input_ids) * (self.ple_dim**0.5)
        identity = identity.reshape(*input_ids.shape, self.n_layers, self.ple_dim)
        # Projected branch: SCALED main embeddings -> L*P, 1/sqrt(D), P-dim RMSNorm.
        # (ponytail: the 1/sqrt(D) factor is inert through the following norm —
        # RMSNorm is scale-invariant — but it is the design §4 contract, kept.)
        projected = self.input_proj(scaled_embeddings) * (self.hidden_dim**-0.5)
        projected = projected.reshape(*input_ids.shape, self.n_layers, self.ple_dim)
        projected = self.projection_norm(projected)
        return (projected + identity) * self.mix_scale
