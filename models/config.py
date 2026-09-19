"""Validated model configuration (design spec §3) and its analytic ledger.

Owns the layer-type map, the KV-producer map and the large-matrix parameter
accounting. The instantiated-model exact total is a Task 5 gate; everything
here is the analytic ledger from the design table.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml

LayerType = Literal["local", "global"]

# weights + gradients + optimizer moments + master copy (design §3 memory note)
_BYTES_PER_PARAM = 16


@dataclass(frozen=True)
class ModelConfig:
    """Architecture contract. Field names match the `model:` section of configs/*.yaml."""

    # Vocabulary / tokenizer (pinned generation convention, design §3)
    vocab_size: int = 50257
    eos_id: int = 50256
    # Core dimensions
    hidden_dim: int = 768  # D
    n_layers: int = 20
    n_heads: int = 6  # one KV head per layer (multi-query), pinned by design
    local_head_dim: int = 128
    global_head_dim: int = 256  # must stay 2x local_head_dim (pinned 1:2 relation)
    # Layer pattern and sharing
    global_layers: tuple[int, ...] = (4, 9, 14, 19)  # zero-indexed; final layer global
    share_boundary: int = 10  # prefix 0..boundary-1 produces K/V; suffix consumes
    # PLE
    ple_dim: int = 64  # P; L == n_layers
    # MLP (gated GELU-tanh; shared suffix doubles the intermediate width)
    mlp_hidden_dim: int = 3072
    mlp_suffix_multiplier: int = 2
    # Attention
    local_window: int = 512
    local_rope_theta: float = 10_000.0  # default (full-dim) RoPE
    global_rope_theta: float = 1_000_000.0  # proportional RoPE
    global_rope_partial_factor: float = 0.25  # rotary subspace = factor * global_head_dim
    attention_scale: float = 1.0  # never SDPA's default 1/sqrt(d)
    # Norm / output
    rms_eps: float = 1e-6
    logit_softcap: float = 30.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError(
                f"eos_id {self.eos_id} outside token range [0, {self.vocab_size})"
            )
        if self.hidden_dim < 1 or self.n_layers < 1 or self.n_heads < 1:
            raise ValueError("hidden_dim, n_layers, n_heads must be positive")
        if self.n_heads * self.local_head_dim != self.hidden_dim:
            raise ValueError(
                f"head grouping broken: n_heads*local_head_dim = "
                f"{self.n_heads * self.local_head_dim} != hidden_dim {self.hidden_dim}"
            )
        if self.global_head_dim != 2 * self.local_head_dim:
            raise ValueError(
                f"global_head_dim {self.global_head_dim} != 2 * local_head_dim "
                f"{2 * self.local_head_dim} (pinned 1:2 relation)"
            )
        if self.local_head_dim % 2:
            raise ValueError(f"local_head_dim {self.local_head_dim} must be even (RoPE pairs)")
        gl = tuple(self.global_layers)
        if len(set(gl)) != len(gl) or gl != tuple(sorted(gl)):
            raise ValueError(f"global_layers must be sorted and unique, got {gl}")
        if any(not 0 <= i < self.n_layers for i in gl):
            raise ValueError(f"global_layers {gl} out of range for n_layers {self.n_layers}")
        if not gl:
            raise ValueError("global_layers is empty: missing global producer type")
        if gl[-1] != self.n_layers - 1:
            raise ValueError(f"final layer must be global; got global_layers {gl}")
        if not 1 <= self.share_boundary <= self.n_layers:
            raise ValueError(
                f"share_boundary {self.share_boundary} leaves no producer prefix "
                f"(must be in [1, {self.n_layers}])"
            )
        # Every consumer layer needs a same-type producer in the prefix.
        for i in self.shared_layers():
            if self.kv_producer(i) is None:
                raise ValueError(
                    f"consumer layer {i} ({self.layer_type(i)}) has no "
                    f"{self.layer_type(i)} producer below share_boundary "
                    f"{self.share_boundary}"
                )
        if self.ple_dim < 1:
            raise ValueError(f"ple_dim must be >= 1, got {self.ple_dim}")
        if self.mlp_hidden_dim < 1 or self.mlp_suffix_multiplier < 1:
            raise ValueError("mlp_hidden_dim and mlp_suffix_multiplier must be positive")
        if self.local_window < 1:
            raise ValueError(f"local_window must be >= 1, got {self.local_window}")
        # Global rotary subspace: even, positive, within the head dimension.
        grot = self.global_rotary_dim
        if grot is None or not 0 < grot <= self.global_head_dim or grot % 2:
            raise ValueError(
                f"global rotary subspace {grot} invalid: must be an even "
                f"0 < d <= global_head_dim {self.global_head_dim} "
                f"(factor {self.global_rope_partial_factor})"
            )
        if self.attention_scale != 1.0:
            raise ValueError(
                f"attention_scale must be 1.0 (design §3), got {self.attention_scale}"
            )

    # -- layer maps ---------------------------------------------------------

    def layer_type(self, layer_index: int) -> LayerType:
        if layer_index in self.global_layers:
            return "global"
        return "local"

    def shared_layers(self) -> range:
        return range(self.share_boundary, self.n_layers)

    def kv_producer(self, layer_index: int) -> int | None:
        """Pure producer map: prefix layers self-produce; consumers inherit the
        last same-type prefix layer (design §4: shared locals <- 8, shared
        globals <- 9 for the production config)."""
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer_index {layer_index} out of range [0, {self.n_layers})")
        if layer_index < self.share_boundary:
            return layer_index
        want = self.layer_type(layer_index)
        for j in range(self.share_boundary - 1, -1, -1):
            if self.layer_type(j) == want:
                return j
        return None  # prevented by validation; kept so the map is total

    # -- per-layer shapes ---------------------------------------------------

    def head_dim(self, layer_index: int) -> int:
        return self.global_head_dim if self.layer_type(layer_index) == "global" else self.local_head_dim

    def qkv_dim(self, layer_index: int) -> int:
        """Width of the layer's Q projection (= n_heads * head_dim)."""
        return self.n_heads * self.head_dim(layer_index)

    def mlp_hidden(self, layer_index: int) -> int:
        factor = self.mlp_suffix_multiplier if layer_index >= self.share_boundary else 1
        return self.mlp_hidden_dim * factor

    @property
    def global_rotary_dim(self) -> int | None:
        d = float(self.global_rope_partial_factor) * self.global_head_dim
        return int(d) if d.is_integer() else None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "global_layers" in kwargs:
            kwargs["global_layers"] = tuple(kwargs["global_layers"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        doc = yaml.safe_load(Path(path).read_text())
        if not isinstance(doc, dict) or "model" not in doc:
            raise ValueError(f"{path}: expected a top-level 'model:' section")
        return cls.from_dict(doc["model"])


# -- analytic ledger (design §3 "Parameter and memory accounting") -----------

def count_large_matrices(config: ModelConfig) -> dict[str, int]:
    """Large-matrix counts. Reproduces the design table exactly; norm weights,
    unit layer-scalar buffers and tying distinctions are reported separately."""
    D, V, L, P = config.hidden_dim, config.vocab_size, config.n_layers, config.ple_dim
    boundary = config.share_boundary
    embedding = V * D  # main embedding; tied LM head counted once
    ple_table = V * L * P
    ple_projections = D * L * P + L * 2 * D * P  # input D->L*P; per-block gate D->P + P->D
    mlp = sum(3 * D * config.mlp_hidden(i) for i in range(L))
    qo = sum(2 * D * config.qkv_dim(i) for i in range(L))  # Q and output projections
    kv = sum(
        2 * D * config.head_dim(i) for i in range(boundary)
    )  # one KV head per layer; producers only
    counts = {
        "main_embedding_tied": embedding,
        "ple_token_table": ple_table,
        "ple_projections": ple_projections,
        "mlp_matrices": mlp,
        "q_and_output_projections": qo,
        "kv_projections_prefix_only": kv,
    }
    counts["subtotal_excluding_norms"] = sum(counts.values())
    return counts


def count_norms_and_buffers(config: ModelConfig) -> dict[str, int]:
    """Norm weights (parameters) and unit layer-scalar buffers (not parameters),
    reported so the ledger can distinguish them from the large matrices.
    Norm set matches the v5.15.1 source: input, post-attention, pre-FFN,
    post-FFN and post-PLE norms per layer, one shared P-dim projection norm,
    Q norm per layer, K/V norms on producers only, final norm."""
    D, P = config.hidden_dim, config.ple_dim
    per_layer_norms = 5 * D
    q_norms = sum(config.head_dim(i) for i in range(config.n_layers))
    # K norm has a weight; the V norm normalizes WITHOUT one (source convention).
    kv_norms = sum(config.head_dim(i) for i in range(config.share_boundary))
    return {
        "layer_norms": per_layer_norms * config.n_layers,
        "ple_projection_norm": P,
        "q_norms": q_norms,
        "kv_norms_producer_only": kv_norms,
        "final_norm": D,
        "unit_layer_scalar_buffers": config.n_layers,  # buffers, not parameters
    }


def memory_assumptions(config: ModelConfig) -> dict[str, float]:
    """Conservative training-memory assumptions in GiB (design §3)."""
    subtotal = count_large_matrices(config)["subtotal_excluding_norms"]
    t = 4096  # design's logit-memory example: B=1, T=4096
    logits_bf16 = config.vocab_size * t * 2
    logits_fp32 = config.vocab_size * t * 4
    return {
        "trainable_bytes_16_per_param_gib": subtotal * _BYTES_PER_PARAM / 2**30,
        "full_logits_b1_t4096_bf16_gib": logits_bf16 / 2**30,
        "full_logits_b1_t4096_fp32_gib": logits_fp32 / 2**30,
    }


def production_config() -> ModelConfig:
    """The design §3 20-layer configuration."""
    return ModelConfig()


def tiny_config() -> ModelConfig:
    """Tiny CPU-test configuration: two full local/global periods after the
    producer prefix, valid same-type producers for every consumer."""
    return ModelConfig(
        vocab_size=50257,
        eos_id=50256,
        hidden_dim=64,
        n_layers=12,
        n_heads=2,
        local_head_dim=32,
        global_head_dim=64,
        global_layers=(2, 5, 8, 11),
        share_boundary=6,
        ple_dim=8,
        mlp_hidden_dim=128,
        mlp_suffix_multiplier=2,
        local_window=32,
    )
