#!/usr/bin/env python3
"""Print the analytic parameter ledger and memory assumptions for a config.

Reconciles the analytic counts against the instantiated model on the meta
device (exact total, Task 5 gate): every parameter is classified into a ledger
group, embedding tying is verified structurally, and absent consumer K/V and
norm weights are asserted. Any mismatch raises instead of printing a total.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `models`

import torch  # noqa: E402

from models.config import (  # noqa: E402
    ModelConfig,
    count_large_matrices,
    count_norms_and_buffers,
    memory_assumptions,
)

DESIGN_SUBTOTAL = 348_882_944  # design §3 large-matrix subtotal, production config


def classify_parameter(name: str) -> str:
    """Map a Gemma4LiteModel parameter name to its analytic ledger group."""
    if name == "embed.weight":
        return "main_embedding_tied"
    if name == "lm_head.weight":
        raise AssertionError("untied lm_head storage is not in the design ledger")
    if name == "ple.table.weight":
        return "ple_token_table"
    if name == "ple.input_proj.weight" or ".ple_gate.weight" in name or ".ple_proj.weight" in name:
        return "ple_projections"
    if name == "ple.projection_norm.weight":
        return "ple_projection_norm"
    if ".mlp." in name:
        return "mlp_matrices"
    if ".attention.q_proj." in name or ".attention.o_proj." in name:
        return "q_and_output_projections"
    if ".attention.k_proj." in name or ".attention.v_proj." in name:
        return "kv_projections_prefix_only"
    if ".attention.q_norm." in name:
        return "q_norms"
    if ".attention.k_norm." in name:
        return "kv_norms_producer_only"
    if name == "final_norm.weight":
        return "final_norm"
    if name.endswith((".input_norm.weight", ".post_attention_norm.weight",
                      ".pre_ffn_norm.weight", ".post_ffn_norm.weight",
                      ".post_ple_norm.weight")):
        return "layer_norms"
    raise AssertionError(f"parameter {name!r} does not belong to any ledger group")


def reconcile_with_ledger(config: ModelConfig) -> dict[str, int]:
    """Instantiate on the meta device and reconcile every parameter with the
    analytic ledger. Returns the exact group counts; raises on any mismatch."""
    from models.transformer import Gemma4LiteModel

    if not config.tie_embeddings:
        raise NotImplementedError("untied-head ledger is a design revision, not a task 5 gate")

    with torch.device("meta"):
        model = Gemma4LiteModel(config)
    parameters = dict(model.named_parameters())  # unique storages: tied head counted once
    buffers = list(model.named_buffers())

    # Structural sharing contract: the tied head is not a second storage, and
    # shared-suffix layers have no K/V projections or K/V norm weights at all
    # (the V norm normalizes without a weight even on producers).
    assert not any("lm_head" in n for n in parameters)
    assert not any("v_norm" in n for n in parameters)
    for i in config.shared_layers():
        absent = (f"blocks.{i}.attention.k_proj", f"blocks.{i}.attention.v_proj",
                  f"blocks.{i}.attention.k_norm", f"blocks.{i}.attention.v_norm")
        assert not any(any(n.startswith(a + ".") for n in parameters) for a in absent)

    scalars = [n for n, _ in buffers if n.endswith("layer_scalar")]
    assert len(scalars) == config.n_layers, f"expected {config.n_layers} unit layer scalars"

    instantiated: dict[str, int] = {}
    for name, parameter in parameters.items():
        group = classify_parameter(name)
        instantiated[group] = instantiated.get(group, 0) + parameter.numel()

    expected = count_large_matrices(config) | count_norms_and_buffers(config)
    for group, count in expected.items():
        if group in ("subtotal_excluding_norms", "unit_layer_scalar_buffers"):
            continue  # derived summary / buffers, not parameter groups
        got = instantiated.get(group, 0)
        if got != count:
            raise AssertionError(
                f"ledger mismatch for {group}: instantiated {got:,} != analytic {count:,}"
            )
    unknown = set(instantiated) - set(expected)
    assert not unknown, f"instantiated groups missing from the ledger: {sorted(unknown)}"
    return instantiated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    large = count_large_matrices(config)
    norms = count_norms_and_buffers(config)
    memory = memory_assumptions(config)

    design_row = {
        "main_embedding_tied": 38_597_376,
        "ple_token_table": 64_328_960,
        "ple_projections": 2_949_120,
        "mlp_matrices": 212_336_640,
        "q_and_output_projections": 28_311_552,
        "kv_projections_prefix_only": 2_359_296,
    }

    print(f"config: {args.config}")
    if config.vocab_size == 50257 and config.n_layers == 20 and config.hidden_dim == 768:
        for key, expected in design_row.items():
            marker = "ok " if large[key] == expected else "MISMATCH (design: "
            if large[key] != expected:
                print(f"  {key}: {large[key]:,} {marker}{expected:,})")
        if large["subtotal_excluding_norms"] != DESIGN_SUBTOTAL:
            print(f"  subtotal MISMATCH vs design {DESIGN_SUBTOTAL:,}")

    print("\nLarge matrices (analytic, excluding norms):")
    for key, value in large.items():
        print(f"  {key:34s} {value:>13,}")

    print("\nNorm parameters and buffers (distinguished, design §3):")
    for key, value in norms.items():
        kind = "buffer" if "buffers" in key else "params"
        print(f"  {key:34s} {value:>13,}  ({kind})")

    total = large["subtotal_excluding_norms"] + sum(
        v for k, v in norms.items() if "buffers" not in k
    )
    print(f"\n  analytic total incl. norms: {total:,}")

    instantiated = reconcile_with_ledger(config)
    exact = sum(instantiated.values())
    print(f"\nInstantiated model (meta device, exact):")
    print(f"  parameters (unique storages): {exact:,}")
    print(f"  unit layer-scalar buffers:    {config.n_layers}")
    verdict = "matches the analytic ledger" if exact == total else "MISMATCH vs analytic total"
    print(f"  reconciliation: {verdict}")

    print("\nMemory assumptions (design §3, conservative):")
    for key, value in memory.items():
        print(f"  {key:34s} {value:13.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
