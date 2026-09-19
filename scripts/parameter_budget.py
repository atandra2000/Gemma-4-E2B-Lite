#!/usr/bin/env python3
"""Print the analytic parameter ledger and memory assumptions for a config.

Analytic counts only (design §3): the exact total comes from instantiating the
model on CPU/meta in Task 5, so no "exact" total is printed here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `models`

from models.config import (  # noqa: E402
    ModelConfig,
    count_large_matrices,
    count_norms_and_buffers,
    memory_assumptions,
)

DESIGN_SUBTOTAL = 348_882_944  # design §3 large-matrix subtotal, production config


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
    print("  exact total: pending Task 5 instantiated-model gate (CPU/meta)")

    print("\nMemory assumptions (design §3, conservative):")
    for key, value in memory.items():
        print(f"  {key:34s} {value:13.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
