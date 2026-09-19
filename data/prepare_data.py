"""Project-specific wrapper around the workspace-wide tokenization pipeline.

It supplies the GPT-2 tokenizer contract required by Gemma-4-E2B-Lite, writes
a local configuration override, and forwards download, tokenize, and packing
options to the shared pipeline. Mirrors DiffusionGemma-Lite/data/prepare_data.py
(house shim pattern; the shared pipeline owns download/clean/dedup/pack).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent  # .../CoreProjects/LLM/ (workspace shared_data lives here)


def _require_shared_data() -> None:
    """Raise a clean, actionable error if the shared pipeline is missing."""
    workspace = _LLM_ROOT / "shared_data" if _LLM_ROOT.exists() else None
    if workspace is not None and workspace.exists():
        return
    raise FileNotFoundError(
        "Gemma-4-E2B-Lite data prep requires the workspace `shared_data` package. "
        f"Expected it at {workspace}. Run data prep from a CoreProjects checkout."
    )


for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Contract with training/pretrain.py: the shared pipeline packs shards to
# <DATA_ROOT>/shards/; pack runs as a subprocess that only honors $LLM_DATA_ROOT.
DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data" / "pretrain_corpus"

GEMMA4_TOKENIZER_NAME = "gpt2"
GEMMA4_VOCAB_SIZE = 50_257
GEMMA4_EOS_TOKEN_ID = 50_256
GEMMA4_PAD_TOKEN_ID = 50_256


def _ensure_gemma4_data_config(project_root: Path) -> Path:
    """Materialise a project-local data_config.yaml with Gemma-4-E2B-Lite's vocab."""
    from shared_data.config import UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.common import load_yaml

    out_path = project_root / "data" / "data_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
    cfg["pipeline"]["tokenizer"]["name"] = GEMMA4_TOKENIZER_NAME
    cfg["pipeline"]["tokenizer"]["vocab_size"] = GEMMA4_VOCAB_SIZE
    cfg["pipeline"]["tokenizer"]["eos_token_id"] = GEMMA4_EOS_TOKEN_ID
    cfg["pipeline"]["tokenizer"]["pad_token_id"] = GEMMA4_PAD_TOKEN_ID
    cfg["_generator"] = "Gemma-4-E2B-Lite/data/prepare_data.py"
    cfg["_tokenizer_family"] = "gpt2"

    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _apply_gemma4_defaults() -> Path:
    """Create the local pipeline config with the model's tokenizer settings."""
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS

    print(f"[data/gemma4] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/gemma4] tokenizer: {GEMMA4_TOKENIZER_NAME} "
          f"(vocab={GEMMA4_VOCAB_SIZE:,}, EOS={GEMMA4_EOS_TOKEN_ID})")
    return _ensure_gemma4_data_config(_PROJECT_ROOT)


def main() -> int:
    """Parse CLI overrides and run the shared data preparation pipeline."""
    _require_shared_data()

    parser = argparse.ArgumentParser(
        description="Gemma-4-E2B-Lite data prep (delegates to universal pipeline)"
    )
    parser.add_argument("--mixture", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None,
                        help=f"Output root for shards (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    project_data_config = _apply_gemma4_defaults()

    data_root = Path(args.data_root).resolve() if args.data_root else DEFAULT_DATA_ROOT
    # pack_shards runs as a subprocess and re-resolves DATA_ROOT from the
    # environment; run_pipeline's in-process set_data_root does not reach it.
    os.environ["LLM_DATA_ROOT"] = str(data_root)
    print(f"[data/gemma4] data root: {data_root} (shards → {data_root / 'shards'})")

    from shared_data.config import UNIVERSAL_MIXTURE_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else project_data_config,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=data_root,
    )


if __name__ == "__main__":
    sys.exit(main())
