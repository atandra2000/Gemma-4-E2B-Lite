"""Checkpoint persistence: one atomic generation per save.

A generation is a complete {model, optimizer, RNG, metadata} tuple written
through sibling temp files and committed by a final atomic pointer rename —
a reader either sees the whole generation or the previous one. Partial
writes (crash mid-save) are never loadable and never overwrite the last
complete generation.
"""
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class CheckpointManager:
    """Atomic generation saves: model.safetensors + optim.pt + rng.pt + meta.json,
    committed via a ``pointer.json`` rename."""

    FILES = ("model.safetensors", "optim.pt", "rng.pt", "meta.json")

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # -- save ----------------------------------------------------------------

    def save(self, model, optimizer, *, step: int, tokens_seen: int,
             rng_states: dict, extra_meta: dict | None = None) -> Path:
        """Write a new generation under ``gen_<step>/`` and point at it.

        The commit is the final ``pointer.json`` rename: everything inside
        the generation directory already exists by then, so a crash before
        the rename leaves the previous generation loadable.
        """
        gen_dir = self.save_dir / f"gen_{step:010d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Same-step re-save (final save after the last window save) replaces
        # the generation in place; the pointer still targets a complete dir
        # until the files are rewritten below, so keep the window short.
        for f in self.FILES:
            (gen_dir / f).unlink(missing_ok=True)

        # Tied embeddings alias one storage; safetensors rejects duplicates.
        independent: dict = {}
        seen: set[int] = set()
        for k, v in model.state_dict().items():
            ptr = v.data_ptr()
            independent[k] = v.clone() if ptr in seen else v
            seen.add(ptr)
        save_file(independent, gen_dir / "model.safetensors")
        torch.save(optimizer.state_dict(), gen_dir / "optim.pt")
        torch.save(rng_states, gen_dir / "rng.pt")

        meta = {"step": step, "tokens_seen": tokens_seen, "format": 1}
        if extra_meta:
            meta.update({k: v for k, v in extra_meta.items() if k != "step"})
        self._atomic_json(gen_dir / "meta.json", meta)

        self._atomic_json(self.save_dir / "pointer.json", {"generation": gen_dir.name})
        return gen_dir

    # -- load ----------------------------------------------------------------

    def load(self, model, optimizer=None, *, device: str = "cpu",
             expect_config_hash: str | None = None) -> dict:
        """Restore the latest complete generation; returns its metadata.

        Raises FileNotFoundError when no complete generation exists (a bare
        ``gen_*`` directory without pointer.json is an incomplete write and
        is ignored, matching the reject-partial contract). Incompatible
        config hashes are rejected without touching the checkpoint.
        """
        gen_dir = self.latest_generation()
        if gen_dir is None:
            raise FileNotFoundError(f"no complete checkpoint under {self.save_dir}")

        meta = json.loads((gen_dir / "meta.json").read_text())
        if expect_config_hash and meta.get("config_hash") not in (None, expect_config_hash):
            raise RuntimeError(
                f"checkpoint config_hash {meta.get('config_hash')!r} != run's "
                f"{expect_config_hash!r} — refusing to resume an incompatible run")

        weights = load_file(str(gen_dir / "model.safetensors"), device=device)
        model.load_state_dict(weights, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(
                torch.load(gen_dir / "optim.pt", map_location=device, weights_only=True))
        rng = torch.load(gen_dir / "rng.pt", map_location="cpu", weights_only=False)
        return {**meta, "rng_states": rng, "generation_dir": str(gen_dir)}

    def latest_generation(self) -> Path | None:
        """The pointed-at generation, but only if every file is present."""
        pointer = self.save_dir / "pointer.json"
        if not pointer.exists():
            return None
        try:
            name = json.loads(pointer.read_text())["generation"]
        except (json.JSONDecodeError, KeyError):
            return None
        gen_dir = self.save_dir / name
        if not all((gen_dir / f).exists() for f in self.FILES):
            return None
        return gen_dir

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _atomic_json(path: Path, obj: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
