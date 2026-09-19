"""Training dataset over packed uint32 shards: flat x0 windows, no next-token shift.

The causal-CE loss applies the +1 shift itself (training/losses.py), so
windows are flat ``seq_len``-token slices. Shards are the ``shard_*.bin``
uint32 memmaps written by ``shared_data`` (via ``data/prepare_data.py``);
this adapter reads them through ``shared_data.dataset.ShardDataset`` —
manifest validation, checksums, and dtype stay the pipeline's job — and
windows the concatenated stream WITHOUT copying tokens into host memory
(no corpus duplication; reads cross shard boundaries through the mmap).
"""
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ShardWindows(Dataset):
    """Windows of exactly ``seq_len`` tokens from manifest-validated shards.

    The window stream is the logical concatenation of all shards; a window
    that crosses a shard boundary is served by two memmap reads. Partial
    tails (a shard whose length is not a multiple of seq_len) are bridged:
    the tail tokens remain part of the stream, they just fall inside a
    window that starts in the previous shard. Only a final partial window at
    the very end of the stream is dropped.
    """

    def __init__(self, data_dir, seq_len: int, manifest_path: Path | None = None,
                 *, expected_tokenizer: str = "gpt2", vocab_size: int | None = None):
        from shared_data.dataset import ShardDataset

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self.seq_len = seq_len
        self.reader = ShardDataset(data_dir, manifest_path=manifest_path)
        manifest = self.reader.manifest

        # Tokenizer contract: a manifest from a different tokenizer family is
        # silently-wrong data (same file shape, different token semantics).
        if expected_tokenizer and manifest.tokenizer_name != expected_tokenizer:
            raise ValueError(
                f"tokenizer mismatch: manifest says {manifest.tokenizer_name!r}, "
                f"this project requires {expected_tokenizer!r}")
        # Every manifest-listed shard must exist at its recorded length.
        missing = [s.path for s in manifest.shards
                   if not (self.reader.data_dir / s.path).exists()]
        if missing:
            raise FileNotFoundError(f"manifest lists missing shard files: {missing}")

        self.shards = self.reader.shards
        for info, mm in zip(manifest.shards, self.shards):
            if mm.shape[0] != info.n_tokens:
                raise ValueError(
                    f"shard {info.path}: {mm.shape[0]} tokens on disk, manifest "
                    f"records {info.n_tokens} (corrupt or wrong shard)")
        self.n_tokens = sum(mm.shape[0] for mm in self.shards)
        self.n_windows = self.n_tokens // seq_len
        if self.n_windows == 0:
            raise ValueError(
                f"corpus too small: {self.n_tokens} tokens < seq_len {seq_len}")
        self._vocab_size = vocab_size

    def validate(self) -> None:
        """Sampled token-boundary check: packed IDs stay under the contract
        vocab (a wrong tokenizer with the same vocab file shape fails here).
        Raises ValueError on the first out-of-range token found in sampled
        windows; cheap enough for preflight, not an exhaustive scan."""
        if self._vocab_size is None:
            return
        stride = max(1, len(self) // 16)
        for w in range(0, len(self), stride):
            bad = (self[w] >= self._vocab_size).nonzero()
            if bad.numel():
                raise ValueError(
                    f"window {w}: token id {int(self[w][bad[0][0]])} >= vocab_size "
                    f"{self._vocab_size} — wrong tokenizer packed this corpus")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        w = int(idx) % self.n_windows
        start = w * self.seq_len
        end = start + self.seq_len
        out = np.empty(self.seq_len, dtype=np.int64)
        filled = 0
        offset = 0
        for mm in self.shards:  # few shards; bisect overhead not worth it
            if filled == self.seq_len:
                break
            nxt = offset + mm.shape[0]
            lo, hi = max(start, offset), min(end, nxt)
            if lo < hi:
                take = np.asarray(mm[lo - offset:hi - offset], dtype=np.int64)
                out[filled:filled + (hi - lo)] = take
                filled += hi - lo
            offset = nxt
        return torch.from_numpy(out)


class ShuffledRangeSampler(Sampler):
    """Deterministic, resumable window shuffler (house contract, shared with
    DiffusionGemma-Lite and shared_data.loader).

    The permutation is fixed by (seed, n_windows); ``offset`` restarts
    mid-order after a checkpoint resume without regenerating any draws.
    """

    def __init__(self, n_windows: int, seed: int = 42, offset: int = 0):
        if n_windows <= 0:
            raise ValueError(f"no complete windows available (n_windows={n_windows})")
        self.n_windows = int(n_windows)
        self.offset = int(offset) % self.n_windows
        self.indices = np.random.default_rng(seed).permutation(n_windows)

    def __iter__(self):
        for i in range(self.offset, len(self.indices)):
            yield int(self.indices[i])

    def __len__(self):
        return len(self.indices) - self.offset


def build_dataloader(data_dir, seq_len, batch_size, seed=42, offset_windows=0,
                     pin_memory=False, manifest_path: Path | None = None):
    """Deterministic shuffled loader over shard windows; offset resumes the order."""
    ds = ShardWindows(data_dir, seq_len, manifest_path=manifest_path)
    sampler = ShuffledRangeSampler(len(ds), seed=seed, offset=offset_windows)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True,
                      pin_memory=pin_memory)
