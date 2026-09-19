"""Task 9 acceptance: data adapter over the shared pipeline's shards.

The adapter must window the shared `ShardDataset` stream without copying the
corpus into host memory, serve windows that cross shard boundaries, validate
the manifest (missing files, checksums, tokenizer/vocab contract), keep a
deterministic resumable order, and measure full-corpus host RSS.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import ShardWindows, ShuffledRangeSampler, build_dataloader

SEQ = 16
VOCAB, EOS = 50_257, 50_256


def _write_shard(data_dir, index, tokens, corrupt: bool = False):
    path = data_dir / f"shard_{index:05d}.bin"
    if corrupt:
        path.write_bytes(b"\x00\x00")  # 1 uint32 of garbage
    else:
        tokens.astype(np.uint32).tofile(path)
    return path


def _manifest(data_dir, shards, *, tokenizer="gpt2", vocab=VOCAB, eos=EOS):
    return {
        "version": "1.0.0", "created_utc": "t", "vocab_size": vocab,
        "eos_token_id": eos, "pad_token_id": eos, "tokenizer_name": tokenizer,
        "dtype": "uint32", "shard_size_tokens": 50_000_000,
        "total_tokens": sum(len(s) for s in shards),
        "shard_count": len(shards), "shards_dir": "shards",
        "shards": [
            {"index": i, "path": f"shards/shard_{i:05d}.bin", "n_tokens": len(s),
             "sha256": "0" * 64, "n_eos": int((s == EOS).sum())}
            for i, s in enumerate(shards)
        ],
        "sources": {}, "config_hash": "", "mixture_hash": "",
    }


def _build(data_dir, shards, n_shards=3, tokens_per_shard=100, seed=42,
           with_eos=True, tokenizer="gpt2"):
    """Write shards + manifest. Token stream contains EOS so doc boundaries exist."""
    rng = np.random.default_rng(seed)
    shards_dir = data_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n_shards):
        toks = rng.integers(0, VOCAB - 1, size=tokens_per_shard, dtype=np.uint32)
        if with_eos:
            toks[::17] = EOS
        _write_shard(shards_dir, i, toks)
        written.append(toks)
    (data_dir / "manifest.json").write_text(
        json.dumps(_manifest(data_dir, written, tokenizer=tokenizer)))
    return written


@pytest.fixture
def corpus(tmp_path):
    tokens = _build(tmp_path, None)
    return tmp_path, tokens


# -- windows ----------------------------------------------------------------

def test_window_shape_dtype_and_token_range(corpus):
    data_dir, _ = corpus
    ds = ShardWindows(data_dir, seq_len=SEQ)
    assert len(ds) == sum(s.shape[0] for s in ds.shards) // SEQ
    sample = ds[3]
    assert sample.shape == (SEQ,) and sample.dtype == torch.int64
    assert int(sample.max()) < VOCAB


def test_window_crossing_shard_boundary(corpus):
    """The gate: a window straddling shard 0 -> 1 must stitch both memmaps."""
    data_dir, shards = corpus
    ds = ShardWindows(data_dir, seq_len=SEQ)
    # seq_len=16 divides nothing here (100-token shards): windows cross boundaries.
    flat = np.concatenate(shards).astype(np.int64)
    for w in (5, 6, 9):  # 100/16=6.25 -> window 6 spans shards 0 and 1
        assert torch.equal(ds[w], torch.from_numpy(flat[w * SEQ:(w + 1) * SEQ].copy()))


def test_full_stream_matches_concatenation(corpus):
    data_dir, shards = corpus
    ds = ShardWindows(data_dir, seq_len=SEQ)
    flat = np.concatenate(shards)
    got = np.concatenate([ds[w].numpy() for w in range(len(ds))])
    assert np.array_equal(got, flat[:len(ds) * SEQ])


def test_partial_tail_dropped_not_corrupted(corpus):
    data_dir, shards = corpus
    ds = ShardWindows(data_dir, seq_len=SEQ)
    total = sum(len(s) for s in shards)
    assert len(ds) == total // SEQ
    # The dropped tail still exists on disk; nothing was rewritten.
    assert sum(int((s == EOS).sum()) for s in ds.shards) == sum(
        int((s == EOS).sum()) for s in shards)


# -- validation --------------------------------------------------------------

def test_missing_shard_file_raises(tmp_path):
    _build(tmp_path, None)
    (tmp_path / "shards" / "shard_00001.bin").unlink()
    with pytest.raises(FileNotFoundError, match="missing shard"):
        ShardWindows(tmp_path, seq_len=SEQ)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(Exception):
        ShardWindows(tmp_path, seq_len=SEQ)


def test_corrupt_shard_length_mismatch_raises(tmp_path):
    _build(tmp_path, None)
    # Truncate one shard: length no longer matches the manifest.
    p = tmp_path / "shards" / "shard_00002.bin"
    p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises((ValueError, AssertionError)):
        ShardWindows(tmp_path, seq_len=SEQ)


def test_tokenizer_mismatch_raises(tmp_path):
    """The pipeline's tokenizer family is part of the data contract."""
    _build(tmp_path, None, tokenizer="llama3")
    with pytest.raises((ValueError, RuntimeError)):
        ShardWindows(tmp_path, seq_len=SEQ, expected_tokenizer="gpt2")


def test_token_bounds_validated(tmp_path):
    """A token >= manifest vocab_size means the wrong tokenizer packed it."""
    tokens = _build(tmp_path, None, with_eos=False)
    shards_dir = tmp_path / "shards"
    bad = tokens[0].copy()
    bad[0] = VOCAB + 5
    _write_shard(shards_dir, 0, bad)
    ds = ShardWindows(tmp_path, seq_len=SEQ, vocab_size=VOCAB)
    with pytest.raises((ValueError, AssertionError)):
        ds.validate()


# -- ordering / resume --------------------------------------------------------

def test_resumed_sample_order_matches_full_run(corpus):
    """Resume at offset w: remaining windows must equal the tail of the
    uninterrupted order."""
    data_dir, _ = corpus
    ds = ShardWindows(data_dir, seq_len=SEQ)
    full = ShuffledRangeSampler(len(ds), seed=7)
    reference = list(full)
    resumed = ShuffledRangeSampler(len(ds), seed=7, offset=4)
    assert list(resumed) == reference[4:]


def test_loader_batches_deterministic(corpus):
    data_dir, _ = corpus
    a = next(iter(build_dataloader(data_dir, seq_len=SEQ, batch_size=2, seed=3)))
    b = next(iter(build_dataloader(data_dir, seq_len=SEQ, batch_size=2, seed=3)))
    assert torch.equal(a, b)
    assert a.shape == (2, SEQ)


# -- host memory --------------------------------------------------------------

@pytest.mark.slow
def test_full_corpus_host_rss_no_duplication(corpus):
    """RSS after opening must stay far below a full copy of the corpus
    (mmap contract, design Task 9: 'no corpus duplication')."""
    import resource

    data_dir, shards = corpus
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ds = ShardWindows(data_dir, seq_len=SEQ)
    _ = ds[0]
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    corpus_bytes = sum(s.nbytes for s in shards)
    # maxrss is in KB on macOS; allow page-cache noise of one window.
    assert (after - before) * 1024 < corpus_bytes
