"""Producer-owned request-local KV cache and cache-aware masking (design §4
"Sharing map and cache invariants").

Every prefix producer layer appends persistent state once per input token,
keyed by its own layer index; consumers alias the stream of their producer
(config.kv_producer). Consumers keep nothing persistent. After a call's last
consumer finishes, `finish_step()` commits the current-call states to history
and trims each LOCAL producer stream to the window horizon; global streams are
never trimmed, so the canonical global stream's length is the absolute token
count. Caches are request-local, resettable, never module-global.
"""

import torch

from models.config import ModelConfig

# Additive-mask filler. finfo.min (not -inf) so an all-masked row -- a padding
# query with no valid keys -- softmaxes to uniform garbage instead of NaN;
# NaN keys would poison valid queries through 0 * NaN in later layers.
MASK_FILLER = torch.finfo(torch.float32).min


class ProducerKVCache:
    """Per-request cache: one K/V stream (plus key validity) per producer layer."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._history: dict[int, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._valid: dict[int, torch.Tensor | None] = {}
        self._inflight: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per-row count of valid keys dropped by local trimming: the offset
        # that turns a trimmed validity vector back into absolute positions.
        self._dropped: dict[int, torch.Tensor] = {}
        # Untrimmed global stream: its length is the absolute token count.
        self._global_stream = next(
            i for i in range(config.share_boundary) if config.layer_type(i) == "global"
        )

    def stream_for(self, layer_idx: int) -> int:
        """The producer layer whose K/V stream this layer reads."""
        producer = self.config.kv_producer(layer_idx)
        return layer_idx if producer is None else producer

    # -- producer side -------------------------------------------------------

    def producer_append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this producer's chunk K/V [B, 1, C, hd] once; return the full
        current-call states (history + chunk) for the layer's own queries."""
        hist = self._history.get(layer_idx)
        if hist is None:
            full = (k, v)
            valid = (
                None
                if key_valid is None
                else key_valid.reshape(key_valid.shape[0], 1, -1, 1).to(k.device)
            )
        else:
            full = (torch.cat([hist[0], k], dim=2), torch.cat([hist[1], v], dim=2))
            # Keep the validity vector in sync with the full current-call
            # states even on padding-free decode steps.
            valid = self._valid[layer_idx]
            if valid is not None:
                if key_valid is not None:
                    chunk = key_valid.reshape(key_valid.shape[0], 1, -1, 1).to(k.device)
                else:
                    chunk = torch.ones(
                        k.shape[0], 1, k.shape[2], 1,
                        dtype=valid.dtype, device=k.device,
                    )
                valid = torch.cat([valid, chunk], dim=2)
        self._inflight[layer_idx] = full
        self._valid[layer_idx] = valid
        return full

    # -- consumer side -------------------------------------------------------

    def consumer_states(self, producer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inflight[producer_idx]

    def attention_mask(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        key_valid: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Additive mask [B, 1, q_len, kv_len] over per-row token positions.

        Position of a token = its row's count of valid tokens before it (the
        cumulative-sum convention), so a left-padded row's valid tokens sit at
        the same positions they would occupy unpadded. Key positions come from
        the stream's tracked validity plus the trim offset; query positions
        from `position_ids`. Keys are kept when key <= query, within the local
        window, and valid; padding keys are excluded through the same keep set
        (MASK_FILLER keeps all-masked padding-query rows finite)."""
        stream = self.stream_for(layer_idx)
        window = (
            self.config.local_window
            if self.config.layer_type(layer_idx) == "local"
            else None
        )
        q_len = position_ids.shape[1]
        kv_len = self.history_length(stream) + q_len
        valid = self._valid.get(stream)
        if valid is None or valid.shape[2] < kv_len:
            # A producer's own mask is built before it appends, so the stored
            # validity covers only the committed history; pad it with the
            # current chunk (key_valid marks padding keys, otherwise the chunk
            # is all valid). Consumers see the already-appended full vector.
            if key_valid is not None:
                chunk = key_valid.reshape(key_valid.shape[0], 1, -1, 1).to(device).bool()
            elif valid is not None:  # padding-free decode step
                chunk = torch.ones(
                    valid.shape[0], 1, kv_len - valid.shape[2], 1,
                    dtype=torch.bool, device=device,
                )
            else:
                chunk = None
            if chunk is not None:
                valid = torch.cat([valid, chunk], dim=2) if valid is not None else chunk
        batch = position_ids.shape[0]
        if valid is not None:
            valid_flat = valid.reshape(valid.shape[0], -1)  # [B, kv]
            dropped = self._dropped.get(stream)
            dropped = (
                torch.zeros(batch, dtype=torch.long, device=device)
                if dropped is None
                else dropped.to(device)
            )
            k_positions = dropped[:, None] + (
                valid_flat.cumsum(1) - 1
            ).clamp(min=0)  # [B, kv]
        else:  # no padding ever: valid count == dense slot index
            k_positions = torch.arange(kv_len, device=device).expand(batch, -1)
            dropped = self._dropped.get(stream)
            if dropped is not None:
                k_positions = k_positions + dropped[:, None]
        keep = k_positions[:, None, :] <= position_ids[:, :, None]  # [B, Q, kv]
        if window is not None:
            keep = keep & (
                position_ids[:, :, None] - k_positions[:, None, :] < window
            )
        if valid is not None:
            keep = keep & valid_flat[:, None, :]
        return torch.zeros(
            batch, 1, q_len, kv_len, device=device
        ).masked_fill(~keep, MASK_FILLER)

    # -- lifecycle -----------------------------------------------------------

    def finish_step(self) -> None:
        """Commit current-call states once every consumer has finished: drop the
        inflight views and trim each local stream to the window horizon (the
        newest `window - 1` keys are all a future query can still attend).
        Global streams are never trimmed."""
        for layer_idx, full in list(self._inflight.items()):
            del self._inflight[layer_idx]
            if self.config.layer_type(layer_idx) == "local":
                keep = self.config.local_window - 1
                if full[0].shape[2] > keep:
                    n_drop = full[0].shape[2] - keep
                    full = (full[0][..., -keep:, :], full[1][..., -keep:, :])
                    valid = self._valid[layer_idx]
                    dropped = (
                        valid[:, :, :n_drop].reshape(valid.shape[0], -1).sum(1)
                        if valid is not None
                        else torch.full(
                            (full[0].shape[0],), n_drop, dtype=torch.long
                        )
                    )
                    prev = self._dropped.get(layer_idx)
                    self._dropped[layer_idx] = dropped if prev is None else prev + dropped
                    if valid is not None:
                        self._valid[layer_idx] = valid[..., -keep:, :]
            self._history[layer_idx] = full

    def valid_counts(self, batch: int) -> torch.Tensor:
        """Per-row count of valid (non-padding) tokens ever appended — each
        row's next position, from the untrimmed global stream's validity."""
        valid = self._valid.get(self._global_stream)
        if valid is None:
            return torch.zeros(batch, dtype=torch.long)
        return valid.reshape(valid.shape[0], -1).sum(1)

    def history_length(self, layer: int | str) -> int:
        """Committed history length for a producer layer index or layer-type
        name ("global"/"local"; all local streams keep the same length)."""
        if isinstance(layer, str):
            layer = self._global_stream if layer == "global" else next(
                i for i in range(self.config.share_boundary)
                if self.config.layer_type(i) == "local"
            )
        hist = self._history.get(layer)
        return 0 if hist is None else hist[0].shape[2]

    @property
    def num_tokens(self) -> int:
        """Absolute token count processed so far (untrimmed global history)."""
        return self.history_length(self._global_stream)

    def unique_storages(self) -> int:
        """Unique tensor storages persistently held by the cache (for the
        no-consumer-copies accounting)."""
        ids = set()
        for hist in self._history.values():
            if hist is not None:
                ids |= {t.untyped_storage().data_ptr() for t in hist}
        for valid in self._valid.values():
            if valid is not None:
                ids.add(valid.untyped_storage().data_ptr())
        return len(ids)

    def reset(self) -> None:
        self._history = {}
        self._valid = {}
        self._inflight = {}
        self._dropped = {}
