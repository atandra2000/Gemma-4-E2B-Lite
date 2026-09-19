"""Greedy generation over the producer-owned cache (design §3/§4 pinned
convention: raw token IDs, no scaffolding, EOS terminates). Supports a batch
of independent requests via one row of the shared cache each, left padding
(keys masked through the cache's validity tracking), and chunked prefill.
"""

import torch

from models.cache import ProducerKVCache


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int | None = None,
    key_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily decode up to `max_new_tokens` tokens per request.

    input_ids: [B, T] prompts (left-padded rows marked in `key_valid`) or [T].
    key_valid: optional bool [B, T]; False marks padding keys.
    chunk_size: prefill chunk in tokens (None = the whole prompt at once).

    Returns (tokens [B, steps], finished [B]): rows that hit EOS carry trailing
    `eos_id` entries and True in `finished`.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if key_valid is not None:
        key_valid = key_valid.reshape(input_ids.shape).bool()
    else:
        key_valid = torch.ones_like(input_ids, dtype=torch.bool)
    batch, total = input_ids.shape
    chunk_size = chunk_size or total
    last_valid = total - 1 - key_valid.long().flip(1).argmax(1)  # per-row dense index
    cache = ProducerKVCache(model.config)
    eos = model.config.eos_id

    # -- chunked prefill: capture each row's trigger logits at its last token --
    prompt_logits = torch.zeros(batch, model.config.vocab_size)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        out = model(input_ids[:, start:stop], cache=cache, key_valid=key_valid[:, start:stop])
        hit = (last_valid >= start) & (last_valid < stop)
        if hit.any():
            offset = (last_valid[hit] - start).long()
            prompt_logits[hit] = out[hit.nonzero(as_tuple=True)[0], offset]

    # -- decode: feed every row each step; finished rows just emit EOS --
    tokens: list[torch.Tensor] = []
    finished = torch.zeros(batch, dtype=torch.bool)
    next_token = prompt_logits.argmax(-1)
    while len(tokens) < max_new_tokens:
        next_token = torch.where(finished, torch.full_like(next_token, eos), next_token)
        tokens.append(next_token)
        finished |= next_token == eos
        if finished.all():
            break
        out = model(
            next_token.unsqueeze(1),
            cache=cache,
            key_valid=torch.ones(batch, 1, dtype=torch.bool),
        )
        next_token = out[:, -1].argmax(-1)
    return torch.stack(tokens, dim=1), finished
