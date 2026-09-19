"""Task 8: softcapped chunked causal cross-entropy.

Computes the same loss as the Task 5 oracle path
(`softcap(linear(hidden))[:, :-1]` vs `labels[:, 1:]`, mean over valid
tokens) without materializing the full [B, T, V] logit tensor or retaining
every chunk's autograd graph at once.
"""

import torch
from torch import nn


def _softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(logits / cap)


def chunked_causal_ce(
    hidden: torch.Tensor,
    tied_weight: nn.Parameter,
    labels: torch.Tensor,
    logit_softcap: float | None = 30.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Softcapped CE between `hidden` [B, T, D] and `labels` [B, T] via the
    tied vocabulary weight [V, D].

    Same next-token shift and softcap as inference/`Gemma4LiteModel.forward`.
    FP32 reductions throughout (linear, cap, CE). Ignored labels (the CE
    default -100) are excluded from both the sum and the count; an
    all-ignored batch raises ValueError (the caller decides between failing
    and skipping the optimizer update — never a silent NaN).

    Memory contract: chunks are processed one at a time under
    non-reentrant checkpointing, so only one chunk's logits live at a time
    and backward recomputes them; no chunk graphs are retained.
    """
    from torch.utils.checkpoint import checkpoint

    batch, seq_len, dim = hidden.shape
    vocab = tied_weight.shape[0]
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    # Shift once on the compact tensors: predictions at t, targets at t+1.
    # [B, T-1, D] view trick doesn't apply (non-contiguous is fine for matmul).
    h = hidden[:, :-1, :].reshape(-1, dim)
    y = labels[:, 1:].reshape(-1)

    def chunk_loss(start: int, end: int) -> torch.Tensor:
        logits = nn.functional.linear(h[start:end], tied_weight)
        if logit_softcap is not None:
            logits = _softcap(logits.float(), logit_softcap)
        return nn.functional.cross_entropy(
            logits.float(), y[start:end], ignore_index=-100, reduction="sum"
        )

    total = h.shape[0]
    if total == 0:
        raise ValueError("empty label sequence after next-token shift")

    # A full-materialization call is checkpoint-ineligible but identical
    # math; keep the single-chunk case on the plain path.
    if total <= chunk_size:
        return chunk_loss(0, total) / _valid_count(y)

    # Non-reentrant checkpoint recomputes each chunk's logits during
    # backward instead of retaining them; `tied_weight` is passed explicitly
    # so its gradient flows through the recomputation.
    sum_losses = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        sum_losses.append(
            checkpoint(chunk_loss, start, end, use_reentrant=False, determinism_check="none")
        )
    summed = torch.stack(sum_losses).sum()
    return summed / _valid_count(y)


def _valid_count(y: torch.Tensor) -> torch.Tensor:
    count = (y != -100).sum()
    if int(count) == 0:
        raise ValueError(
            "all labels are ignored (-100); refusing to produce a silent NaN — "
            "skip the optimizer update or fix the batch"
        )
    return count.to(torch.float32)
