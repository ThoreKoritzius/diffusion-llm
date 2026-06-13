"""Confidence-based iterative unmasking (MaskGIT / LLaDA style).

At every step only the currently-masked positions are predicted. The most
confident predictions are committed permanently; the rest stay masked for the
next round. Committed tokens are never resampled, so generation converges
instead of flickering.
"""
import math
from typing import Iterator, List, Optional, Tuple

import torch


def cosine_masked_count(n_total: int, step: int, total_steps: int) -> int:
    """Number of tokens that should still be masked after `step` (0-based) completes."""
    if step + 1 >= total_steps:
        return 0
    return int(math.floor(n_total * math.cos(math.pi / 2 * (step + 1) / total_steps)))


@torch.no_grad()
def denoise_steps(
    model,
    current_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    fill_positions: List[int],
    mask_id: int,
    n_steps: int = 10,
    temperature: float = 0.0,
    forbid_token_ids: Optional[List[int]] = None,
    should_stop=None,
) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Iteratively fill mask tokens at `fill_positions` in `current_ids` (shape (1, L)).

    Yields (step_idx, total_steps, current_ids) after each commit step.
    `temperature > 0` adds Gumbel noise to the commit-order confidences for
    sampling diversity; 0 is fully greedy.
    """
    device = current_ids.device
    positions = torch.as_tensor(fill_positions, dtype=torch.long, device=device)
    n_total = int(positions.numel())
    if n_total == 0:
        raise RuntimeError("denoise_steps: no positions to fill")
    total_steps = max(1, min(int(n_steps), n_total))

    forbid = set(forbid_token_ids or [])
    forbid.add(int(mask_id))

    for step_idx in range(total_steps):
        if should_stop is not None and should_stop():
            raise TimeoutError("Run cancelled or timed out")

        still_masked = current_ids[0, positions] == mask_id
        masked_pos = positions[still_masked]
        if masked_pos.numel() == 0:
            yield step_idx, total_steps, current_ids
            return

        logits = model(input_ids=current_ids, attention_mask=attention_mask).logits
        logits = logits[0, masked_pos, :].float()
        for tok_id in forbid:
            logits[:, tok_id] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = torch.log(probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
        if temperature > 0:
            u = torch.rand(conf.shape, device=device).clamp_min(1e-9)
            conf = conf + temperature * (-torch.log(-torch.log(u)))

        target_masked = cosine_masked_count(n_total, step_idx, total_steps)
        n_commit = int(masked_pos.numel()) - target_masked
        n_commit = max(1, min(n_commit, int(masked_pos.numel())))

        commit = torch.topk(conf, n_commit).indices
        current_ids[0, masked_pos[commit]] = pred[commit]

        yield step_idx, total_steps, current_ids
