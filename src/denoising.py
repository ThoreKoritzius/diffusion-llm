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


@torch.inference_mode()
def denoise_steps(
    model,
    current_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    fill_positions: List[int],
    mask_id: int,
    n_steps: int = 10,
    temperature: float = 0.0,
    forbid_token_ids: Optional[List[int]] = None,
    bias_token_ids: Optional[dict] = None,
    should_stop=None,
    confidence_stop: Optional[float] = None,
    on_step_stats=None,
) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Iteratively fill mask tokens at `fill_positions` in `current_ids` (shape (1, L)).

    Yields (step_idx, total_steps, current_ids) after each commit step.
    `temperature > 0` adds Gumbel noise to the commit-order confidences for
    sampling diversity; 0 is fully greedy.

    `confidence_stop` (a probability in (0, 1], e.g. 0.9) enables adaptive early
    stopping: on any step where *every* still-masked position already predicts
    its top token with probability >= the threshold, all remaining positions are
    committed at once and generation finishes. Easy inputs converge in a couple
    of forward passes; hard ones still use the full `n_steps` budget. `None`
    disables it (fixed-step behaviour). Only applies when temperature == 0
    (greedy), since the threshold compares calibrated log-probabilities.
    """
    conf_stop_logp = None
    if confidence_stop is not None and temperature == 0.0:
        p = float(confidence_stop)
        if 0.0 < p < 1.0:
            conf_stop_logp = math.log(p)
    device = current_ids.device
    positions = torch.as_tensor(fill_positions, dtype=torch.long, device=device)
    n_total = int(positions.numel())
    if n_total == 0:
        raise RuntimeError("denoise_steps: no positions to fill")
    total_steps = max(1, min(int(n_steps), n_total))

    forbid = set(forbid_token_ids or [])
    forbid.add(int(mask_id))
    token_bias = {int(k): float(v) for k, v in (bias_token_ids or {}).items()}

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
        for tok_id, bias in token_bias.items():
            if tok_id not in forbid:
                logits[:, tok_id] = logits[:, tok_id] + bias

        pred = logits.argmax(dim=-1)
        chosen_logits = logits.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        conf = chosen_logits - torch.logsumexp(logits, dim=-1)

        # Confidence (calibrated log-prob) is captured before any Gumbel noise so
        # the early-stop threshold compares against real probabilities.
        clean_conf = conf
        if temperature > 0:
            u = torch.rand(conf.shape, device=device).clamp_min(1e-9)
            conf = conf + temperature * (-torch.log(-torch.log(u)))

        target_masked = cosine_masked_count(n_total, step_idx, total_steps)
        n_commit = int(masked_pos.numel()) - target_masked
        n_commit = max(1, min(n_commit, int(masked_pos.numel())))

        # Adaptive early stop: also commit every position already above the
        # confidence threshold, so easy inputs empty the window in fewer passes
        # (the loop returns once nothing is left to fill). Hard tokens stay on the
        # gradual cosine schedule.
        if conf_stop_logp is not None:
            n_conf = int((clean_conf >= conf_stop_logp).sum())
            n_commit = max(n_commit, min(n_conf, int(masked_pos.numel())))

        commit = torch.topk(conf, n_commit).indices
        current_ids[0, masked_pos[commit]] = pred[commit]

        # Per-step confidence telemetry (probabilities) for visualisation.
        if on_step_stats is not None:
            cc = clean_conf
            on_step_stats({
                "step": step_idx + 1,
                "masked": int(masked_pos.numel()),
                "commit": int(n_commit),
                "min_p": float(torch.exp(cc.min())),
                "mean_p": float(torch.exp(cc.mean())),
                "commit_p": float(torch.exp(cc[commit].mean())),
            })

        yield step_idx, total_steps, current_ids
