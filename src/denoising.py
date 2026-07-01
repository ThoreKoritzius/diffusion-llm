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


def _zscore(x: torch.Tensor) -> torch.Tensor:
    """Standardize a 1-D score vector; returns zeros if it has no spread."""
    if x.numel() <= 1:
        return torch.zeros_like(x)
    sd = x.std()
    if not torch.isfinite(sd) or float(sd) < 1e-6:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


def _forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, want_attn: bool):
    """Run the model, optionally requesting attentions.

    Returns (logits, attentions_or_None). Backends that cannot emit attention
    matrices (e.g. the ONNX runner, whose ``__call__`` only accepts
    ``input_ids``/``attention_mask``) transparently degrade to ``None``.
    """
    if want_attn:
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        except TypeError:
            return model(input_ids=input_ids, attention_mask=attention_mask).logits, None
        return out.logits, getattr(out, "attentions", None)
    return model(input_ids=input_ids, attention_mask=attention_mask).logits, None


def _dependency_scores(
    attentions, masked_pos: torch.Tensor, revealed: torch.Tensor, dep_layers: int,
    dep_layer_index: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """DOS-style dependency score for each masked position.

    For a masked query position i, the score is the fraction of its attention
    mass that lands on already-revealed (unmasked, in-window) positions,
    averaged over heads. A high score means the token is well-grounded in known
    context, so committing it is less likely to be revised by later reveals.
    Attention rows sum to 1, so the score is naturally in [0, 1] and needs no
    extra normalization.

    Layer selection follows the DOS paper, which reads a single (typically
    shallow, syntax-bearing) layer tuned per task. If `dep_layer_index` is given
    it selects exactly that layer (0 = first/shallowest, negatives index from
    the end). Otherwise the score is averaged over the last `dep_layers` layers.
    """
    if not attentions:
        return None
    if dep_layer_index is not None:
        layers = [attentions[dep_layer_index]]
    else:
        layers = attentions[-max(1, dep_layers):]
    revealed_f = revealed.float()
    acc = None
    for layer in layers:
        # layer: (1, H, L, L) -> rows for masked queries, averaged over heads
        a = layer[0][:, masked_pos, :].mean(0).float()  # (n_masked, L)
        dep = (a * revealed_f.unsqueeze(0)).sum(-1)      # (n_masked,)
        acc = dep if acc is None else acc + dep
    return acc / float(len(layers))


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
    should_stop=None,
    confidence_stop: Optional[float] = None,
    on_step_stats=None,
    strategy: str = "confidence",
    dep_alpha: float = 0.5,
    dep_layers: int = 1,
    dep_layer_index: Optional[int] = None,
) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Iteratively fill mask tokens at `fill_positions` in `current_ids` (shape (1, L)).

    Yields (step_idx, total_steps, current_ids) after each commit step.
    `temperature > 0` adds Gumbel noise to the commit-order score for
    sampling diversity; 0 is fully greedy.

    `strategy` selects how the commit *order* is chosen each step:
      - "confidence" (default): commit the most confident predictions first
        (MaskGIT / LLaDA behaviour, unchanged).
      - "dependency": DOS-style ordering. Commit the masked tokens whose
        attention is most grounded in already-revealed context first, blended
        with confidence by `dep_alpha` (0 = pure confidence, 1 = pure
        dependency). Requires a backend that emits attentions and a model
        loaded with eager attention; otherwise it falls back to confidence
        ordering. `dep_layers` averages the dependency signal over the last N
        transformer layers (1 = last layer only).

    The commit *count* per step (cosine schedule) and the `confidence_stop`
    early-stop criterion always use calibrated confidence, independent of
    `strategy`, so adaptive stopping stays correctly calibrated.

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

    want_attn = strategy == "dependency" and dep_alpha > 0.0

    for step_idx in range(total_steps):
        if should_stop is not None and should_stop():
            raise TimeoutError("Run cancelled or timed out")

        still_masked = current_ids[0, positions] == mask_id
        masked_pos = positions[still_masked]
        if masked_pos.numel() == 0:
            yield step_idx, total_steps, current_ids
            return

        full_logits, attentions = _forward(model, current_ids, attention_mask, want_attn)
        logits = full_logits[0, masked_pos, :].float()
        for tok_id in forbid:
            logits[:, tok_id] = -float("inf")

        pred = logits.argmax(dim=-1)
        chosen_logits = logits.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        # Calibrated log-prob of the chosen token; the canonical confidence
        # used for the early-stop threshold and telemetry (noise-free).
        clean_conf = chosen_logits - torch.logsumexp(logits, dim=-1)

        # Commit-ordering score. Confidence by default; for the dependency
        # strategy, blend in how grounded each token is in revealed context.
        dep = None
        if want_attn:
            revealed = (current_ids[0] != mask_id) & (attention_mask[0] == 1)
            dep = _dependency_scores(attentions, masked_pos, revealed, dep_layers, dep_layer_index)
        if dep is not None:
            order_score = (1.0 - dep_alpha) * _zscore(clean_conf) + dep_alpha * _zscore(dep)
        else:
            order_score = clean_conf.clone()

        if temperature > 0:
            u = torch.rand(order_score.shape, device=device).clamp_min(1e-9)
            order_score = order_score + temperature * (-torch.log(-torch.log(u)))

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

        commit = torch.topk(order_score, n_commit).indices
        current_ids[0, masked_pos[commit]] = pred[commit]

        # Per-step confidence telemetry (probabilities) for visualisation.
        if on_step_stats is not None:
            cc = clean_conf
            stat = {
                "step": step_idx + 1,
                "masked": int(masked_pos.numel()),
                "commit": int(n_commit),
                "min_p": float(torch.exp(cc.min())),
                "mean_p": float(torch.exp(cc.mean())),
                "commit_p": float(torch.exp(cc[commit].mean())),
            }
            if dep is not None:
                stat["mean_dep"] = float(dep.mean())
                stat["commit_dep"] = float(dep[commit].mean())
            on_step_stats(stat)

        yield step_idx, total_steps, current_ids
