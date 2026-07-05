"""Verify-repair loop for diffusion text-to-SQL decoding.

Our judge analysis showed ~14% of predictions are syntactically invalid SQL —
the single largest failure bucket. This module closes part of it with a cheap,
training-free loop:

    decode -> parse with sqlglot -> if invalid, re-mask a small token window
    around the parse-error location -> re-decode just those tokens with full
    surrounding context -> re-check (up to `rounds` times).

Only invalid queries pay the extra decoding steps, so the average step cost is
~ (invalid_rate x repair_steps). Inspired by grammar-constrained diffusion
decoding (arXiv:2508.10111) but implemented as post-hoc targeted repair rather
than per-step CFG masking.
"""
from typing import List, Optional, Tuple

import torch

import sqlglot
from sqlglot.errors import ParseError

from denoising import denoise_steps


def parse_error_char(sql: str) -> Optional[int]:
    """None if `sql` parses; else the (0-based) char offset of the first error
    (falls back to end-of-string when sqlglot reports no position)."""
    if not sql.strip():
        return 0
    try:
        sqlglot.parse_one(sql)
        return None
    except ParseError as e:
        err = (e.errors or [{}])[0]
        col = err.get("col")
        return int(col) - 1 if col else len(sql) - 1
    except Exception:
        return len(sql) - 1


def is_valid(sql: str) -> bool:
    return parse_error_char(sql) is None


def _char_to_window_positions(tokenizer, text: str, win_ids: List[int],
                              pad_id: int, mask_id: int, lo: int,
                              char_pos: int, before: int = 2, after: int = 4):
    """Map a char offset in the decoded text to absolute token positions in the
    generation window; returns a small span around it (plus trailing pads when
    the error is at the end — truncation repairs may need to extend)."""
    nonpad = [i for i, t in enumerate(win_ids) if t not in (pad_id, mask_id)]
    if not nonpad:
        return [lo + i for i in range(min(6, len(win_ids)))]
    gen_tokens = [win_ids[i] for i in nonpad]

    j = None
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) == gen_tokens:
        for k, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= char_pos < max(e, s + 1):
                j = k
                break
    if j is None:  # round-trip mismatch -> fractional fallback
        frac = min(max(char_pos / max(len(text), 1), 0.0), 1.0)
        j = int(round(frac * (len(nonpad) - 1)))

    span = range(max(0, j - before), min(len(nonpad), j + after))
    positions = [lo + nonpad[k] for k in span]
    # error at/near the end: also open a few trailing pad slots so the model
    # can extend a truncated query
    if char_pos >= len(text) - 2:
        last = nonpad[-1]
        trailing = [i for i in range(last + 1, len(win_ids))
                    if win_ids[i] == pad_id][:3]
        positions += [lo + i for i in trailing]
    return sorted(set(positions))


@torch.no_grad()
def verify_and_repair(model, tokenizer, ids: torch.Tensor, attn: torch.Tensor,
                      lo: int, hi: int, mask_id: int, pad_id: int,
                      forbid_token_ids: List[int], conf_stop: Optional[float],
                      rounds: int = 2, steps: int = 6) -> Tuple[str, int, bool]:
    """Returns (final_sql, extra_steps_used, final_is_valid). `ids` is the
    (1, L) tensor after normal decoding; modified in place on repair."""

    def decode() -> str:
        out = [t for t in ids[0, lo:hi].tolist() if t not in (pad_id, mask_id)]
        return tokenizer.decode(out, skip_special_tokens=True)

    text = decode()
    extra = 0
    for _ in range(rounds):
        err = parse_error_char(text)
        if err is None:
            return text, extra, True
        win_ids = ids[0, lo:hi].tolist()
        remask = _char_to_window_positions(tokenizer, text, win_ids, pad_id,
                                           mask_id, lo, err)
        if not remask:
            break
        ids[0, torch.as_tensor(remask, dtype=torch.long, device=ids.device)] = mask_id
        for _ in denoise_steps(model, ids, attn, list(range(lo, hi)), mask_id,
                               n_steps=steps, forbid_token_ids=forbid_token_ids,
                               confidence_stop=conf_stop):
            extra += 1
            if (ids[0, lo:hi] == mask_id).sum().item() == 0:
                break
        text = decode()
    return text, extra, parse_error_char(text) is None
