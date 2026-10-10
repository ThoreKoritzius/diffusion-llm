"""One-way attention between prompt and SQL window for ModernBERT masked-diffusion models.

The SQL window attends to everything (prompt + window), but prompt tokens cannot attend to the window. Prompt hidden
states then stay identical across denoising steps, which is what allows caching them (compute the prompt once,
then run only the window per step).

Usage: call `isolate_prompt(model)` once, then before every forward set
`set_prompt_len(model, n)` with an int (batch of one) or a LongTensor [B] (per-example prompt lengths).
Requires the "sdpa" or "eager" attention implementation; flash-attention's unpadding path ignores custom masks.
"""

import torch

CONFIG_FLAG = "prompt_isolation"


def _encoder(model):
    return model.model if hasattr(model, "model") else model


def set_prompt_len(model, prompt_len):
    _encoder(model)._prompt_len = prompt_len


def isolate_prompt(model):
    enc = _encoder(model)
    if getattr(enc, "_prompt_isolated", False):
        return model
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in (None, "sdpa", "eager"):
        raise ValueError(f"prompt isolation needs sdpa/eager attention, got {impl!r}")
    original = enc._update_attention_mask
    enc._prompt_len = None

    def patched(attention_mask, *args, **kwargs):
        global_mask, sliding_mask = original(attention_mask, *args, **kwargs)
        lens = enc._prompt_len
        if lens is None:
            return global_mask, sliding_mask
        L = global_mask.shape[-1]
        pos = torch.arange(L, device=global_mask.device)
        lens = torch.as_tensor(lens, device=global_mask.device).reshape(-1, 1)  # [B or 1, 1]
        is_prompt = pos[None, :] < lens                                          # [B, L]
        blocked = is_prompt[:, :, None] & ~is_prompt[:, None, :]                 # query in prompt, key after it
        neg = torch.finfo(global_mask.dtype).min
        blocked = blocked[:, None]                                               # [B, 1, L, L]
        return global_mask.masked_fill(blocked, neg), sliding_mask.masked_fill(blocked, neg)

    enc._update_attention_mask = patched
    enc._prompt_isolated = True
    setattr(model.config, CONFIG_FLAG, True)  # saved with the checkpoint so loaders can re-apply it
    return model


def is_isolated_checkpoint(config) -> bool:
    return bool(getattr(config, CONFIG_FLAG, False))
