"""Prompt KV cache for prompt-isolated ModernBERT masked-diffusion models.

With prompt isolation (see prompt_isolation.py) prompt tokens never attend to the SQL window, so their keys and values
are identical at every denoising step. `PromptCache.prefill` runs the prompt once and stores per-layer keys/values;
`PromptCache.step` then runs only the window tokens (SQL window + closing tags), attending to the cached prompt plus
themselves. Outputs match a full isolated forward pass exactly (up to float rounding).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


class PromptCache:
    def __init__(self, mlm_model):
        self.mlm = mlm_model
        self.enc = mlm_model.model
        cfg = mlm_model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.prompt_len = 0

    def _attend(self, layer, h, pos, past=None):
        """One attention block for hidden states `h` at absolute positions `pos`; `past` = cached (k, v) to prepend.
        Returns (attention output, (k, v) of these tokens)."""
        attn = layer.attn
        B, T, _ = h.shape
        qkv = attn.Wqkv(layer.attn_norm(h)).view(B, T, 3, self.n_heads, self.head_dim)
        cos, sin = attn.rotary_emb(qkv, position_ids=pos[None])
        q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # [B, H, T, D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        own = (k, v)
        key_pos = pos
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
            key_pos = torch.cat([torch.arange(past[0].shape[2], device=pos.device), pos])
        mask = None
        if attn.local_attention != (-1, -1):  # sliding-window layer: |query - key| <= window / 2
            mask = (pos[:, None] - key_pos[None, :]).abs() <= attn.local_attention[0]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return attn.Wo(out), own

    def prefill_kv(self, prompt_ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """prompt_ids: [1, P] (everything up to and including <SQL>) -> per-layer (k, v)."""
        h = self.enc.embeddings(input_ids=prompt_ids)
        pos = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
        kv = []
        for layer in self.enc.layers:
            a, layer_kv = self._attend(layer, h, pos)
            h = h + a
            h = h + layer.mlp(layer.mlp_norm(h))
            kv.append(layer_kv)
        return kv

    def step_kv(self, window_ids: torch.Tensor, kv) -> torch.Tensor:
        """window_ids: [1, W] tokens after the prompt, kv: prefill_kv output. Returns MLM logits [1, W, V]."""
        h = self.enc.embeddings(input_ids=window_ids)
        prompt_len = kv[0][0].shape[2]
        pos = torch.arange(window_ids.shape[1], device=window_ids.device) + prompt_len
        for layer, past in zip(self.enc.layers, kv):
            a, _ = self._attend(layer, h, pos, past)
            h = h + a
            h = h + layer.mlp(layer.mlp_norm(h))
        h = self.enc.final_norm(h)
        return self.mlm.decoder(self.mlm.head(h))

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor):
        self.kv = self.prefill_kv(prompt_ids)
        self.prompt_len = prompt_ids.shape[1]

    @torch.no_grad()
    def step(self, window_ids: torch.Tensor) -> torch.Tensor:
        return self.step_kv(window_ids, self.kv)


class PrefillGraph(torch.nn.Module):
    """ONNX export wrapper: prompt_ids -> k0, v0, k1, v1, ..."""

    def __init__(self, cache: PromptCache):
        super().__init__()
        self.cache = cache
        self.layers = cache.enc  # register parameters

    def forward(self, prompt_ids):
        return tuple(t for kv in self.cache.prefill_kv(prompt_ids) for t in kv)


class StepGraph(torch.nn.Module):
    """ONNX export wrapper: (window_ids, k0, v0, k1, v1, ...) -> logits."""

    def __init__(self, cache: PromptCache):
        super().__init__()
        self.cache = cache
        self.layers = cache.mlm

    def forward(self, window_ids, *flat_kv):
        kv = [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]
        return self.cache.step_kv(window_ids, kv)
