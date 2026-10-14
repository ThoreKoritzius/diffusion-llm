"""Autoregressive SQL generation for the playground: Qwen token by token, or Qwen with a DFlash drafter.

DFlash (block diffusion) drafts a block of tokens in one forward pass, conditioned on three of the verifier's hidden
states; the verifier checks the whole block in one pass and keeps the longest prefix it agrees with, plus one token of
its own. With greedy decoding the output is the verifier's own greedy output (lossless up to floating-point ties).

The drafter is the Speculators 0.8.1 DFlash checkpoint (`speculators.models.dflash`), re-implemented here for plain
PyTorch inference on CPU (Speculators itself depends on vLLM):
  context   fc(concat of the verifier's hidden states at aux_hidden_state_layer_ids) -> RMSNorm, one row per earlier
            position; injected as extra keys/values in every draft layer
  block     [last accepted token, mask x (block_size - 1)] at the next positions; causal within the block
  output    slot i > 0 predicts the token at anchor + i over a reduced vocabulary (draft id + d2t[id] = target id)

    eng = QwenEngine("checkpoints/sql-ar-qwen25-coder-0.5b-ar", drafter_dir="checkpoints/spec-qwen25/dflash")
    for ev in eng.generate(prompt, context, speculative=True): ...
"""

import json
import os
import time
from typing import Callable, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class DraftLayer(nn.Module):
    """Qwen3 decoder layer whose attention also sees the projected verifier context (Qwen3DFlashDecoderLayer)."""

    def __init__(self, c: dict):
        super().__init__()
        h, nh, nkv, hd, eps = (c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
                               c["rms_norm_eps"])
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.q_proj, self.o_proj = nn.Linear(h, nh * hd, bias=False), nn.Linear(nh * hd, h, bias=False)
        self.k_proj, self.v_proj = nn.Linear(h, nkv * hd, bias=False), nn.Linear(h, nkv * hd, bias=False)
        self.q_norm, self.k_norm = RMSNorm(hd, eps), RMSNorm(hd, eps)
        self.gate_proj = nn.Linear(h, c["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(h, c["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(c["intermediate_size"], h, bias=False)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(h, eps), RMSNorm(h, eps)

    def forward(self, x, ctx, cos, sin, mask):
        b, q_len, _ = x.shape
        y = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(y).view(b, q_len, self.nh, self.hd)).transpose(1, 2)
        kv_in = torch.cat([ctx, y], dim=1)
        k = self.k_norm(self.k_proj(kv_in).view(b, -1, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, -1, self.nkv, self.hd).transpose(1, 2)
        q = q * cos[:, None, -q_len:] + rotate_half(q) * sin[:, None, -q_len:]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, q_len, -1))
        y = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))


class DFlashDrafter(nn.Module):
    def __init__(self, drafter_dir: str, embed_tokens: nn.Embedding):
        super().__init__()
        cfg = json.load(open(os.path.join(drafter_dir, "config.json")))
        c = cfg["transformer_layer_config"]
        self.block_size, self.mask_token_id = cfg["block_size"], cfg["mask_token_id"]
        self.layer_ids = cfg["aux_hidden_state_layer_ids"]
        h, eps = c["hidden_size"], c["rms_norm_eps"]
        self.layers = nn.ModuleList([DraftLayer(c) for _ in range(c["num_hidden_layers"])])
        self.fc = nn.Linear(len(self.layer_ids) * h, h, bias=False)
        self.hidden_norm, self.norm = RMSNorm(h, eps), RMSNorm(h, eps)
        self.lm_head = nn.Linear(h, cfg["draft_vocab_size"], bias=False)
        self.embed_tokens = embed_tokens  # shared with the verifier, not stored in the checkpoint
        theta = (c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", 1e6))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, c["head_dim"], 2, dtype=torch.float32) / c["head_dim"]))

        state = load_file(os.path.join(drafter_dir, "model.safetensors"))
        self.d2t = state.pop("d2t")
        state.pop("t2d", None)
        renamed = {}
        for k, v in state.items():  # flatten self_attn.* / mlp.* into DraftLayer attributes
            renamed[k.replace("self_attn.", "").replace("mlp.", "")] = v
        missing, unexpected = self.load_state_dict(renamed, strict=False)
        missing = [m for m in missing if not m.startswith("embed_tokens")]
        if missing or unexpected:
            raise ValueError(f"DFlash checkpoint mismatch: missing {missing}, unexpected {unexpected}")

    def rope(self, positions: torch.Tensor, dtype):
        freqs = positions[:, None].float() * self.inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None].to(dtype), emb.sin()[None].to(dtype)

    @torch.no_grad()
    def logits(self, aux_hidden: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """aux_hidden: [1, p, len(layer_ids) * hidden] for positions 0..p-1; blocks: [b, size] token ids for positions
        p.. (the anchor, then masks or fixed tokens). Returns draft-vocab logits [b, size - 1, draft_vocab]; row i
        predicts the token at anchor + 1 + i."""
        p = aux_hidden.shape[1]
        b, size = blocks.shape
        ctx = self.hidden_norm(self.fc(aux_hidden)).expand(b, -1, -1)
        x = self.embed_tokens(blocks).to(ctx.dtype)
        cos, sin = self.rope(torch.arange(p + size), ctx.dtype)
        # every block slot sees the whole context; within the block, causal (sliding_window_non_causal=False)
        mask = torch.ones(size, p + size, dtype=torch.bool)
        mask[:, p:] = torch.tril(torch.ones(size, size, dtype=torch.bool))
        for layer in self.layers:
            x = layer(x, ctx, cos, sin, mask)
        return self.lm_head(self.norm(x[:, 1:]))

    def to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]

    def block(self, anchor_token: int, n: int, fixed: Optional[List[int]] = None) -> torch.Tensor:
        ids = torch.full((1, min(self.block_size, n + 1)), self.mask_token_id, dtype=torch.long)
        ids[0, 0] = anchor_token
        for i, t in enumerate(fixed or []):
            ids[0, 1 + i] = t
        return ids

    @torch.no_grad()
    def draft(self, aux_hidden: torch.Tensor, anchor_token: int, n: int) -> List[int]:
        """Greedy block: one forward pass, the anchor followed by masks."""
        out = self.logits(aux_hidden, self.block(anchor_token, n))
        return self.to_target(out.argmax(-1)[0]).tolist()

    @torch.no_grad()
    def draft_candidates(self, aux_hidden: torch.Tensor, anchor_token: int, n: int, k: int) -> List[List[int]]:
        """k blocks that differ in the first drafted token: the greedy block, plus blocks starting with the drafter's
        2nd..k-th choice for that token, each re-drafted with that token fixed so the rest of the block follows it."""
        out = self.logits(aux_hidden, self.block(anchor_token, n))
        greedy = self.to_target(out.argmax(-1)[0]).tolist()
        if k <= 1 or n <= 1:
            return [greedy]
        firsts = self.to_target(out[0, 0].topk(k).indices).tolist()[1:]
        blocks = torch.cat([self.block(anchor_token, n, fixed=[t]) for t in firsts])
        rest = self.to_target(self.logits(aux_hidden, blocks)[:, 1:].argmax(-1)).tolist()
        return [greedy] + [[t] + r for t, r in zip(firsts, rest)]


class QwenEngine:
    def __init__(self, model_dir: str, drafter_dir: Optional[str] = None, dtype: torch.dtype = torch.float32,
                 threads: Optional[int] = None):
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()
        self.tid = {t: self.tok.convert_tokens_to_ids(t) for t in TAGS}
        self.stop_ids = {self.tid["</SQL>"], self.tok.eos_token_id}
        self.drafter = None
        if drafter_dir:
            self.drafter = DFlashDrafter(drafter_dir, self.model.get_input_embeddings()).to(dtype).eval()

    def prompt_ids(self, prompt: str, context: str) -> List[int]:
        """Same layout as src/train_ar.py and bench/backends.py ARBackend."""
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c = self.tok(context, add_special_tokens=False)["input_ids"]
        budget = 512 - 128 - 12
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        t = self.tid
        return bos + [t["<PROMPT>"]] + p + [t["</PROMPT>"], t["<CONTEXT>"]] + c + [t["</CONTEXT>"], t["<SQL>"]]

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode([i for i in ids if i not in self.stop_ids], skip_special_tokens=True).strip()

    def _forward(self, ids: List[int], cache: DynamicCache, start: int, hidden: bool):
        pos = torch.arange(start, start + len(ids))[None]
        return self.model(torch.tensor([ids]), past_key_values=cache, position_ids=pos, use_cache=True,
                          output_hidden_states=hidden)

    def _aux(self, out) -> torch.Tensor:
        # vLLM's aux layer L = the residual stream after L decoder layers = Transformers' hidden_states[L]
        return torch.cat([out.hidden_states[i] for i in self.drafter.layer_ids], dim=-1)

    @torch.no_grad()
    def generate(self, prompt: str, context: str, speculative: bool = False, max_new_tokens: int = 128,
                 num_speculative_tokens: int = 15, candidates: int = 1, should_stop: Optional[Callable[[], bool]] = None
                 ) -> Iterator[Dict]:
        """Yield one event per verifier forward pass:
        {"tokens": all generated ids, "new": ids added this step, "drafted": ids proposed (speculative only),
         "accepted": how many drafted ids were kept, "p_slots": the verifier's top probability at each position it
         checked this pass (len(drafted) + 1), "forward": verifier passes so far, "elapsed": seconds}.
        candidates > 1 drafts that many blocks per pass (different first tokens, see DFlashDrafter.draft_candidates)
        and verifies them as one batch, keeping the block with the longest accepted prefix; still lossless."""
        if speculative and self.drafter is None:
            raise ValueError("no drafter loaded")
        t0 = time.perf_counter()
        ids = self.prompt_ids(prompt, context)
        cache = DynamicCache()
        out = self._forward(ids, cache, 0, hidden=speculative)
        aux = self._aux(out) if speculative else None
        probs = out.logits[0, -1:].float().softmax(-1).max(-1)
        gen = [int(probs.indices[0])]
        forward = 1
        yield {"tokens": list(gen), "new": list(gen), "drafted": [], "accepted": 0, "p_slots": probs.values.tolist(),
               "forward": forward, "elapsed": time.perf_counter() - t0}
        while gen[-1] not in self.stop_ids and len(gen) < max_new_tokens:
            if should_stop and should_stop():
                return
            pos = len(ids) + len(gen) - 1  # position of the newest token (not yet fed to the verifier)
            if not speculative:
                out = self._forward([gen[-1]], cache, pos, hidden=False)
                probs = out.logits[0].float().softmax(-1).max(-1)
                new, drafted, accepted = [int(probs.indices[-1])], [], 0
            else:
                n = min(num_speculative_tokens, max_new_tokens - len(gen))
                blocks = [d[:n] for d in self.drafter.draft_candidates(aux, gen[-1], n, candidates)]
                if len(blocks) > 1:
                    cache.batch_repeat_interleave(len(blocks))
                pos_ids = torch.arange(pos, pos + 1 + n)[None].expand(len(blocks), -1)
                out = self.model(torch.tensor([[gen[-1]] + d for d in blocks]), past_key_values=cache,
                                 position_ids=pos_ids, use_cache=True, output_hidden_states=True)
                top = out.logits.float().softmax(-1).max(-1)
                best, accepted = 0, -1
                for r, d in enumerate(blocks):  # longest prefix the verifier agrees with; ties keep the greedy block
                    a = 0
                    while a < len(d) and d[a] == int(top.indices[r, a]):
                        a += 1
                    if a > accepted:
                        best, accepted = r, a
                if len(blocks) > 1:
                    cache.batch_select_indices(torch.tensor([best]))
                drafted = blocks[best]
                probs = torch.return_types.max((top.values[best], top.indices[best]))
                greedy = probs.indices.tolist()  # greedy[i] = verifier's token after block slot i
                new = drafted[:accepted] + [greedy[accepted]]
                cache.crop(pos + 1 + accepted)  # keep the anchor and the accepted drafts
                hs = [h[best: best + 1] for h in out.hidden_states]
                aux = torch.cat([aux, torch.cat([hs[i] for i in self.drafter.layer_ids], -1)[:, : 1 + accepted]], 1)
            forward += 1
            for i, t in enumerate(new):  # stop at the first end token inside an accepted run
                if t in self.stop_ids:
                    new = new[: i + 1]
                    break
            gen += new
            yield {"tokens": list(gen), "new": new, "drafted": drafted, "accepted": accepted,
                   "p_slots": probs.values.tolist(), "forward": forward, "elapsed": time.perf_counter() - t0}
