"""CUDA-graph decoding for ModernBERT masked-diffusion models at batch 1 (the diffusion counterpart of vLLM's /
torch.compile's CUDA graphs for AR). Sequences are padded to a multiple of `bucket` tokens so all denoising steps of
an input share one captured graph; each step copies the ids into a static buffer and replays the graph."""

import torch


def device_masks(enc):
    """Replace ModernBERT's mask builder (which creates CPU tensors, not capturable) with an equivalent device-only
    version. Honors key padding, the sliding window of local layers, and prompt isolation (`enc._prompt_len_t`)."""

    def masks(attention_mask, *args, **kwargs):
        B, L = attention_mask.shape
        dev, dtype = attention_mask.device, enc.dtype
        pos = torch.arange(L, device=dev)
        allowed = attention_mask[:, None, None, :].bool().expand(B, 1, L, L)
        lens = getattr(enc, "_prompt_len_t", None)
        if lens is not None:
            is_prompt = pos[None, :] < lens.view(-1, 1)
            allowed = allowed & ~(is_prompt[:, :, None] & ~is_prompt[:, None, :])[:, None]
        neg = torch.finfo(dtype).min
        global_mask = torch.zeros(B, 1, L, L, dtype=dtype, device=dev).masked_fill(~allowed, neg)
        window = (pos[:, None] - pos[None, :]).abs() <= enc.config.local_attention // 2
        return global_mask, global_mask.masked_fill(~window, neg)

    enc._update_attention_mask = masks


class GraphRunner:
    def __init__(self, model, isolated: bool, bucket: int = 64):
        self.model, self.isolated, self.bucket = model, isolated, bucket
        self.enc = model.model
        model.config.reference_compile = False
        device_masks(self.enc)
        self.graphs = {}

    def _capture(self, L):
        dev = next(self.model.parameters()).device
        ids = torch.zeros(1, L, dtype=torch.long, device=dev)
        attn = torch.zeros(1, L, dtype=torch.long, device=dev)
        attn[:, :8] = 1
        lens = torch.zeros(1, dtype=torch.long, device=dev)
        self.enc._prompt_len_t = lens if self.isolated else None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self.model(input_ids=ids, attention_mask=attn)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=attn).logits
        self.graphs[L] = (graph, ids, attn, lens, out)

    def prepare(self, n_tokens: int, prompt_len: int):
        L = -(-n_tokens // self.bucket) * self.bucket
        if L not in self.graphs:
            self._capture(L)
        graph, ids, attn, lens, out = self.graphs[L]
        attn.zero_()
        attn[:, :n_tokens] = 1
        lens.fill_(prompt_len)
        self._active = (graph, ids, out, n_tokens)

    def __call__(self, input_ids, attention_mask=None):
        graph, ids, out, n = self._active
        ids[:, :n].copy_(input_ids)
        graph.replay()
        return out[:, :n]
