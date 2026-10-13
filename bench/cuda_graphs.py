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
    """compiled=False: manually captured CUDA graph per padded length. compiled=True: torch.compile "reduce-overhead"
    (inductor kernel fusion + CUDA graphs), the same tooling ARGraphDecoder uses; all lengths are pre-compiled."""

    def __init__(self, model, isolated: bool, bucket: int = 64, compiled: bool = False, max_len: int = 512):
        self.model, self.isolated, self.bucket, self.compiled = model, isolated, bucket, compiled
        self.enc = model.model
        model.config.reference_compile = False
        device_masks(self.enc)
        self.graphs = {}
        if compiled:
            dev = next(model.parameters()).device
            self.fn = torch.compile(self._fwd, mode="reduce-overhead", dynamic=False)
            for L in range(bucket, max_len + 1, bucket):  # compile every padded length up front (not timed)
                ids = torch.zeros(1, L, dtype=torch.long, device=dev)
                attn = torch.ones(1, L, dtype=torch.long, device=dev)
                lens = torch.full((1,), 8, dtype=torch.long, device=dev)
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    self.fn(ids, attn, lens)
                self.graphs[L] = (None, ids, attn, lens, None)

    def _fwd(self, ids, attn, lens):
        self.enc._prompt_len_t = lens if self.isolated else None
        return self.model(input_ids=ids, attention_mask=attn).logits

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
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
            _, _, attn, lens, _ = self.graphs[ids.shape[1]]
            return self.fn(ids, attn, lens)[:, :n].clone()
        graph.replay()
        return out[:, :n]


class ARGraphDecoder:
    """Greedy AR decoding at batch 1 with a static KV cache; each single-token decode step runs as a captured CUDA
    graph (torch.compile "reduce-overhead" with fixed shapes). Same tooling as GraphRunner for diffusion, so the
    two arms are compared at equal implementation effort. With graphs=False it runs eagerly (CPU correctness check)."""

    def __init__(self, model, max_len: int = 700, graphs: bool = True):
        from transformers import StaticCache

        self.model, self.graphs = model, graphs
        self.dev = next(model.parameters()).device
        self.cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device=self.dev,
                                 dtype=model.dtype)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.step = torch.compile(self._step, mode="reduce-overhead", fullgraph=False) if graphs else self._step

    def _step(self, tok, pos):
        return self.model(input_ids=tok, past_key_values=self.cache, cache_position=pos, use_cache=True).logits[:, -1]

    @torch.no_grad()
    def generate(self, ids, max_new_tokens: int, stop_ids):
        self.cache.reset()
        P = len(ids)
        x = torch.tensor([ids], device=self.dev)
        logits = self.model(input_ids=x, past_key_values=self.cache, cache_position=torch.arange(P, device=self.dev),
                            use_cache=True).logits[:, -1]
        out = []
        tok = logits.argmax(-1)
        for i in range(max_new_tokens):
            t = int(tok)
            out.append(t)
            if t in stop_ids:
                break
            self.tok.copy_(tok.view(1, 1))
            self.pos.fill_(P + i)
            if self.graphs:
                torch.compiler.cudagraph_mark_step_begin()
            tok = self.step(self.tok, self.pos).clone().argmax(-1)
        return out


class ARManualGraphDecoder(ARGraphDecoder):
    """Like ARGraphDecoder, but the single-token decode step is captured manually with torch.cuda.graph (exactly the
    technique GraphRunner uses for diffusion), instead of torch.compile."""

    def __init__(self, model, max_len: int = 700):
        super().__init__(model, max_len=max_len, graphs=False)
        self.graph = None

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._step(self.tok, self.pos)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = self._step(self.tok, self.pos)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens: int, stop_ids):
        self.cache.reset()
        P = len(ids)
        x = torch.tensor([ids], device=self.dev)
        logits = self.model(input_ids=x, past_key_values=self.cache, cache_position=torch.arange(P, device=self.dev),
                            use_cache=True).logits[:, -1]
        out, tok = [], logits.argmax(-1)
        for i in range(max_new_tokens):
            t = int(tok)
            out.append(t)
            if t in stop_ids:
                break
            self.tok.copy_(tok.view(1, 1))
            self.pos.fill_(P + i)
            if self.graph is None:  # first decode step: capture (writes this position's KV, replay rewrites it)
                self._capture()
            self.graph.replay()
            tok = self.out.argmax(-1)
        return out
