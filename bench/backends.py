"""Generation backends under test. Each exposes `generate(prompt, context) -> (sql, n_forward)`.

Engines:
  diffusion / torch   ModernBERT masked diffusion, PyTorch (cpu / mps / cuda)
  diffusion / onnx    same model exported to ONNX Runtime (what the hosted playground runs)
  ar / torch          causal LM with HF generate + KV cache
  ar / onnx           causal LM exported by optimum to ONNX Runtime with KV cache (CPU twin of diffusion/onnx)
  ar / vllm           causal LM in vLLM (CUDA graphs, paged KV cache) -- GPU only
"""

import json
import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from denoising import denoise_steps  # noqa: E402
from prompt_cache import PrefillGraph, PromptCache, StepGraph  # noqa: E402
from prompt_isolation import is_isolated_checkpoint, isolate_prompt, set_prompt_len  # noqa: E402

from bench.common import TAGS  # noqa: E402

ONNX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".onnx_cache")


def ort_session(path: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


class _FullHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class _WindowHead(torch.nn.Module):
    """Encoder + MLM head, with the head applied only to the SQL window positions.
    The full-sequence head is wasted work: denoising only reads logits inside the window."""

    def __init__(self, model, window: int):
        super().__init__()
        self.model, self.window = model, window

    def forward(self, input_ids, attention_mask, window_start):
        hidden = self.model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = window_start + torch.arange(self.window, device=hidden.device)
        return self.model.decoder(self.model.head(hidden[:, idx]))


CORE_VOCAB_MIN_COUNT = 20


def sql_core_vocab(tok) -> torch.Tensor:
    """Tokens that occur at least CORE_VOCAB_MIN_COUNT times in the training split's SQL (keywords, operators,
    digits, common words), as a bool mask; cached per tokenizer. Combined per query with the prompt/schema tokens,
    it keeps the decoder from inventing identifiers."""
    import collections
    import json

    path = os.path.join(ONNX_CACHE, f"core_vocab-{tok.name_or_path.replace('/', '_')}-{len(tok)}.json")
    if os.path.exists(path):
        ids = json.load(open(path))
    else:
        from datasets import load_dataset

        sql = list(load_dataset("gretelai/synthetic_text_to_sql", split="train").select(range(100_000))["sql"])
        counts = collections.Counter(t for ids in tok(sql, add_special_tokens=False)["input_ids"] for t in ids)
        ids = sorted(t for t, c in counts.items() if c >= CORE_VOCAB_MIN_COUNT)
        os.makedirs(ONNX_CACHE, exist_ok=True)
        json.dump(ids, open(path, "w"))
    mask = torch.zeros(len(tok), dtype=torch.bool)
    mask[ids] = True
    mask[tok.pad_token_id] = True
    return mask


class DiffusionBackend:
    def __init__(self, model_dir: str, engine: str = "torch", device: str = "cpu", threads: int = 4,
                 window: int = 128, steps: int = 16, conf_stop: Optional[float] = 0.9, window_head: bool = False,
                 int8: bool = False, prompt_isolation: bool = False, dtype: str = "float32",
                 prompt_cache: bool = False, cuda_graphs: bool = False, refine: float = 0.0,
                 schema_vocab: bool = False, grow_window: int = 0, repair: bool = False,
                 grammar: bool = False):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tid = {t: self.tok.convert_tokens_to_ids(t) for t in TAGS}
        self.window, self.steps, self.conf_stop = window, steps, conf_stop
        self.refine, self.grow_window, self.repair = refine, grow_window, repair
        self.auto_window = os.environ.get("DIFF_AUTO_WINDOW") == "1"  # 2x window when the schema has several tables
        self.core_vocab = sql_core_vocab(self.tok) if schema_vocab else None
        self.window_head = window_head
        self.grammar = None
        if grammar:
            from bench.grammar import SqlGrammar, train_sql_words

            self.grammar = SqlGrammar(self.tok, len(self.tok), train_sql_words())
        torch.set_num_threads(threads)
        model = AutoModelForMaskedLM.from_pretrained(model_dir, attn_implementation="sdpa", torch_dtype=torch.float32).eval()
        # checkpoints trained with prompt isolation must be run with it
        self.prompt_isolation = prompt_isolation or is_isolated_checkpoint(model.config)
        if self.prompt_isolation:
            isolate_prompt(model)  # ONNX: only via the cached graphs (checked below)
        if prompt_cache and not self.prompt_isolation:
            raise ValueError("--prompt-cache needs a prompt-isolated checkpoint")
        self.cache = None
        self.onnx_cache = None
        if engine == "torch":
            self.device = torch.device(device)
            self.model = model.to(self.device, dtype=getattr(torch, dtype))
            if prompt_cache:
                self.cache = PromptCache(self.model)
            self._forward = self._torch_forward
        elif engine == "onnx" and prompt_cache:
            self.device = torch.device("cpu")
            self.onnx_cache = self._export_cached(model, model_dir, int8, threads)
        elif engine == "onnx":
            if self.prompt_isolation:
                raise ValueError("isolated checkpoints on ONNX need --prompt-cache (full graph would bake the mask)")
            self.device = torch.device("cpu")
            self.session = ort_session(self._export(model, model_dir, int8), threads)
            self._forward = self._onnx_forward
        else:
            raise ValueError(engine)
        self.graphs = None
        if cuda_graphs:
            from bench.cuda_graphs import GraphRunner

            self.graphs = GraphRunner(self.model, self.prompt_isolation, compiled=os.environ.get("DIFF_COMPILE") == "1")

    # ---- ONNX export (same graph as src/inference.py, optionally window-only head / int8) ----
    def _export(self, model, model_dir: str, int8: bool) -> str:
        tag = f"{os.path.basename(os.path.normpath(model_dir))}-w{self.window if self.window_head else 'full'}"
        path = os.path.join(ONNX_CACHE, f"{tag}.onnx")
        if not os.path.exists(path):
            os.makedirs(ONNX_CACHE, exist_ok=True)
            ids = torch.full((1, 200), self.tok.mask_token_id, dtype=torch.long)
            attn = torch.ones_like(ids)
            if self.window_head:
                wrapper = _WindowHead(model, self.window)
                args, names = (ids, attn, torch.tensor(50)), ["input_ids", "attention_mask", "window_start"]
            else:
                wrapper = _FullHead(model)
                args, names = (ids, attn), ["input_ids", "attention_mask"]
            dyn = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                   "logits": {0: "batch"} if self.window_head else {0: "batch", 1: "sequence"}}
            with torch.inference_mode():
                torch.onnx.export(wrapper, args, path, input_names=names, output_names=["logits"],
                                  dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
        if int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            qpath = os.path.splitext(path)[0] + ".int8.onnx"
            if not os.path.exists(qpath):
                quantize_dynamic(path, qpath, weight_type=QuantType.QInt8)
            path = qpath
        return path

    def _export_cached(self, model, model_dir: str, int8: bool, threads: int):
        """Two ONNX graphs: prompt -> per-layer k/v, and window + k/v -> window logits."""
        tag = os.path.basename(os.path.normpath(model_dir))
        paths = [os.path.join(ONNX_CACHE, f"{tag}-prefill.onnx"), os.path.join(ONNX_CACHE, f"{tag}-step.onnx")]
        cache = PromptCache(model)
        names = [f"{t}{i}" for i in range(len(model.model.layers)) for t in ("k", "v")]
        if not all(os.path.exists(p) for p in paths):
            os.makedirs(ONNX_CACHE, exist_ok=True)
            ids, lo = self.encode("How many rows are there?", "CREATE TABLE t (id INT);")
            x = torch.tensor([ids])
            dyn = {n: {2: "prompt"} for n in names}
            with torch.inference_mode():
                torch.onnx.export(PrefillGraph(cache), (x[:, :lo],), paths[0], input_names=["prompt_ids"],
                                  output_names=names, dynamic_axes={"prompt_ids": {1: "prompt"}, **dyn},
                                  opset_version=17, dynamo=False)
                flat = tuple(t for kv in cache.prefill_kv(x[:, :lo]) for t in kv)
                torch.onnx.export(StepGraph(cache), (x[:, lo:], *flat), paths[1], input_names=["window_ids", *names],
                                  output_names=["logits"], dynamic_axes={"window_ids": {1: "window"}, **dyn,
                                                                         "logits": {1: "window"}},
                                  opset_version=17, dynamo=False)
        if int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            qpaths = [os.path.splitext(p)[0] + ".int8.onnx" for p in paths]
            for src, dst in zip(paths, qpaths):
                if not os.path.exists(dst):
                    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
            paths = qpaths
        return {"prefill": ort_session(paths[0], threads), "step": ort_session(paths[1], threads), "names": names}

    def _torch_forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _onnx_forward(self, input_ids, attention_mask):
        feed = {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
        if self.window_head:
            feed["window_start"] = np.array(self._lo, dtype=np.int64)
        logits = torch.from_numpy(self.session.run(None, feed)[0])
        if self.window_head:  # scatter window logits back so denoise_steps can index absolute positions
            full = torch.zeros(1, input_ids.shape[1], logits.shape[-1])
            full[:, self._lo:self._lo + self.window] = logits
            logits = full
        return SimpleNamespace(logits=logits)

    def encode(self, prompt: str, context: str) -> Tuple[List[int], int]:
        """[CLS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL> [MASK]*window </SQL> [SEP] -- no padding."""
        t = self.tid
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c = self.tok(context, add_special_tokens=False)["input_ids"]
        budget = 512 - self.window - 9
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        ids = [self.tok.cls_token_id, t["<PROMPT>"]] + p + [t["</PROMPT>"], t["<CONTEXT>"]] + c + [t["</CONTEXT>"], t["<SQL>"]]
        lo = len(ids)
        return ids + [self.tok.mask_token_id] * self.window + [t["</SQL>"], self.tok.sep_token_id], lo

    @torch.no_grad()
    def generate(self, prompt: str, context: str, sample: bool = False) -> Tuple[str, int]:
        from bench.common import compiles

        allowed = None
        if self.core_vocab is not None:  # SQL vocabulary + every token of the prompt and schema
            allowed = self.core_vocab.clone()
            allowed[self.tok(prompt + " " + context, add_special_tokens=False)["input_ids"]] = True
        base_window = self.window
        if self.auto_window and context.upper().count("CREATE TABLE") >= 2:
            self.window = 2 * base_window
        try:
            sql, calls, full = self._decode(prompt, context, sample, self.refine, allowed)
        finally:
            self.window = base_window
        if full and self.grow_window:  # SQL ran to the end of the window: decode again with more room
            self.window += self.grow_window
            try:
                sql, more, _ = self._decode(prompt, context, sample, self.refine, allowed)
            finally:
                self.window -= self.grow_window
            calls += more
        if self.repair and not sample and not compiles(context, sql):  # re-mask the shakiest tokens and redo
            fixed, more, _ = self._decode(prompt, context, False, 0.3, allowed)
            calls += more
            if compiles(context, fixed):
                sql = fixed
        return sql, calls

    def _decode(self, prompt: str, context: str, sample: bool, refine: float, allowed) -> Tuple[str, int, bool]:
        ids, lo = self.encode(prompt, context)
        self._lo = lo
        self.prompt_tokens = lo
        if self.prompt_isolation and self.onnx_cache is None:
            set_prompt_len(self.model, lo)
        ids = torch.tensor([ids], device=self.device)
        if self.graphs is not None:
            self.graphs.prepare(ids.shape[1], lo)
        if self.cache is not None:  # prompt runs once; each step only processes the window (+ closing tags)
            self.cache.prefill(ids[:, :lo])
            self._full = torch.zeros(1, ids.shape[1], self.model.config.vocab_size, device=self.device)
        if self.onnx_cache is not None:
            kv = self.onnx_cache["prefill"].run(None, {"prompt_ids": ids[:, :lo].numpy()})
            self._onnx_kv = dict(zip(self.onnx_cache["names"], kv))
            self._full = torch.zeros(1, ids.shape[1], len(self.tok))
        attn = torch.ones_like(ids)
        self._calls = 0
        constrain = self._grammar_constraint(ids, lo, prompt, context) if self.grammar is not None else None
        for _ in denoise_steps(
            self._counted_forward, ids, attn, list(range(lo, lo + self.window)), self.tok.mask_token_id,
            n_steps=self.steps, forbid_token_ids=list(self.tid.values()),
            confidence_stop=None if sample else self.conf_stop,
            temperature=1.0 if sample else 0.0, token_temperature=0.7 if sample else 0.0,
            refine_frac=refine, allowed_tokens=allowed, constrain=constrain,
        ):
            pass
        window = ids[0, lo:lo + self.window].tolist()
        out = [x for x in window if x not in (self.tok.pad_token_id, self.tok.mask_token_id)]
        full = window[-1] not in (self.tok.pad_token_id, self.tok.mask_token_id)
        return self.tok.decode(out, skip_special_tokens=True).strip(), self._calls, full

    def _grammar_constraint(self, ids, lo: int, prompt: str, context: str):
        """Per step: most probable window completion that the lexical SQL grammar accepts (bench/grammar.py)."""
        from bench.grammar import NEG, as_aliases

        q = self.grammar.for_query(prompt, context)
        mask, pad, W = self.tok.mask_token_id, self.tok.pad_token_id, self.window

        def constrain(masked_pos, logits):
            window = ids[0, lo:lo + W].tolist()
            q.add_words(as_aliases(self.tok.decode([t for t in window if t not in (mask, pad)])))
            V = len(self.tok)
            lp = np.full((W, V), NEG, dtype=np.float32)
            rows = (masked_pos - lo).cpu().numpy()
            lp[rows, : logits.shape[-1]] = torch.log_softmax(logits, -1).cpu().numpy()
            out = q.decode([None if t == mask else t for t in window], lp)
            if out is None:
                return logits.argmax(dim=-1)
            return torch.tensor([out[r] for r in rows], device=logits.device)

        return constrain

    def _counted_forward(self, input_ids, attention_mask):
        self._calls += 1
        if self.graphs is not None:
            return SimpleNamespace(logits=self.graphs(input_ids))
        if self.cache is not None:
            self._full[:, self._lo:] = self.cache.step(input_ids[:, self._lo:])
            return SimpleNamespace(logits=self._full)
        if self.onnx_cache is not None:
            window = input_ids[:, self._lo:].numpy()
            logits = self.onnx_cache["step"].run(None, {"window_ids": window, **self._onnx_kv})[0]
            self._full[:, self._lo:, : logits.shape[-1]] = torch.from_numpy(logits)
            return SimpleNamespace(logits=self._full)
        return self._forward(input_ids, attention_mask)


class ARBackend:
    def __init__(self, model_dir: str, engine: str = "torch", device: str = "cpu", threads: int = 4,
                 max_new_tokens: int = 130, int8: bool = False, dtype: str = "float32", compile: bool = False,
                 cuda_graphs: bool = False, eager_static: bool = False):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tid = {t: self.tok.convert_tokens_to_ids(t) for t in TAGS}
        self.stop_ids = [self.tid["</SQL>"], self.tok.eos_token_id]
        self.max_new_tokens = max_new_tokens
        self.engine = engine
        torch.set_num_threads(threads)
        if engine == "torch":
            from transformers import AutoModelForCausalLM

            self.device = torch.device(device)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=getattr(torch, dtype)).eval().to(self.device)
            if compile:  # static KV cache; transformers then compiles the decode step itself (CUDA graphs)
                self.model.generation_config.cache_implementation = "static"
            self.graph_decoder = None
            if cuda_graphs or eager_static:  # same-effort harness as diffusion's CUDA graphs
                from bench.cuda_graphs import ARGraphDecoder, ARManualGraphDecoder

                if cuda_graphs and os.environ.get("AR_GRAPHS") == "manual":
                    self.graph_decoder = ARManualGraphDecoder(self.model)
                else:
                    self.graph_decoder = ARGraphDecoder(self.model, graphs=cuda_graphs)
        elif engine == "onnx":
            from optimum.onnxruntime import ORTModelForCausalLM

            self.device = torch.device("cpu")
            export_dir = os.path.join(ONNX_CACHE, os.path.basename(os.path.normpath(model_dir)) + "-ar")
            if not os.path.exists(os.path.join(export_dir, "model.onnx")):
                ORTModelForCausalLM.from_pretrained(model_dir, export=True, use_cache=True).save_pretrained(export_dir)
            if int8:  # same dynamic int8 weight quantization as the diffusion arm
                import shutil

                from onnxruntime.quantization import QuantType, quantize_dynamic

                qdir = export_dir + "-int8"
                if not os.path.exists(os.path.join(qdir, "model.onnx")):
                    shutil.copytree(export_dir, qdir, ignore=shutil.ignore_patterns("model.onnx*"), dirs_exist_ok=True)
                    quantize_dynamic(os.path.join(export_dir, "model.onnx"), os.path.join(qdir, "model.onnx"),
                                     weight_type=QuantType.QInt8, use_external_data_format=True)
                export_dir = qdir
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            self.model = ORTModelForCausalLM.from_pretrained(export_dir, use_cache=True, session_options=opts,
                                                             provider="CPUExecutionProvider")
        elif engine == "llamacpp":  # CPU-only llama-server on a GGUF export (pass the .gguf via LLAMACPP_GGUF)
            import atexit
            import subprocess
            import urllib.request

            gguf = os.environ["LLAMACPP_GGUF"]
            self.port = int(os.environ.get("LLAMACPP_PORT", "8091"))
            self.server = subprocess.Popen(
                ["llama-server", "-m", gguf, "-t", str(threads), "-tb", str(threads), "-ngl", "0", "-c", "1024",
                 "--port", str(self.port), "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(self.server.terminate)
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                    break
                except Exception:
                    __import__("time").sleep(0.5)
        elif engine == "vllm":  # GPU only
            from vllm import LLM, SamplingParams

            # VLLM_SPEC: speculative-decoding config as JSON, e.g. '{"method": "ngram", "num_speculative_tokens": 8,
            # "prompt_lookup_max": 4}' or '{"method": "dflash", "model": "<drafter dir>", "num_speculative_tokens": 15}'
            spec = os.environ.get("VLLM_SPEC")
            # stats on only with a drafter, for the acceptance numbers in spec_stats()
            self.model = LLM(model=model_dir, dtype="bfloat16", enforce_eager=False,
                             gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")),
                             **({"speculative_config": json.loads(spec), "disable_log_stats": False} if spec else {}))
            self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=self.stop_ids)
        else:
            raise ValueError(engine)

    def spec_stats(self) -> Optional[dict]:
        """Speculative decoding in vLLM: drafts, mean accepted tokens per draft, acceptance per draft position."""
        if self.engine != "vllm" or not os.environ.get("VLLM_SPEC"):
            return None
        m = {x.name: x for x in self.model.get_metrics()}
        get = lambda name: getattr(m.get(name), "value", 0)  # noqa: E731
        drafts, drafted = get("vllm:spec_decode_num_drafts"), get("vllm:spec_decode_num_draft_tokens")
        accepted = get("vllm:spec_decode_num_accepted_tokens")
        per_pos = getattr(m.get("vllm:spec_decode_num_accepted_tokens_per_pos"), "values", [])
        return {"drafts": drafts, "draft_tokens": drafted, "accepted_tokens": accepted,
                "mean_accepted": round(accepted / drafts, 3) if drafts else None,
                "acceptance_rate": round(accepted / drafted, 3) if drafted else None,
                "accept_per_pos": [round(v / drafts, 3) for v in per_pos] if drafts else []}

    def prompt_ids(self, prompt: str, context: str) -> List[int]:
        """Same layout as src/train_ar.py: [BOS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL>"""
        t = self.tid
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c = self.tok(context, add_special_tokens=False)["input_ids"]
        budget = 512 - 128 - 12
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        return bos + [t["<PROMPT>"]] + p + [t["</PROMPT>"], t["<CONTEXT>"]] + c + [t["</CONTEXT>"], t["<SQL>"]]

    @torch.no_grad()
    def generate(self, prompt: str, context: str, sample: bool = False) -> Tuple[str, int]:
        ids = self.prompt_ids(prompt, context)
        if self.engine == "llamacpp":
            import json as _json
            import urllib.request

            body = {"prompt": ids, "n_predict": self.max_new_tokens, "temperature": 0.7 if sample else 0.0,
                    "top_k": 20 if sample else 1, "cache_prompt": False, "stop": ["</SQL>"]}
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", _json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            res = _json.loads(urllib.request.urlopen(req, timeout=120).read())
            text = res["content"].split("</SQL>")[0]
            return text.replace(self.tok.eos_token or "", "").strip(), res.get("tokens_predicted", 0)
        if self.engine == "vllm":
            params = self.params.clone() if sample else self.params
            if sample:
                params.temperature = 0.7
            out = self.model.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
            gen = list(out.token_ids)
        elif getattr(self, "graph_decoder", None) is not None and not sample:
            gen = self.graph_decoder.generate(ids, self.max_new_tokens, self.stop_ids)
        else:
            x = torch.tensor([ids], device=self.device)
            out = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=self.max_new_tokens,
                                      do_sample=sample, temperature=0.7 if sample else None, top_k=20 if sample else None,
                                      num_beams=1, eos_token_id=self.stop_ids,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            gen = out[0, len(ids):].tolist()
        n_forward = len(gen)  # one prefill (which also emits token 1) + one decode step per further token
        gen = [g for g in gen if g not in self.stop_ids]
        return self.tok.decode(gen, skip_special_tokens=True).strip(), n_forward


class VerifiedBackend:
    """Greedy first; if the result does not compile against the schema, draw up to `k` sampled candidates and
    return the first that compiles (else the greedy one). Forward passes of all attempts are counted."""

    def __init__(self, backend, k: int):
        self.backend, self.k = backend, k
        self.device = getattr(backend, "device", torch.device("cpu"))
        self.model = getattr(backend, "model", None)

    def generate(self, prompt: str, context: str) -> Tuple[str, int]:
        from bench.common import compiles

        best, total = self.backend.generate(prompt, context)
        if compiles(context, best):
            return best, total
        for _ in range(self.k):
            cand, n = self.backend.generate(prompt, context, sample=True)
            total += n
            if compiles(context, cand):
                return cand, total
        return best, total
