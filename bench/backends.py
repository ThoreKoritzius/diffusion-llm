"""Generation backends under test. Each exposes `generate(prompt, context) -> (sql, n_forward)`.

Engines:
  diffusion / torch   ModernBERT masked diffusion, PyTorch (cpu / mps / cuda)
  diffusion / onnx    same model exported to ONNX Runtime (what the hosted playground runs)
  ar / torch          causal LM with HF generate + KV cache
  ar / onnx           causal LM exported by optimum to ONNX Runtime with KV cache (CPU twin of diffusion/onnx)
  ar / vllm           causal LM in vLLM (CUDA graphs, paged KV cache) -- GPU only
"""

import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from denoising import denoise_steps  # noqa: E402

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


class DiffusionBackend:
    def __init__(self, model_dir: str, engine: str = "torch", device: str = "cpu", threads: int = 4,
                 window: int = 128, steps: int = 16, conf_stop: Optional[float] = 0.9, window_head: bool = False,
                 int8: bool = False):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tid = {t: self.tok.convert_tokens_to_ids(t) for t in TAGS}
        self.window, self.steps, self.conf_stop = window, steps, conf_stop
        self.window_head = window_head
        torch.set_num_threads(threads)
        model = AutoModelForMaskedLM.from_pretrained(model_dir, attn_implementation="sdpa", torch_dtype=torch.float32).eval()
        if engine == "torch":
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self._forward = self._torch_forward
        elif engine == "onnx":
            self.device = torch.device("cpu")
            self.session = ort_session(self._export(model, model_dir, int8), threads)
            self._forward = self._onnx_forward
        else:
            raise ValueError(engine)

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
        ids, lo = self.encode(prompt, context)
        self._lo = lo
        ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(ids)
        self._calls = 0
        for _ in denoise_steps(
            self._counted_forward, ids, attn, list(range(lo, lo + self.window)), self.tok.mask_token_id,
            n_steps=self.steps, forbid_token_ids=list(self.tid.values()),
            confidence_stop=None if sample else self.conf_stop,
            temperature=1.0 if sample else 0.0, token_temperature=0.7 if sample else 0.0,
        ):
            pass
        out = [x for x in ids[0, lo:lo + self.window].tolist() if x not in (self.tok.pad_token_id, self.tok.mask_token_id)]
        return self.tok.decode(out, skip_special_tokens=True).strip(), self._calls

    def _counted_forward(self, input_ids, attention_mask):
        self._calls += 1
        return self._forward(input_ids, attention_mask)


class ARBackend:
    def __init__(self, model_dir: str, engine: str = "torch", device: str = "cpu", threads: int = 4,
                 max_new_tokens: int = 130, int8: bool = False):
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
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval().to(self.device)
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
        elif engine == "vllm":  # GPU only; untested on this machine
            from vllm import LLM, SamplingParams

            self.model = LLM(model=model_dir, dtype="bfloat16", enforce_eager=False)
            self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=self.stop_ids)
        else:
            raise ValueError(engine)

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
        if self.engine == "vllm":
            params = self.params.clone() if sample else self.params
            if sample:
                params.temperature = 0.7
            out = self.model.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
            gen = list(out.token_ids)
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
