"""Uniform local (Mac) latency benchmark for the four Current Results models.

Protocol — identical for every model so the column is comparable:
- same device (MPS if available, else CPU) and same dtype for all
- attn_implementation="eager" for all (uniform; avoids the MPS SDPA 2^32 assert)
- same 32 eval examples (first 32 filtered gretelai test rows, as in gen-eval)
- 3 warmup examples before timing
- each model decodes at ITS OWN documented operating point:
    AR-Qwen                greedy, KV cache, stop at </SQL>/EOS
    Diffusion-Qwen v2      window 64, cap 12, conf_stop 0.9, bidir + shift
    Diffusion-Qwen v1      window 128, fixed 24 steps, bidir, no shift
    Diffusion-ModernBERT   window 128, cap 16, conf_stop 0.9

Writes data/local_latency.json and prints a table.
Usage: DTYPE=float16 python3 bench_local_latency.py
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from denoising import denoise_steps  # noqa: E402

DEVICE = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = {"float16": torch.float16, "float32": torch.float32}[os.environ.get("DTYPE", "float16")]
N = int(os.environ.get("N", "32"))
WARMUP = 3
MAX_LEN = 512

ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
EXAMPLES = [ds[i] for i in range(N + WARMUP)]

TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
ANNEAL = {"alpha": 0.0}


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def make_bidirectional(model):
    base = model.model if hasattr(model, "model") else model

    def _bias(attention_mask, input_tensor, kv_length=None):
        b, q = input_tensor.shape[0], input_tensor.shape[1]
        k = int(kv_length or (attention_mask.shape[-1] if attention_mask is not None else q))
        dtype, device = input_tensor.dtype, input_tensor.device
        bias = torch.zeros((b, 1, q, k), dtype=dtype, device=device)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask
            pad = attention_mask[:, :k] == 0
            bias = bias.masked_fill(pad[:, None, None, :], torch.finfo(dtype).min)
        return bias

    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = lambda am, it, *a, **k: _bias(am, it)
    else:
        import transformers.models.qwen2.modeling_qwen2 as qm

        def _create(config, input_embeds, attention_mask, cache_position,
                    past_key_values=None, position_ids=None, **kw):
            kvl = attention_mask.shape[-1] if attention_mask is not None else input_embeds.shape[1]
            return _bias(attention_mask, input_embeds, kv_length=kvl)

        qm.create_causal_mask = _create
    model.config.is_causal = False
    return model


class Shifted:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return SimpleNamespace(logits=torch.cat([logits[:, :1], logits[:, :-1]], dim=1))


def encode_diffusion(tok, tag_ids, cls_id, sep_id, pad_id, mask_id, window, ex):
    p = tok(ex["sql_prompt"], add_special_tokens=False)["input_ids"]
    c = tok(ex.get("sql_context", ""), add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - window - 9
    if len(p) + len(c) > budget:
        c = c[: max(0, budget - len(p))]
        p = p[:budget]
    head = [cls_id] if cls_id is not None else []
    ids = (head + [tag_ids['<PROMPT>']] + p + [tag_ids['</PROMPT>'], tag_ids['<CONTEXT>']]
           + c + [tag_ids['</CONTEXT>'], tag_ids['<SQL>']])
    lo = len(ids)
    ids = ids + [mask_id] * window + [tag_ids['</SQL>'], sep_id]
    hi = lo + window
    attn = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [pad_id] * (MAX_LEN - len(ids))
    return ids, attn, lo, hi


@torch.no_grad()
def bench_diffusion(path, window, n_steps, conf_stop, shifted, masked_lm=False):
    tok = AutoTokenizer.from_pretrained(path)
    cls_ = AutoModelForMaskedLM if masked_lm else AutoModelForCausalLM
    model = cls_.from_pretrained(path, torch_dtype=DTYPE, attn_implementation="eager").to(DEVICE)
    if not masked_lm:
        model = make_bidirectional(model)
        model.config.use_cache = False
    model.eval()
    tag_ids = {t: tok.convert_tokens_to_ids(t) for t in TAGS}
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    runner = Shifted(model) if shifted else model
    times, steps_used, sample = [], [], None
    for i, ex in enumerate(EXAMPLES):
        ids_l, attn_l, lo, hi = encode_diffusion(
            tok, tag_ids, cls_id, sep_id, tok.pad_token_id, tok.mask_token_id, window, ex)
        ids = torch.tensor([ids_l], device=DEVICE)
        attn = torch.tensor([attn_l], device=DEVICE)
        used = 0
        sync(); t0 = time.perf_counter()
        for used, _, _ in denoise_steps(
            runner, ids, attn, list(range(lo, hi)), tok.mask_token_id,
            n_steps=n_steps, forbid_token_ids=list(tag_ids.values()),
            confidence_stop=conf_stop,
        ):
            pass
        sync(); dt = time.perf_counter() - t0
        if i >= WARMUP:
            times.append(dt); steps_used.append(used + 1)
        if i == WARMUP and sample is None:
            out = [t for t in ids[0, lo:hi].tolist()
                   if t not in (tok.pad_token_id, tok.mask_token_id)]
            sample = tok.decode(out, skip_special_tokens=True)[:80]
    del model
    return times, sum(steps_used) / len(steps_used), sample


@torch.no_grad()
def bench_ar(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, attn_implementation="eager").to(DEVICE)
    model.config.use_cache = True
    model.eval()
    tag_ids = {t: tok.convert_tokens_to_ids(t) for t in TAGS}
    close_sql = tag_ids['</SQL>']
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    times, new_tokens, sample = [], [], None
    for i, ex in enumerate(EXAMPLES):
        p = tok(ex["sql_prompt"], add_special_tokens=False)["input_ids"]
        c = tok(ex.get("sql_context", ""), add_special_tokens=False)["input_ids"]
        budget = MAX_LEN - 128 - 12
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        ids = (bos + [tag_ids['<PROMPT>']] + p + [tag_ids['</PROMPT>'], tag_ids['<CONTEXT>']]
               + c + [tag_ids['</CONTEXT>'], tag_ids['<SQL>']])
        ids = torch.tensor([ids], device=DEVICE)
        sync(); t0 = time.perf_counter()
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=130,
            do_sample=False, num_beams=1,
            eos_token_id=[tok.eos_token_id, close_sql], pad_token_id=tok.pad_token_id)
        sync(); dt = time.perf_counter() - t0
        gen = out[0, ids.shape[1]:].tolist()
        if i >= WARMUP:
            times.append(dt); new_tokens.append(len(gen))
        if i == WARMUP and sample is None:
            cut = gen.index(close_sql) if close_sql in gen else len(gen)
            sample = tok.decode(gen[:cut], skip_special_tokens=True)[:80]
    del model
    return times, sum(new_tokens) / len(new_tokens), sample


def stats(times):
    s = sorted(times)
    return {"mean_ms": 1e3 * sum(s) / len(s), "p50_ms": 1e3 * s[len(s) // 2],
            "p90_ms": 1e3 * s[int(len(s) * 0.9)]}


results = {}
runs = [
    ("AR-Qwen", lambda: bench_ar(os.path.join(REPO, "checkpoints/sql-ar-qwen25-coder-0.5b-ar"))),
    ("Diffusion-Qwen v2", lambda: bench_diffusion(
        os.path.join(REPO, "sql-diffusion-qwen2.5-coder-0.5b"), 64, 12, 0.9, shifted=True)),
    ("Diffusion-Qwen v1 ckpt-3126", lambda: bench_diffusion(
        os.path.join(REPO, "checkpoints/sql-diffusion-qwen25-coder-0.5b-checkpoint-3126"),
        128, 24, None, shifted=False)),
    ("Diffusion-ModernBERT", lambda: bench_diffusion(
        os.path.join(REPO, "diffusion-sql-modernbert"), 128, 16, 0.9,
        shifted=False, masked_lm=True)),
]
for name, fn in runs:
    print(f"[bench] {name} ...", flush=True)
    try:
        times, units, sample = fn()
        results[name] = {**stats(times), "avg_steps_or_tokens": units,
                         "n": len(times), "sample": sample}
        print(f"  mean={results[name]['mean_ms']:.0f}ms p50={results[name]['p50_ms']:.0f}ms "
              f"units={units:.1f}  sample: {sample!r}", flush=True)
    except Exception as e:  # noqa: BLE001 — keep benching the rest
        results[name] = {"error": str(e)[:300]}
        print(f"  FAILED: {e}", flush=True)

meta = {"device": DEVICE, "dtype": str(DTYPE), "attn": "eager", "n": N,
        "warmup": WARMUP, "torch": torch.__version__}
json.dump({"meta": meta, "results": results},
          open(os.path.join(HERE, "data", "local_latency.json"), "w"), indent=2)
print("\n== summary ==", json.dumps(meta))
for k, v in results.items():
    if "mean_ms" in v:
        print(f"{k:32s} mean={v['mean_ms']:7.0f}ms  p50={v['p50_ms']:7.0f}ms  "
              f"steps/tok={v['avg_steps_or_tokens']:.1f}")
    else:
        print(f"{k:32s} ERROR {v['error']}")
