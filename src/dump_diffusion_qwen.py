"""Dump n test-set predictions from the v2 diffusion checkpoint (Qwen2.5-Coder).

The saved checkpoint loads as a plain causal Qwen2 — it does NOT carry the
bidirectional mask patch or the Dream shift. This script re-applies both
(same code as train_diffusion_qwen.py) and decodes with the same settings as
the training-time gen-eval (conf_stop 0.9, 12-step cap).

Usage (on the GPU box):
  CKPT=sql-diffusion-qwen2.5-coder-0.5b N=256 OUT=preds_diffusion_v2.jsonl \
    python3 src/dump_diffusion_qwen.py

Output JSONL per line: idx, prompt, context, gold, pred, steps, latency_ms,
plus exec-match fields from sql_benchmark for the deterministic metric.
"""

import json
import os
import time
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from denoising import denoise_steps
from sql_benchmark import execution_match, normalize_sql

CKPT = os.environ.get("CKPT", "checkpoints/sql-diffusion-qwen2.5-coder-0.5b")
N = int(os.environ.get("N", "256"))
OUT = os.environ.get("OUT", "preds_diffusion_v2.jsonl")
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "64"))
GEN_STEPS = int(os.environ.get("GEN_EVAL_STEPS", "12"))
CONF_STOP = float(os.environ.get("CONF_STOP", "0.9"))

ANNEAL = {"alpha": 0.0}  # inference: fully bidirectional
CAUSAL_NEG = -20.0

tokenizer = AutoTokenizer.from_pretrained(CKPT)
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
SEP_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id
assert MASK_ID is not None and PAD_ID != tokenizer.eos_token_id, \
    "checkpoint tokenizer must carry <|mask|> and a distinct <|pad|>"


def encode_text(prompt, context, sql):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - SQL_WINDOW - 9
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]
    head = [CLS_ID] if CLS_ID is not None else []
    ids = (head + [TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
           TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'], TAG_IDS['<SQL>']])
    sql_start = len(ids)
    ids = ids + [MASK_ID] * SQL_WINDOW + [TAG_IDS['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW
    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return ids, attention, sql_start, sql_end


def make_bidirectional(model):
    base = model.model if hasattr(model, "model") else model

    def _annealed_bias(attention_mask, input_tensor, kv_length=None):
        b, q = input_tensor.shape[0], input_tensor.shape[1]
        k = int(kv_length or (attention_mask.shape[-1] if attention_mask is not None else q))
        dtype, device = input_tensor.dtype, input_tensor.device
        bias = torch.zeros((b, 1, q, k), dtype=dtype, device=device)
        alpha = ANNEAL["alpha"]
        if alpha > 0.0 and q == k:
            future = torch.triu(torch.ones(q, k, dtype=torch.bool, device=device), diagonal=1)
            bias = bias + future[None, None] * (alpha * CAUSAL_NEG)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask
            pad = attention_mask[:, :k] == 0
            bias = bias.masked_fill(pad[:, None, None, :], torch.finfo(dtype).min)
        return bias

    def _bidir_mask(attention_mask, input_tensor, *args, **kwargs):
        return _annealed_bias(attention_mask, input_tensor)

    patched = False
    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = _bidir_mask
        patched = True
    else:
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

            def _create(config, input_embeds, attention_mask, cache_position,
                        past_key_values=None, position_ids=None, **kw):
                kvl = attention_mask.shape[-1] if attention_mask is not None else input_embeds.shape[1]
                return _annealed_bias(attention_mask, input_embeds, kv_length=kvl)

            qwen2_modeling.create_causal_mask = _create
            patched = base.__class__.__module__ == qwen2_modeling.__name__
        except Exception:
            patched = False
    if not patched:
        raise RuntimeError("could not patch causal mask")
    model.config.is_causal = False
    return model


class ShiftedForDenoise:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return SimpleNamespace(logits=shifted)


device = os.environ.get("DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    # sdpa hits the MPS 2^32-byte NDArray limit at seq 512; eager is fine here.
    attn_implementation="sdpa" if device == "cuda" else "eager",
).to(device)
model = make_bidirectional(model)
model.config.use_cache = False
model.eval()

# sanity: verify bidirectional before trusting 256 predictions
with torch.no_grad():
    ids = torch.randint(0, 1000, (1, 8), device=device)
    attn = torch.ones((1, 8), dtype=torch.long, device=device)
    a = model(input_ids=ids, attention_mask=attn).logits[0, 2].float().clone()
    ids2 = ids.clone(); ids2[0, 6] = (ids2[0, 6] + 7) % 1000
    b = model(input_ids=ids2, attention_mask=attn).logits[0, 2].float()
    assert not torch.allclose(a, b, atol=1e-4), "mask patch did not take — aborting"
print("[bidir] verified")

denoiser = ShiftedForDenoise(model)
ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")

n_done, em, ex_ok, ex_valid = 0, 0, 0, 0
with open(OUT, "w") as f, torch.no_grad():
    for i in range(min(N, len(ds))):
        e = ds[i]
        p, c, s = e["sql_prompt"], e.get("sql_context", ""), e.get("sql", "")
        ids_l, attn_l, lo, hi = encode_text(p, c, s)
        ids = torch.tensor([ids_l], dtype=torch.long, device=device)
        attn = torch.tensor([attn_l], dtype=torch.long, device=device)
        used = 0
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
                else torch.no_grad():
            for used, _, _ in denoise_steps(
                denoiser, ids, attn, list(range(lo, hi)), MASK_ID,
                n_steps=GEN_STEPS, forbid_token_ids=list(TAG_IDS.values()),
                confidence_stop=CONF_STOP if CONF_STOP > 0 else None,
            ):
                pass
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        lat = (time.perf_counter() - t0) * 1e3
        out_ids = [t for t in ids[0, lo:hi].tolist() if t not in (PAD_ID, MASK_ID)]
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        hit = normalize_sql(pred) == normalize_sql(s)
        valid, match = execution_match(c, s, pred)
        em += hit; ex_ok += match; ex_valid += valid; n_done += 1
        f.write(json.dumps({
            "idx": i, "prompt": p, "context": c, "gold": s, "pred": pred,
            "steps": used + 1, "latency_ms": round(lat, 1),
            "exact": hit, "exec_valid": valid, "exec_match": match,
        }) + "\n")
        if (i + 1) % 32 == 0:
            print(f"{i+1}/{N} em={em/n_done:.3f} exec={ex_ok/n_done:.3f} valid={ex_valid/n_done:.3f}")

print(f"FINAL n={n_done}: exact={em/n_done:.3f} exec_match={ex_ok/n_done:.3f} "
      f"exec_valid={ex_valid/n_done:.3f} -> {OUT}")
