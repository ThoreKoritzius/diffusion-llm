"""Dump model predictions for LLM-judge grading.

bench_papl.py computed exact-match but discarded the predicted SQL. This re-runs
the confidence-planner decoding on a fixed eval slice and saves, per model, a
JSONL of {idx, prompt, context, gold, pred, exact_match} so an external judge
(judge_sql.py) can grade semantic correctness and annotate failure modes.

Pure inference — runs on CPU/MPS (slow but fine for a few hundred examples) or
CUDA. Same eval slice across models for apples-to-apples.

Usage:
    python3 src/dump_predictions.py <ab_dir> [--n 200] [--models base,tau_0.3]
        <ab_dir> = the papl_ab_* dir holding tau_0/ and tau_0.3/
Env:
    DUMP_N (default 200), DUMP_OUT (default <ab_dir>/predictions),
    CONF_STOP (default 0.9), GEN_STEPS (default 24)
"""
import contextlib
import json
import os
import re
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from denoising import denoise_steps

MAX_LEN = 512
SQL_WINDOW = 128

AB_DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "."
BASE_DIR = os.environ.get("BASE_DIR", "diffusion-sql-modernbert")


def _resolve(name, *candidates):
    """First existing dir among env override + candidates (box layout or local names)."""
    env = os.environ.get(name.upper().replace(".", "") + "_DIR")
    for c in ([env] if env else []) + list(candidates):
        if c and os.path.isdir(c):
            return c
    return candidates[0]


ALL_MODELS = {
    "base":    _resolve("base", BASE_DIR),
    "tau_0":   _resolve("tau0", os.path.join(AB_DIR, "tau_0"), "diffusion-sql-modernbert-papl-tau0"),
    "tau_0.3": _resolve("tau03", os.path.join(AB_DIR, "tau_0.3"), "diffusion-sql-modernbert-papl"),
}
want = os.environ.get("DUMP_MODELS", "base,tau_0,tau_0.3").split(",")
MODELS = {k: v for k, v in ALL_MODELS.items() if k in want}

N = int(os.environ.get("DUMP_N", "200"))
CONF_STOP = float(os.environ.get("CONF_STOP", "0.9"))
GEN_STEPS = int(os.environ.get("GEN_STEPS", "24"))
# Inference-improvement knobs (training-free):
#   REFINE_FRAC  >0 -> targeted-remask refinement in the sampler (see denoising.py)
#   REPAIR=1        -> sqlglot verify-repair loop after decoding (see sql_repair.py)
#   TAG             -> appended to the model name in output filenames (config A/B)
REFINE_FRAC = float(os.environ.get("REFINE_FRAC", "0"))
REFINE_ROUNDS = int(os.environ.get("REFINE_ROUNDS", "1"))
REPAIR = os.environ.get("REPAIR", "0") == "1"
TAG = os.environ.get("TAG", "")
OUT_DIR = os.environ.get("DUMP_OUT", os.path.join(AB_DIR, "predictions"))
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
tokenizer.model_max_length = MAX_LEN
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
SEP_ID = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id


def encode_text(prompt, context, sql):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    sql_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))
    budget = MAX_LEN - SQL_WINDOW - 9
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]
    ids = ([CLS_ID, TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
           TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'], TAG_IDS['<SQL>']])
    sql_start = len(ids)
    ids = ids + sql_ids + [TAG_IDS['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW
    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {"input_ids": ids, "attention_mask": attention, "sql_start": sql_start, "sql_end": sql_end}


def normalize_sql(s):
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def predict(model, ex, conf_stop=CONF_STOP, n_steps=GEN_STEPS):
    """Returns (predicted_sql, steps_used). Steps include any refinement and
    repair passes, so the frontier accounting stays honest."""
    p, c, s = ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", "")
    enc = encode_text(p, c, s)
    ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=DEVICE)
    attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=DEVICE)
    lo, hi = enc["sql_start"], enc["sql_end"]
    ids[0, lo:hi] = MASK_ID
    steps = 0
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if DEVICE == "cuda" else contextlib.nullcontext())
    with ctx:
        for _ in denoise_steps(model, ids, attn, list(range(lo, hi)), MASK_ID,
                               n_steps=n_steps, forbid_token_ids=list(TAG_IDS.values()),
                               confidence_stop=conf_stop,
                               refine_frac=REFINE_FRAC, refine_rounds=REFINE_ROUNDS):
            steps += 1
            if REFINE_FRAC <= 0 and (ids[0, lo:hi] == MASK_ID).sum().item() == 0:
                break
        if REPAIR:
            from sql_repair import verify_and_repair
            text, extra, _ = verify_and_repair(
                model, tokenizer, ids, attn, lo, hi, MASK_ID, PAD_ID,
                list(TAG_IDS.values()), conf_stop)
            return text, steps + extra
    out = [t for t in ids[0, lo:hi].tolist() if t not in (PAD_ID, MASK_ID)]
    return tokenizer.decode(out, skip_special_tokens=True), steps


# FIXED_STEPS="4,8,16,24" switches to fixed-budget mode: for each K, decode with
# early-stop OFF and exactly K steps, writing pred_<model>_k<K>.jsonl. Used to
# re-grade the fixed-NFE curve with the semantic judge. Default (unset) = the
# normal early-stop dump (one pred_<model>.jsonl per model).
FIXED_STEPS = [int(k) for k in os.environ.get("FIXED_STEPS", "").split(",") if k.strip()]
# CONF_SWEEP="0.8,0.9,0.95,0.99" -> early-stop at each threshold, one file per
# (model, threshold): pred_<model>_c<cs>.jsonl. Used for the semantic frontier.
CONF_SWEEP = [float(c) for c in os.environ.get("CONF_SWEEP", "").split(",") if c.strip()]


def dump_one(model, name, examples, conf_stop, n_steps, suffix=""):
    out_path = os.path.join(OUT_DIR, f"pred_{name}{suffix}.jsonl")
    hits = 0
    tot_steps = 0
    with open(out_path, "w") as f:
        for i, ex in enumerate(examples):
            pred, steps = predict(model, ex, conf_stop=conf_stop, n_steps=n_steps)
            em = normalize_sql(pred) == normalize_sql(ex.get("sql", ""))
            hits += int(em); tot_steps += steps
            f.write(json.dumps({
                "idx": i, "prompt": ex["sql_prompt"], "context": ex.get("sql_context", ""),
                "gold": ex.get("sql", ""), "pred": pred, "exact_match": em, "steps": steps,
            }) + "\n")
            if (i + 1) % 50 == 0:
                print(f"    {name}{suffix}: {i+1}/{len(examples)} (em {hits/(i+1):.3f})", flush=True)
    print(f"[dump] {name}{suffix}: wrote {out_path} | exact_match={hits/len(examples):.3f} "
          f"avg_steps={tot_steps/len(examples):.2f}")


ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
examples = [ds[i] for i in range(min(N, len(ds)))]
mode = (f"fixed-NFE K={FIXED_STEPS}" if FIXED_STEPS else
        f"conf-sweep {CONF_SWEEP}" if CONF_SWEEP else f"early-stop conf_stop={CONF_STOP}")
print(f"[dump] device={DEVICE} n={len(examples)} models={list(MODELS)} mode={mode} -> {OUT_DIR}")

for name, path in MODELS.items():
    if not os.path.isdir(path):
        print(f"[dump] skip {name}: {path} missing"); continue
    print(f"[dump] {name}: loading {path}")
    model = AutoModelForMaskedLM.from_pretrained(path).to(DEVICE).eval()
    out_name = f"{name}+{TAG}" if TAG else name
    if FIXED_STEPS:
        for k in FIXED_STEPS:
            dump_one(model, out_name, examples, conf_stop=None, n_steps=k, suffix=f"_k{k}")
    elif CONF_SWEEP:
        for cs in CONF_SWEEP:
            dump_one(model, out_name, examples, conf_stop=cs, n_steps=GEN_STEPS, suffix=f"_c{cs}")
    else:
        dump_one(model, out_name, examples, conf_stop=CONF_STOP, n_steps=GEN_STEPS)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("[dump] done")
