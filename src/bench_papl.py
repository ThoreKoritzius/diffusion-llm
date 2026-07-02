"""
PAPL A/B benchmark: turn the single operating-point result into a frontier.

Inference-only (no Trainer / dataloaders -> no persistent-worker hang). Loads
each model in turn -- base, tau=0 control, tau=0.3 PAPL -- and runs:

  1. confidence_stop sweep {0.8,0.9,0.95,0.99}: exact_match vs avg_steps.
     Tests whether PAPL shifts the speed/quality frontier, not just one point.
  2. fixed-NFE curve {4,8,16,24} steps, early-stop OFF: accuracy per compute
     budget -- the low-step regime that matters for production latency.
  3. headline at conf_stop=0.9 on a larger eval (clean + augmented): confirms
     the ~5%/identical-accuracy isn't small-sample noise; aug probes robustness.

Writes bench_results.json + prints tables. Same eval examples across all models
(loaded once) so comparisons are apples-to-apples.

Usage:
    python3 src/bench_papl.py <ab_dir>
        <ab_dir> = the papl_ab_* run dir holding tau_0/ and tau_0.3/
"""
import contextlib
import json
import os
import random
import re
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from augment import augment_example
from denoising import denoise_steps

# Layout constants -- must match training (train.py / finetune_papl.py).
MAX_LEN = 512
SQL_WINDOW = 128

AB_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
BASE_DIR = os.environ.get("BASE_DIR", "diffusion-sql-modernbert")
MODELS = {
    "base":    BASE_DIR,
    "tau_0":   os.path.join(AB_DIR, "tau_0"),
    "tau_0.3": os.path.join(AB_DIR, "tau_0.3"),
}

N_SWEEP = int(os.environ.get("BENCH_N_SWEEP", "256"))   # #1 and #2 eval size
N_HEAD = int(os.environ.get("BENCH_N_HEAD", "512"))     # #3 eval size
CONF_STOPS = [0.8, 0.9, 0.95, 0.99]
NFE_STEPS = [4, 8, 16, 24]
HEAD_CONF = 0.9
MAX_STEPS = 24   # step budget cap for the early-stop sweeps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.environ.get("BENCH_OUT", os.path.join(AB_DIR, "bench_results.json"))

# Tokenizer is identical across all three (same base); load once so every model
# sees byte-identical inputs.
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
    ids = (
        [CLS_ID, TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
        TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'],
        TAG_IDS['<SQL>']]
    )
    sql_start = len(ids)
    ids = ids + sql_ids + [TAG_IDS['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW
    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {"input_ids": ids, "attention_mask": attention,
            "sql_start": sql_start, "sql_end": sql_end}


def normalize_sql(s):
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def gen_eval(model, examples, n_steps, conf_stop, transform=None):
    """Returns (exact_match, avg_steps) over `examples` (list of raw rows)."""
    rng = random.Random(0)
    hits, total_steps = 0, 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda" else contextlib.nullcontext()
    )
    for ex in examples:
        p, c, s = ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", "")
        if transform is not None:
            p, c, s = transform(p, c, s, rng=rng)
        enc = encode_text(p, c, s)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=DEVICE)
        lo, hi = enc["sql_start"], enc["sql_end"]
        ids[0, lo:hi] = MASK_ID
        steps_used = 0
        with autocast_ctx:
            for _ in denoise_steps(
                model, ids, attn, list(range(lo, hi)), MASK_ID,
                n_steps=n_steps, forbid_token_ids=list(TAG_IDS.values()),
                confidence_stop=conf_stop,
            ):
                steps_used += 1
                if (ids[0, lo:hi] == MASK_ID).sum().item() == 0:
                    break
        total_steps += steps_used
        out_ids = [t for t in ids[0, lo:hi].tolist() if t not in (PAD_ID, MASK_ID)]
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        if normalize_sql(pred) == normalize_sql(s):
            hits += 1
    n = max(1, len(examples))
    return hits / n, total_steps / n


# Eval examples (same filter as training: non-empty prompt), loaded once.
ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
n_needed = max(N_SWEEP, N_HEAD)
examples = [ds[i] for i in range(min(n_needed, len(ds)))]
sweep_ex = examples[:N_SWEEP]
head_ex = examples[:N_HEAD]
print(f"[bench] device={DEVICE} | sweep_n={len(sweep_ex)} head_n={len(head_ex)} "
      f"| models={list(MODELS)}")

results = {"config": {"n_sweep": len(sweep_ex), "n_head": len(head_ex),
                      "conf_stops": CONF_STOPS, "nfe_steps": NFE_STEPS,
                      "head_conf": HEAD_CONF, "max_steps": MAX_STEPS},
           "models": {}}

for name, path in MODELS.items():
    if not os.path.isdir(path):
        print(f"[bench] !! skipping {name}: {path} not found")
        continue
    print(f"\n[bench] ===== {name} ({path}) =====")
    model = AutoModelForMaskedLM.from_pretrained(path).to(DEVICE).eval()

    mres = {"conf_sweep": [], "nfe_curve": [], "headline": {}}

    print(f"[bench] {name}: confidence_stop sweep")
    for cs in CONF_STOPS:
        em, st = gen_eval(model, sweep_ex, n_steps=MAX_STEPS, conf_stop=cs)
        mres["conf_sweep"].append({"conf_stop": cs, "exact_match": em, "avg_steps": st})
        print(f"    cs={cs:<4} em={em:.3f} steps={st:.2f}")

    print(f"[bench] {name}: fixed-NFE curve (early-stop off)")
    for k in NFE_STEPS:
        em, st = gen_eval(model, sweep_ex, n_steps=k, conf_stop=None)
        mres["nfe_curve"].append({"n_steps": k, "exact_match": em, "avg_steps": st})
        print(f"    K={k:<3} em={em:.3f} steps={st:.2f}")

    print(f"[bench] {name}: headline (conf_stop={HEAD_CONF}, n={len(head_ex)})")
    em_c, st_c = gen_eval(model, head_ex, n_steps=MAX_STEPS, conf_stop=HEAD_CONF)
    em_a, st_a = gen_eval(model, head_ex, n_steps=MAX_STEPS, conf_stop=HEAD_CONF,
                          transform=augment_example)
    mres["headline"] = {"exact_match": em_c, "avg_steps": st_c,
                        "exact_match_aug": em_a, "avg_steps_aug": st_a}
    print(f"    clean em={em_c:.3f} steps={st_c:.2f} | aug em={em_a:.3f} steps={st_a:.2f}")

    results["models"][name] = mres
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[bench] wrote {OUT_JSON}")

# --- Printed comparison tables -------------------------------------------------
def fmt_delta(x, ref):
    return f"{x - ref:+.3f}" if ref is not None else "    -"

ms = results["models"]
names = [n for n in MODELS if n in ms]

print("\n================ HEADLINE (conf_stop=%.2f, n=%d) ================" % (HEAD_CONF, len(head_ex)))
print(f"{'model':>8} | {'em':>6} | {'steps':>6} | {'em_aug':>6} | {'steps_aug':>9}")
for n in names:
    h = ms[n]["headline"]
    print(f"{n:>8} | {h['exact_match']:>6.3f} | {h['avg_steps']:>6.2f} | "
          f"{h['exact_match_aug']:>6.3f} | {h['avg_steps_aug']:>9.2f}")

print("\n================ CONF_STOP SWEEP (em / steps), n=%d ================" % len(sweep_ex))
print(f"{'cs':>5} | " + " | ".join(f"{n:>14}" for n in names))
for i, cs in enumerate(CONF_STOPS):
    cells = []
    for n in names:
        r = ms[n]["conf_sweep"][i]
        cells.append(f"{r['exact_match']:.3f}/{r['avg_steps']:4.1f}")
    print(f"{cs:>5} | " + " | ".join(f"{c:>14}" for c in cells))

print("\n================ FIXED-NFE CURVE (em / steps), n=%d ================" % len(sweep_ex))
print(f"{'K':>5} | " + " | ".join(f"{n:>14}" for n in names))
for i, k in enumerate(NFE_STEPS):
    cells = []
    for n in names:
        r = ms[n]["nfe_curve"][i]
        cells.append(f"{r['exact_match']:.3f}/{r['avg_steps']:4.1f}")
    print(f"{k:>5} | " + " | ".join(f"{c:>14}" for c in cells))
print()
