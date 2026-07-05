"""Evidence for the pass@k / distribution-collapse finding (Follow-up 2).

Two measurements on the base checkpoint, archived to data/passk_probe.json and
plotted to plots/collapse.png:

1. PEAKEDNESS — top-1 token probability at every window position of the
   fully-masked FIRST denoising step (the model's highest-entropy state),
   over N examples. If the mass sits at ~1.0, sampling has nothing to flip.
2. DIVERSITY — k seeded stochastic rollouts per example at several token
   temperatures, compared to the greedy rollout. If rollouts are string-
   identical, then oracle pass@k == pass@1 for ANY grading metric — no judge
   needed to close the question.

Self-contained on purpose (no dump_predictions import: that module executes a
full dump at import time). Run from anywhere:
    python3 passk_probe.py [N_examples]
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from denoising import denoise_steps  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
K = 3                      # samples per temperature
TEMPS = [0.7, 1.5, 3.0]
STEPS = 12                 # fixed budget (~ the conf_stop=0.9 sweet spot)
MAX_LEN, SQL_WINDOW = 512, 128
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

BASE = os.path.join(ROOT, "diffusion-sql-modernbert")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForMaskedLM.from_pretrained(BASE).to(DEVICE).eval()
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id


def encode(prompt, context, sql):
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(context, add_special_tokens=False)["input_ids"]
    s = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    s = s + [PAD_ID] * (SQL_WINDOW - len(s))
    budget = MAX_LEN - SQL_WINDOW - 9
    if len(p) + len(c) > budget:
        c = c[: max(0, budget - len(p))]
        p = p[:budget]
    ids = ([CLS_ID, TAG_IDS['<PROMPT>']] + p + [TAG_IDS['</PROMPT>'],
           TAG_IDS['<CONTEXT>']] + c + [TAG_IDS['</CONTEXT>'], TAG_IDS['<SQL>']])
    lo = len(ids)
    ids += s + [TAG_IDS['</SQL>'], SEP_ID]
    hi = lo + SQL_WINDOW
    attn = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids, attn, lo, hi


ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
examples = [ds[i] for i in range(N)]
print(f"[probe] device={DEVICE} n={N} k={K} temps={TEMPS} steps={STEPS}", flush=True)

# --- 1. peakedness at the fully-masked first step -------------------------
top1 = []
with torch.no_grad():
    for ex in examples:
        ids, attn, lo, hi = encode(ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", ""))
        t_ids = torch.tensor([ids], device=DEVICE)
        t_attn = torch.tensor([attn], device=DEVICE)
        t_ids[0, lo:hi] = MASK_ID
        logits = model(input_ids=t_ids, attention_mask=t_attn).logits[0, lo:hi, :].float()
        for t in TAG_IDS.values():
            logits[:, t] = -float("inf")
        top1.append(torch.softmax(logits, -1).max(-1).values.cpu())
top1 = torch.cat(top1).numpy()
pk = {f"p{q}": float(np.percentile(top1, q)) for q in (5, 10, 25, 50, 75, 90)}
pk["share_gt_0.99"] = float((top1 > 0.99).mean())
pk["share_gt_0.9"] = float((top1 > 0.9).mean())
pk["n_positions"] = int(top1.size)
print(f"[probe] peakedness: median={pk['p50']:.4f} >0.99={pk['share_gt_0.99']:.1%}", flush=True)


# --- 2. rollout diversity vs temperature ----------------------------------
def rollout(ex, token_temp, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids, attn, lo, hi = encode(ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", ""))
    t_ids = torch.tensor([ids], device=DEVICE)
    t_attn = torch.tensor([attn], device=DEVICE)
    t_ids[0, lo:hi] = MASK_ID
    for _ in denoise_steps(model, t_ids, t_attn, list(range(lo, hi)), MASK_ID,
                           n_steps=STEPS, forbid_token_ids=list(TAG_IDS.values()),
                           confidence_stop=None, token_temperature=token_temp):
        if (t_ids[0, lo:hi] == MASK_ID).sum().item() == 0:
            break
    out = [t for t in t_ids[0, lo:hi].tolist() if t not in (PAD_ID, MASK_ID)]
    return tokenizer.decode(out, skip_special_tokens=True)


div = {}
greedy = [rollout(ex, 0.0) for ex in examples]
print("[probe] greedy rollouts done", flush=True)
for T in TEMPS:
    samples = [[rollout(ex, T, seed=1000 + s) for s in range(K)] for ex in examples]
    all_identical = float(np.mean([len(set(ss)) == 1 for ss in samples]))
    eq_greedy = float(np.mean([all(s == g for s in ss) for ss, g in zip(samples, greedy)]))
    mean_distinct = float(np.mean([len(set(ss)) for ss in samples]))
    div[str(T)] = {"all_k_identical": all_identical, "all_eq_greedy": eq_greedy,
                   "mean_distinct_of_k": mean_distinct}
    print(f"[probe] T={T}: identical={all_identical:.1%} ==greedy={eq_greedy:.1%} "
          f"distinct/k={mean_distinct:.2f}", flush=True)

out = {"n_examples": N, "k": K, "steps": STEPS, "temps": TEMPS,
       "peakedness_step1": pk, "diversity": div}
with open(os.path.join(HERE, "data", "passk_probe.json"), "w") as f:
    json.dump(out, f, indent=2)

# --- figure: CDF of step-1 top-1 probability -------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(figsize=(7, 4.8))
xs = np.sort(top1)
ax.plot(xs, np.arange(1, xs.size + 1) / xs.size, color="#1f77b4", linewidth=2)
ax.axvline(0.9, color="0.5", linestyle=":", linewidth=1.2)
below = float((top1 <= 0.9).mean())
ax.annotate(f"only {below:.0%} of positions below 0.9\n(the only places sampling can act)",
            xy=(0.9, below), xytext=(0.34, 0.55), fontsize=9, color="0.35",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=1))
ax.annotate(f"median = {pk['p50']:.3f}", xy=(pk["p50"], 0.5),
            xytext=(0.55, 0.32), fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
ax.set_xlabel("top-1 token probability at the fully-masked first step")
ax.set_ylabel("fraction of positions ≤ x  (CDF)")
ax.set_title(f"Distribution collapse: step-1 confidence across {pk['n_positions']:,} positions "
             f"({N} examples)\n{pk['share_gt_0.99']:.0%} of positions exceed 0.99 "
             "before any token is revealed", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "collapse.png"))
print(f"[probe] wrote data/passk_probe.json and plots/collapse.png", flush=True)
