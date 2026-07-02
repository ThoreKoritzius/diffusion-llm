"""Semantic fixed-NFE curve: re-grade the fixed-step-budget predictions with the
GPT-5.4-mini judge, so the "did the finetune get worse at fixed NFE?" question is
answered on semantic equivalence instead of brittle exact-match.

Inputs:  <pred_dir>/pred_<model>_k<K>.jsonl   (from dump_predictions.py FIXED_STEPS=...)
Outputs: <pred_dir>/graded_<model>_k<K>.jsonl
         data/judge_nfe_summary.json
         plots/judge_nfe.png  (semantic vs exact accuracy per fixed step budget K)

Requires OPENAI_API_KEY. Reuses the judge from judge_sql.py.

Usage:  OPENAI_API_KEY=... python3 judge_nfe.py <pred_dir>
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_sql import grade_one, JUDGE_MODEL  # noqa: E402 — reuse the same judge

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "predictions_nfe")
DATA_DIR = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "8"))

FNAME = re.compile(r"pred_(?P<model>.+)_k(?P<k>\d+)\.jsonl$")
ORDER = ["base", "tau_0", "tau_0.3"]
STYLE = {"base": ("base", "#8c8c8c", "o"), "tau_0": ("τ=0", "#ff7f0e", "s"),
         "tau_0.3": ("τ=0.3", "#1f77b4", "^")}


def grade_file(path):
    rows = [json.loads(l) for l in open(path)]
    graded = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(grade_one, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            graded[futs[fut]] = fut.result()
    with open(path.replace("pred_", "graded_"), "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    valid = [g for g in graded if g["verdict"] != "error"]
    n = len(graded)
    sem = sum(g["verdict"] == "equivalent" for g in valid) / len(valid) if valid else 0.0
    exact = sum(g["exact_match"] for g in graded) / n if n else 0.0
    return n, sem, exact


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*_k*.jsonl")))
    if not files:
        print(f"no pred_*_k*.jsonl in {PRED_DIR} — run dump_predictions.py FIXED_STEPS=... first")
        sys.exit(1)
    print(f"[judge-nfe] model={JUDGE_MODEL} files={len(files)}")
    res = defaultdict(dict)  # res[model][K] = {n, semantic, exact}
    for path in files:
        m = FNAME.search(os.path.basename(path))
        model, k = m["model"], int(m["k"])
        n, sem, exact = grade_file(path)
        res[model][k] = {"n": n, "semantic": sem, "exact": exact}
        print(f"    {model} K={k:<3} n={n} exact={exact:.3f} semantic={sem:.3f}")

    out = {"judge_model": JUDGE_MODEL, "results": res}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "judge_nfe_summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    # Plot: semantic (solid) vs exact (dashed) per model, x = fixed step budget K
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ORDER:
        if model not in res:
            continue
        ks = sorted(res[model])
        lab, col, mk = STYLE[model]
        ax.plot(ks, [res[model][k]["semantic"] for k in ks], color=col, marker=mk,
                linewidth=2, label=f"{lab} (semantic)")
        ax.plot(ks, [res[model][k]["exact"] for k in ks], color=col, marker=mk,
                linewidth=1, linestyle="--", alpha=0.5, label=f"{lab} (exact)")
    n_any = next(iter(next(iter(res.values())).values()))["n"]
    ax.set_xlabel("fixed step budget K (no early stop)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Fixed-NFE curve — semantic (solid) vs exact (dashed), n={n_any}\n"
                 f"judge: {JUDGE_MODEL}")
    ax.set_xticks(ks)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "judge_nfe.png"))
    print(f"[judge-nfe] wrote {os.path.join(PLOTS, 'judge_nfe.png')} and data/judge_nfe_summary.json")


if __name__ == "__main__":
    main()
