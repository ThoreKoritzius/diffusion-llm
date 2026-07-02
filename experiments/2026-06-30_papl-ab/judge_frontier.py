"""Semantic speed/quality frontier: grade the confidence_stop sweep with the
GPT-5.4-mini judge and plot semantic accuracy vs avg denoising steps per model.
The early-stop analogue of judge_nfe.py.

Inputs:  <pred_dir>/pred_<model>_c<cs>.jsonl  (dump_predictions.py CONF_SWEEP=...)
Outputs: <pred_dir>/graded_<model>_c<cs>.jsonl
         data/judge_frontier_summary.json
         plots/frontier_semantic.png

Requires OPENAI_API_KEY. Usage:  OPENAI_API_KEY=... python3 judge_frontier.py <pred_dir>
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
from judge_sql import grade_one, JUDGE_MODEL  # noqa: E402

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "predictions_frontier")
DATA_DIR = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "8"))

FNAME = re.compile(r"pred_(?P<model>.+)_c(?P<cs>[0-9.]+)\.jsonl$")
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
    avg_steps = sum(g.get("steps", 0) for g in graded) / n if n else 0.0
    return n, sem, exact, avg_steps


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*_c*.jsonl")))
    if not files:
        print(f"no pred_*_c*.jsonl in {PRED_DIR} — run dump_predictions.py CONF_SWEEP=... first")
        sys.exit(1)
    print(f"[judge-frontier] model={JUDGE_MODEL} files={len(files)}")
    res = defaultdict(dict)  # res[model][cs] = {n, semantic, exact, avg_steps}
    for path in files:
        m = FNAME.search(os.path.basename(path))
        model, cs = m["model"], float(m["cs"])
        n, sem, exact, avg_steps = grade_file(path)
        res[model][cs] = {"n": n, "semantic": sem, "exact": exact, "avg_steps": avg_steps}
        print(f"    {model} cs={cs:<5} n={n} steps={avg_steps:.2f} exact={exact:.3f} semantic={sem:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "judge_frontier_summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ORDER:
        if model not in res:
            continue
        pts = sorted(res[model].items())  # by cs -> increasing steps
        lab, col, mk = STYLE[model]
        xs = [d["avg_steps"] for _, d in pts]
        ys = [d["semantic"] for _, d in pts]
        ax.plot(xs, ys, color=col, marker=mk, linewidth=2, markersize=7, label=lab)
        for cs, d in pts:
            ax.annotate(f"{cs:g}", (d["avg_steps"], d["semantic"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color=col)
    n_any = next(iter(next(iter(res.values())).values()))["n"]
    ax.set_xlabel("avg denoising steps  (← faster)")
    ax.set_ylabel("semantic accuracy  (↑ better)")
    ax.set_title(f"Speed/quality frontier — semantic, early-stop sweep (n={n_any})\n"
                 f"labels = confidence_stop threshold · judge: {JUDGE_MODEL}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "frontier_semantic.png"))
    print(f"[judge-frontier] wrote {os.path.join(PLOTS, 'frontier_semantic.png')} "
          f"and data/judge_frontier_summary.json")


if __name__ == "__main__":
    main()
