"""Build the final, coherent figure set for the report from the judge summaries.

Semantic accuracy (GPT-5.4-mini judge) is the primary metric everywhere; exact
string-match appears only in the quality-reframe panel as the contrast. Produces
five files in plots/ and retires the older ad-hoc plots.

Reads:  data/judge_summary.json          (early-stop headline + failure modes)
        data/judge_nfe_summary.json       (fixed-NFE semantic curve)
        data/judge_frontier_summary.json  (confidence_stop frontier, semantic)
Writes: plots/quality.png  failures.png  frontier.png  nfe.png  summary.png

Run:  python3 make_report_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)

JS = json.load(open(os.path.join(D, "judge_summary.json")))["models"]
NFE = json.load(open(os.path.join(D, "judge_nfe_summary.json")))
FR = json.load(open(os.path.join(D, "judge_frontier_summary.json")))

ORDER = ["base", "tau_0", "tau_0.3"]
LAB = {"base": "base", "tau_0": "τ=0", "tau_0.3": "τ=0.3"}
COL = {"base": "#8c8c8c", "tau_0": "#ff7f0e", "tau_0.3": "#1f77b4"}
MK = {"base": "o", "tau_0": "s", "tau_0.3": "^"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def quality(ax):
    x = np.arange(len(ORDER)); w = 0.38
    ex = [JS[m]["exact_match_acc"] for m in ORDER]
    se = [JS[m]["semantic_acc"] for m in ORDER]
    ax.bar(x - w/2, ex, w, label="exact match (string)", color="#cccccc")
    ax.bar(x + w/2, se, w, label="semantic (judge)", color="#1f77b4")
    for i, (e, s) in enumerate(zip(ex, se)):
        ax.text(i - w/2, e + .005, f"{e:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, s + .005, f"{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in ORDER]); ax.set_ylim(0, 0.6)
    ax.set_ylabel("accuracy")
    ax.set_title("Real quality ≈ 47%, not ≈ 29%\nexact-match undercounts correct SQL (~60% relative)")
    ax.legend(fontsize=8)


def failures(ax):
    cats = sorted({c for m in ORDER for c in JS[m]["failure_histogram"]},
                  key=lambda c: -JS["base"]["failure_histogram"].get(c, 0))
    x = np.arange(len(cats)); w = 0.26
    for j, m in enumerate(ORDER):
        vals = [JS[m]["failure_histogram"].get(c, 0)/JS[m]["n"] for c in cats]
        ax.bar(x + (j-1)*w, vals, w, label=LAB[m], color=COL[m])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("share of predictions")
    ax.set_title("Failure modes — syntax errors dominate (~14%)\nnearly identical across models")
    ax.legend(fontsize=8)


def _failure_hist(graded_path):
    """(correct_share, {failure_category: share}, n) from a graded_*.jsonl."""
    rows = [json.loads(l) for l in open(graded_path)]
    n = len(rows)
    correct = sum(r["verdict"] == "equivalent" for r in rows) / n
    hist = {}
    for r in rows:
        if r["verdict"] not in ("equivalent", "error"):
            hist[r["failure_category"]] = hist.get(r["failure_category"], 0) + 1
    return correct, {k: v / n for k, v in hist.items()}, n


def repair_effect(ax):
    """Standalone follow-up figure (NOT part of the PAPL A/B board): failure
    modes before/after the sqlglot verify-repair loop, same 256 examples.
    Failures only; the (near-zero) change in correct answers goes in a note."""
    inf = os.path.join(HERE, "predictions_inference")
    c0, h0, n = _failure_hist(os.path.join(inf, "graded_base+none_c0.9.jsonl"))
    c1, h1, _ = _failure_hist(os.path.join(inf, "graded_base+repair_c0.9.jsonl"))
    cats = sorted(set(h0) | set(h1), key=lambda c: -h0.get(c, 0))
    x = np.arange(len(cats)); w = 0.38
    v0 = [h0.get(c, 0) for c in cats]
    v1 = [h1.get(c, 0) for c in cats]
    ax.bar(x - w/2, v0, w, label="base", color=COL["base"])
    ax.bar(x + w/2, v1, w, label="base + repair", color="#7b52ab")
    for i, (a, b) in enumerate(zip(v0, v1)):
        if abs(b - a) >= 0.015:
            ax.annotate(f"{(b-a)*100:+.1f}pt", (i + w/2, b), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5, color="#7b52ab",
                        fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("share of predictions")
    ax.set_title("Verify-repair moves failures between buckets, not out of them\n"
                 f"(same {n} examples; correct answers: {c0:.1%} → {c1:.1%})", fontsize=10)
    ax.legend(fontsize=8)


THRESH = sorted({float(c) for m in ORDER for c in FR[m]})
def _fr(m, t):
    return FR[m][f"{t:g}"]


def _repair_frontier():
    """(steps, semantic) per threshold for the base+repair config, from the
    graded sweep files in predictions_inference/. None if incomplete."""
    import glob as _glob
    import re as _re
    pts = {}
    pat = _re.compile(r"graded_base\+repair_c([0-9.]+)\.jsonl$")
    for p in _glob.glob(os.path.join(HERE, "predictions_inference",
                                     "graded_base+repair_c*.jsonl")):
        cs = float(pat.search(os.path.basename(p)).group(1))
        rows = [json.loads(l) for l in open(p)]
        valid = [r for r in rows if r["verdict"] != "error"]
        pts[cs] = {
            "avg_steps": sum(r.get("steps", 0) for r in rows) / len(rows),
            "semantic": sum(r["verdict"] == "equivalent" for r in valid) / len(valid),
        }
    return pts if len(pts) == len(THRESH) else None


def frontier(ax):
    """Cleveland dot plot over the early-stop settings.

    Models are only at-a-glance comparable when their marks share a common
    position — so the x-axis is the 4 confidence_stop settings (the knob),
    with the resulting step cost folded into the tick labels (steps are ~a
    function of the threshold; per-model deltas go in the subtitle). Three
    dodged dots per setting: same height = models tied. The hump across
    settings = over-decoding. Green column = recommended setting.
    """
    G = "#2ca02c"
    n = 256

    def xy(m):
        return ([_fr(m, t)["avg_steps"] for t in THRESH],
                [_fr(m, t)["semantic"] for t in THRESH])

    # light ties joining the same-threshold points across models, labeled once
    for t in THRESH:
        px = [_fr(m, t)["avg_steps"] for m in ORDER]
        py = [_fr(m, t)["semantic"] for m in ORDER]
        order = np.argsort(px)
        ax.plot(np.array(px)[order], np.array(py)[order], color="0.78",
                linewidth=1.1, zorder=1)
        peak = (t == 0.9)
        ax.text(float(np.mean(px)), max(py) + 0.007,
                f"{t:g}" + (" · peak" if peak else ""), ha="center", fontsize=10,
                fontweight="bold", color=G if peak else "0.45")

    # the comparison: base (grey reference) vs τ=0.3 (blue); τ=0 as thin context
    bx, by = xy("base")
    zx, zy = xy("tau_0")
    px_, py_ = xy("tau_0.3")
    ax.plot(zx, zy, color=COL["tau_0"], linewidth=1.2, linestyle="--", alpha=0.65,
            marker=MK["tau_0"], markersize=4.5, zorder=2, label="τ=0 (control)")
    ax.plot(bx, by, color=COL["base"], linewidth=2.4, marker=MK["base"],
            markersize=7, zorder=3, label="base")
    ax.plot(px_, py_, color=COL["tau_0.3"], linewidth=2.4, marker=MK["tau_0.3"],
            markersize=7, zorder=4, label="τ=0.3 (PAPL)")
    rep = _repair_frontier()
    if rep:
        rx = [rep[t]["avg_steps"] for t in THRESH]
        ry = [rep[t]["semantic"] for t in THRESH]
        ax.plot(rx, ry, color="#7b52ab", linewidth=1.6, linestyle="-.",
                marker="D", markersize=5, zorder=4, label="base + repair")

    # three annotations, one per takeaway
    ax.text(14.9, 0.5125, "← τ=0.3 tracks base shifted left:", fontsize=9,
            color=COL["tau_0.3"], ha="left", fontweight="bold")
    ax.text(14.9, 0.5045, "same accuracy, ≈7% fewer steps", fontsize=9,
            color=COL["tau_0.3"], ha="left")
    ax.text(16.3, 0.4755, "over-decoding: accuracy falls\nwhile cost keeps rising",
            fontsize=8.5, color="0.4", style="italic", ha="center")
    ax.errorbar([7.85], [0.42], yerr=[0.031], fmt="none", ecolor="0.5",
                elinewidth=1.2, capsize=3)
    ax.text(8.05, 0.42, "±1 s.e.", fontsize=7.5, color="0.4", va="center")

    ax.set_xlim(7.5, 20.6)
    ax.set_ylim(0.386, 0.527)
    ax.set_xlabel("avg denoising steps per query  (← cheaper)")
    ax.set_ylabel("semantic accuracy (LLM judge, n=256)")
    ax.set_title("PAPL shifts the curve left, not up: same accuracy, ≈7% fewer steps\n"
                 "ties join equal-threshold points · peak at 0.9 — past it, more steps lose accuracy",
                 fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    want = [w for w in ["base", "τ=0.3 (PAPL)", "τ=0 (control)", "base + repair"]
            if w in labels]
    hl = {l: h for h, l in zip(handles, labels)}
    ax.legend([hl[w] for w in want], want, loc="upper left", fontsize=8.5,
              framealpha=0.9)


def nfe(ax):
    for m in ORDER:
        ks = sorted(int(k) for k in NFE[m])
        ys = [NFE[m][str(k)]["semantic"] for k in ks]
        ax.plot(ks, ys, color=COL[m], marker=MK[m], linewidth=2, markersize=7, label=LAB[m])
    ax.set_xticks(ks); ax.set_xlabel("fixed step budget K (no early stop)")
    ax.set_ylabel("semantic accuracy")
    ax.set_title("Quality per compute (fixed-NFE)\nmore steps genuinely help; models tied")
    ax.legend(loc="upper left", fontsize=8)


# standalone single-axis figures
for fn, draw in [("quality", quality), ("failures", failures), ("frontier", frontier),
                 ("nfe", nfe), ("repair_effect", repair_effect)]:
    fig, ax = plt.subplots(figsize=(7.4, 5.4)); draw(ax); fig.tight_layout()
    fig.savefig(os.path.join(P, f"{fn}.png")); plt.close(fig)

# 2x2 board
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
quality(axes[0, 0]); failures(axes[0, 1]); frontier(axes[1, 0]); nfe(axes[1, 1])
fig.suptitle("PAPL A/B — semantic evaluation (GPT-5.4-mini judge, n=256) · base vs τ=0 vs τ=0.3",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(P, "summary.png")); plt.close(fig)

# retire the older ad-hoc plots so the archive isn't confusing
for old in ["nfe_curve.png", "headline_steps.png", "judge.png", "judge_nfe.png", "frontier_semantic.png"]:
    op = os.path.join(P, old)
    if os.path.exists(op):
        os.remove(op); print(f"retired {old}")

print("wrote:", ", ".join(sorted(f for f in os.listdir(P) if f.endswith(".png"))))
