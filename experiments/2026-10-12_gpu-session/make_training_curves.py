"""Accuracy per epoch during training: Ettin-150M diffusion vs. Ettin-150M AR (same pretraining, same SQL data).

    python experiments/2026-10-12_gpu-session/make_training_curves.py [path/to/logs]

Training-time evals come from the training logs (every 500 steps, 32 held-out queries, exact match) and are cached in
data/training_evals.json; benchmark markers are the 256-query results in data/cpu (PyTorch fp32).
"""

import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4"
COLOR = {"diffusion": "#2a78d6", "ar": "#eb6834"}
RUNS = {  # log file: (arm, steps per epoch, first epoch)
    "train_ettin150-diffusion.log": ("diffusion", 100_000 / 128, 0),
    "train_ettin150-diffusion-iso-10ep.log": ("diffusion", 100_000 / 128, 3),
    "train_ettin150-ar.log": ("ar", 100_000 / 64, 0),
    "train_ettin150-ar-10ep.log": ("ar", 100_000 / 64, 3),
}
BENCH = {  # arm: {epoch: summary}
    "diffusion": {3: "ettin150-diffusion-torch-w64", 10: "diffusion-iso-10ep-torch"},
    "ar": {3: "ettin150-ar-torch", 10: "ar-10ep-torch"},
}
EVAL_EVERY = 500


def parse_logs(log_dir):
    points = {"diffusion": [], "ar": []}
    for name, (arm, steps_per_epoch, start) in RUNS.items():
        text = open(os.path.join(log_dir, name)).read().replace("\r", "\n")
        values = [float(v) for v in re.findall(r"\[gen-eval\] (?:exact_match|em)=([0-9.]+)", text)]
        points[arm] += [(start + (k + 1) * EVAL_EVERY / steps_per_epoch, v) for k, v in enumerate(values)]
    return points


def per_epoch(points):
    """Mean of the evals that fall inside each epoch k (k-1, k]."""
    bins = {}
    for epoch, value in points:
        bins.setdefault(math.ceil(epoch), []).append(value)
    return sorted((k, sum(v) / len(v)) for k, v in bins.items())


def main():
    cache = os.path.join(HERE, "data", "training_evals.json")
    if len(sys.argv) > 1:
        json.dump(parse_logs(sys.argv[1]), open(cache, "w"), indent=1)
    points = json.load(open(cache))

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = {"diffusion": "Ettin-150M diffusion", "ar": "Ettin-150M AR"}
    for arm in ("ar", "diffusion"):
        raw = points[arm]
        ax.scatter([e for e, _ in raw], [v * 100 for _, v in raw], s=16, color=COLOR[arm], alpha=0.28,
                   linewidth=0, zorder=2)
        curve = per_epoch(raw)
        ax.plot([k for k, _ in curve], [v * 100 for _, v in curve], color=COLOR[arm], linewidth=2, zorder=3,
                solid_capstyle="round")
        ax.scatter([k for k, _ in curve], [v * 100 for _, v in curve], s=34, color=COLOR[arm],
                   edgecolor=SURFACE, linewidth=2, zorder=4)
        k_last, v_last = curve[-1]
        ax.annotate(names[arm], (k_last, v_last * 100), xytext=(16, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
        for epoch, name in BENCH[arm].items():
            s = json.load(open(os.path.join(HERE, "data", "cpu", f"{name}.summary.json")))
            x = epoch + (0.12 if arm == "diffusion" else -0.12)  # side by side where both arms are measured
            ax.scatter(x, s["exact"] * 100, s=95, marker="D", color=COLOR[arm], edgecolor=SURFACE, linewidth=2,
                       zorder=5)
            ax.annotate(f"{s['exact'] * 100:.1f}", (x, s["exact"] * 100), fontsize=8, color=INK_2, va="center",
                        xytext=(9, 0) if arm == "diffusion" else (-9, 0), textcoords="offset points",
                        ha="left" if arm == "diffusion" else "right")
    ax.axvline(3, color=MUTED, linewidth=1, zorder=1)
    ax.text(3.08, 45.5, "both continued to 10 epochs\n(diffusion: + prompt mask)", fontsize=8, color=MUTED, va="top",
            linespacing=1.3)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(10, 47)
    ax.set_xticks(range(0, 11))
    ax.set_xlabel("epoch", fontsize=9, color=INK_2)
    ax.set_ylabel("exact match (%)", fontsize=9, color=INK_2)
    ax.grid(True, color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    handles = [
        plt.Line2D([], [], color=MUTED, linewidth=2, marker="o", markersize=5, label="mean per epoch"),
        plt.Line2D([], [], color=MUTED, linestyle="", marker="o", markersize=4, alpha=0.4,
                   label="single eval (32 queries)"),
        plt.Line2D([], [], color=MUTED, linestyle="", marker="D", markersize=7, label="benchmark (256 queries)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Diffusion keeps improving with more epochs; AR plateaus after ~2", x=0.012, ha="left",
                 fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.012, 0.925, "Ettin-150M encoder (diffusion) vs. decoder (AR): same pretraining, same 100k text-to-SQL "
             "examples.\nLines are noisy 32-query evals during training; diamonds are the 256-query benchmark.",
             fontsize=9, color=INK_2, va="top", linespacing=1.4)
    fig.text(0.012, 0.015, "Training-time evals: diffusion decodes with a fixed 24 steps over a 128-token window, AR "
             "greedy. Benchmark points use the final decoding setup (64-token window, early stop).",
             fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
    out = os.path.join(HERE, "plots", "training_curves.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
