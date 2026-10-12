"""Headline chart: best AR vs. best diffusion, accuracy vs. speed, GPU and CPU side by side.

    python experiments/2026-10-12_gpu-session/make_headline_chart.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Reference palette (validated): slot 1 blue, slot 2 orange. Text uses text tokens, never series colors.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4"
COLOR = {"diffusion": "#2a78d6", "ar": "#eb6834"}

# (label, arm, size, summary) -- the best configuration per model on each hardware target
GPU = [
    ("Qwen2.5-0.5B AR\nvLLM", "ar", "494M", "sql-ar-qwen25-coder-0.5b-ar-vllm"),
    ("Ettin-150M AR\nHF compiled", "ar", "150M", "ettin150-ar-10ep-bf16-compiled"),
    ("SmolLM2-135M AR\nvLLM", "ar", "150M", "smollm135-ar-vllm"),
    ("ModernBERT-150M diffusion\nCUDA graphs", "diffusion", "150M", "modernbert-isolated-distilled-bf16-cudagraphs"),
    ("Ettin-150M diffusion\nCUDA graphs", "diffusion", "150M", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs"),
]
CPU = [
    ("Qwen2.5-0.5B AR\nllama.cpp Q8", "ar", "494M", "qwen25-ar-llamacpp-q8_0"),
    ("Ettin-150M AR\nPyTorch", "ar", "150M", "ar-10ep-torch"),
    ("SmolLM2-135M AR\nllama.cpp Q8", "ar", "150M", "smollm135-ar-llamacpp-q8_0"),
    ("ModernBERT-150M diffusion\nONNX + cache, int8", "diffusion", "150M", "modernbert-isolated-distilled-onnx-cached-int8",
     (9, -6, "top")),
    ("Ettin-150M diffusion\nONNX + cache, int8", "diffusion", "150M", "ettin150-iso-10ep-distilled-onnx-cached-int8",
     (9, -6, "top")),
]
CPU_BEFORE = ("playground today", "diffusion-onnx-w128-s16-c0.9")


def load(kind, name):
    path = os.path.join(HERE, "data", kind, f"{name}.summary.json")
    return json.load(open(path)) if os.path.exists(path) else None


def panel(ax, kind, rows, title, ticks, before=None):
    ax.set_facecolor(SURFACE)
    for row in rows:
        label, arm, size, name = row[:4]
        dx, dy, va = row[4] if len(row) > 4 else (9, 0, "center")
        s = load(kind, name)
        if s is None:
            continue
        x, y = s["lat_mean_ms"], s["exec"] * 100
        ax.scatter(x, y, s=150 if size == "494M" else 110, marker="s" if size == "494M" else "o",
                   color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", va=va, fontsize=8.2,
                    color=INK_2, linespacing=1.25)
    if before is not None:
        b = load(kind, before[1])
        best = load(kind, rows[3][3])
        ax.scatter(b["lat_mean_ms"], b["exec"] * 100, s=90, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=3)
        ax.annotate(before[0], (b["lat_mean_ms"], b["exec"] * 100), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)
        ax.annotate("", xy=(best["lat_mean_ms"] * 1.06, best["exec"] * 100 + 0.5),
                    xytext=(b["lat_mean_ms"] * 0.97, b["exec"] * 100 + 0.8),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=4, shrinkB=6,
                                    connectionstyle="arc3,rad=0.32"))
        ax.text((b["lat_mean_ms"] * best["lat_mean_ms"]) ** 0.5, (b["exec"] + best["exec"]) * 50 + 8.0,
                f"{b['lat_mean_ms'] / best['lat_mean_ms']:.0f}× faster than today,\n+{(best['exec'] - b['exec']) * 100:.1f} pts accuracy",
                ha="center", fontsize=8, color=MUTED, linespacing=1.3)
    ax.set_xscale("log")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10, fontweight="bold")
    ax.set_xlabel("mean latency per query (ms, log scale)  ← faster", fontsize=9, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=1, linestyle="-")
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, which="both", length=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE, sharey=True)
    panel(axes[0], "gpu", GPU, "GPU · NVIDIA H100, batch 1", [20, 50, 100, 200, 500])
    panel(axes[1], "cpu", CPU, "CPU · 3 threads, batch 1", [200, 300, 500, 1000], before=CPU_BEFORE)
    axes[0].set_xlim(15, 600)
    axes[1].set_xlim(150, 1600)
    axes[0].set_ylim(45, 77)
    axes[0].set_ylabel("execution accuracy (%)", fontsize=9, color=INK_2)
    for ax in axes:
        ax.text(0.02, 0.97, "better ↖", transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["diffusion"], label="diffusion"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["ar"], label="autoregressive (AR)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED, label="~150M params"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=MUTED, label="494M params"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.006, 0.925))
    fig.suptitle("Best autoregressive vs. best diffusion model: accuracy vs. speed (text-to-SQL)",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.012, 0.015, "Same 256 gretelai test queries for every point. Each point is a model on its fastest "
             "engine. The larger Qwen2.5-0.5B AR (squares) is the accuracy bar; matched-size AR models are circles.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = os.path.join(HERE, "plots", "headline.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
