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

# (label, arm, size, summary[, (dx, dy, va, ha)]) -- the best configuration per model on each hardware target
LEFT = (-10, 0, "center", "right")
GPU = [
    ("Qwen2.5-0.5B AR\nvLLM", "ar", "494M", "sql-ar-qwen25-coder-0.5b-ar-vllm"),
    ("Ettin-150M AR\nHF compiled", "ar", "150M", "ettin150-ar-10ep-bf16-compiled"),
    ("SmolLM2-135M AR\nvLLM", "ar", "150M", "smollm135-ar-vllm"),
    ("ModernBERT-150M diffusion\nCUDA graphs", "diffusion", "150M", "modernbert-isolated-distilled-bf16-cudagraphs", LEFT),
    ("Ettin-150M diffusion\nCUDA graphs", "diffusion", "150M", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs", LEFT),
]
CPU = [
    ("Qwen2.5-0.5B AR\nllama.cpp Q8", "ar", "494M", "qwen25-ar-llamacpp-q8_0"),
    ("Ettin-150M AR\nPyTorch", "ar", "150M", "ar-10ep-torch"),
    ("SmolLM2-135M AR\nllama.cpp Q8", "ar", "150M", "smollm135-ar-llamacpp-q8_0"),
    ("ModernBERT-150M diffusion\nONNX + cache, int8", "diffusion", "150M", "modernbert-isolated-distilled-onnx-cached-int8", LEFT),
    ("Ettin-150M diffusion\nONNX + cache, int8", "diffusion", "150M", "ettin150-iso-10ep-distilled-onnx-cached-int8", LEFT),
]
# AR -> diffusion arrows between comparable checkpoints: solid = same size *and* same pretraining (Ettin pair),
# dashed = similar size, different pretraining.
GPU_PAIRS = [("ettin150-ar-10ep-bf16-compiled", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs", "solid"),
             ("smollm135-ar-vllm", "modernbert-isolated-distilled-bf16-cudagraphs", "dashed")]
CPU_PAIRS = [("ar-10ep-torch", "ettin150-iso-10ep-distilled-onnx-cached-int8", "solid"),
             ("smollm135-ar-llamacpp-q8_0", "modernbert-isolated-distilled-onnx-cached-int8", "dashed")]


def load(kind, name):
    path = os.path.join(HERE, "data", kind, f"{name}.summary.json")
    return json.load(open(path)) if os.path.exists(path) else None


def panel(ax, kind, rows, title, ticks, pairs):
    ax.set_facecolor(SURFACE)
    for row in rows:
        label, arm, size, name = row[:4]
        dx, dy, va, ha = row[4] if len(row) > 4 else (10, 0, "center", "left")
        s = load(kind, name)
        if s is None:
            continue
        x, y = s["lat_mean_ms"], s["exec"] * 100
        ax.scatter(x, y, s=150 if size == "494M" else 110, marker="s" if size == "494M" else "o",
                   color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", va=va, ha=ha, fontsize=8.2,
                    color=INK_2, linespacing=1.25)
    for ar_name, diff_name, style in pairs:
        a, d = load(kind, ar_name), load(kind, diff_name)
        if a is None or d is None:
            continue
        (xa, ya), (xd, yd) = (a["lat_mean_ms"], a["exec"] * 100), (d["lat_mean_ms"], d["exec"] * 100)
        ax.annotate("", xy=(xd, yd), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.3, linestyle=style, shrinkA=9, shrinkB=9,
                                    mutation_scale=12))
        speed, delta = xa / xd, yd - ya
        ax.annotate(f"{speed:.1f}× faster · {delta:+.1f} pts", ((xa * xd) ** 0.5, (ya + yd) / 2), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.2, color=INK,
                    fontweight="bold" if style == "solid" else "normal",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))
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
    panel(axes[0], "gpu", GPU, "GPU · NVIDIA H100, batch 1", [10, 20, 50, 100, 200, 500], GPU_PAIRS)
    panel(axes[1], "cpu", CPU, "CPU · 3 threads, batch 1", [50, 100, 200, 300, 500], CPU_PAIRS)
    axes[0].set_xlim(5, 700)
    axes[1].set_xlim(45, 800)
    axes[0].set_ylim(45, 77)
    axes[0].set_ylabel("execution accuracy (%)", fontsize=9, color=INK_2)
    for ax in axes:
        ax.text(0.02, 0.97, "better ↖", transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["diffusion"], label="diffusion"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["ar"], label="autoregressive (AR)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED, label="~150M params"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=MUTED, label="494M params (accuracy bar)"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.006, 0.925))
    fig.suptitle("Best autoregressive vs. best diffusion model: accuracy vs. speed (text-to-SQL)",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.012, 0.012, "Same 256 gretelai test queries for every point; each model on its fastest available engine. "
             "Arrows go from an AR model to the comparable diffusion model (speedup · change in execution accuracy).\n"
             "Solid: same size and same pretraining (Ettin decoder vs. encoder). Dashed: similar size, different "
             "pretraining. Ettin AR has no vLLM / llama.cpp support, so it runs on Hugging Face / PyTorch.",
             fontsize=7.8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    out = os.path.join(HERE, "plots", "headline.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
