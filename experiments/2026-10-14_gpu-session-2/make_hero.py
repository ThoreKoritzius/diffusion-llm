"""Hero chart (accuracy vs. speed), every point from one H100 PCIe, batch 1, the same 1,000 test queries:
- the highlighted pair is Ettin-150M decoder (AR, vLLM) vs. encoder (diffusion, CUDA graphs): same size, same
  pretraining data, same SQL data, both on their best engine;
- the second arrow is the same Qwen2.5-0.5B with a diffusion drafter (DFlash) for speculative decoding;
- EAGLE-3 (an autoregressive drafter) and Ettin-400M diffusion are shown for context.

    python experiments/2026-10-14_gpu-session-2/make_hero.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-12_gpu-session"))
from make_headline_chart import COLOR, GRID, INK, INK_2, MUTED, SURFACE  # noqa: E402

HERO_AR, HERO_DIFF = "ettin150-ar-10ep-vllm-pr61276", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs"
HERO = [("Autoregressive", "Ettin-150M decoder", "ar", HERO_AR),
        ("Diffusion", "Ettin-150M encoder", "diffusion", HERO_DIFF)]
QWEN, QWEN_DFLASH = "sql-ar-qwen25-coder-0.5b-ar-vllm", "sql-ar-qwen25-coder-0.5b-ar-vllm-dflash"
# (label, arm, summary, label offset (dx, dy, ha, va))
CONTEXT = [
    ("Qwen2.5-0.5B AR", "ar", QWEN, (-12, 0, "right", "center")),
    ("+ EAGLE-3 drafter", "ar", "sql-ar-qwen25-coder-0.5b-ar-vllm-eagle3", (0, -12, "center", "top")),
    ("+ diffusion drafter\n(DFlash)", "diffusion", QWEN_DFLASH, (0, 12, "center", "bottom")),
    ("Ettin-400M diffusion", "diffusion", "ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (12, 0, "left", "center")),
]
REPO = "github.com/ThoreKoritzius/diffusion-llm"
LEFT_X = 0.06  # shared left edge of title, subtitle and footer


def point(name):
    r = json.load(open(os.path.join(HERE, "data", "bench", f"{name}.summary.json")))
    return r["lat_mean_ms"], r["exec"] * 100


def arrow(ax, a, b, lw):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw, shrinkA=16, shrinkB=16, mutation_scale=18))


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(110, 17)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(44, 80)

    for label, arm, name, (dx, dy, ha, va) in CONTEXT:
        x, y = point(name)
        ax.scatter(x, y, s=140, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9, color=INK_2, linespacing=1.4)

    (xq, yq), (xf, yf) = point(QWEN), point(QWEN_DFLASH)
    arrow(ax, (xq, yq), (xf, yf), 1.4)
    ax.annotate(f"{xq / xf:.1f}× faster", ((xq * xf) ** 0.5, yq), xytext=(-14, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)

    (xa, ya), (xd, yd) = point(HERO_AR), point(HERO_DIFF)
    arrow(ax, (xa, ya), (xd, yd), 2)
    mid = ((xa * xd) ** 0.5, (ya + yd) / 2)
    ax.annotate(f"{xa / xd:.1f}× faster", mid, xytext=(0, -16), textcoords="offset points", ha="center",
                va="top", fontsize=19, fontweight="bold", color=INK)
    ax.annotate("at tied accuracy", mid, xytext=(0, -44), textcoords="offset points", ha="center",
                va="top", fontsize=11, color=INK_2)

    for label, model, arm, name in HERO:
        x, y = point(name)
        ax.scatter(x, y, s=380, color=COLOR[arm], edgecolor=SURFACE, linewidth=3, zorder=4)
        ax.annotate(label, (x, y), xytext=(0, -20), textcoords="offset points", ha="center", va="top",
                    fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"{model}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(0, -38), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=INK_2, linespacing=1.4)

    ax.set_xticks([100, 70, 50, 30, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("latency per query  →  faster", fontsize=10, color=INK_2, labelpad=8)
    ax.set_ylabel("execution accuracy", fontsize=10, color=INK_2, labelpad=8)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0, which="both")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["diffusion"],
                          label="diffusion (model or drafter)"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["ar"], label="autoregressive")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.text(LEFT_X, 0.95, "Diffusion makes text-to-SQL faster, two ways", fontsize=18, fontweight="bold",
             color=INK, va="top")
    fig.text(LEFT_X, 0.902, f"As the model: {xa / xd:.1f}× faster at 150M  ·  as Qwen's drafter: {xq / xf:.1f}× faster",
             fontsize=11, color=INK_2, va="top")
    fig.text(LEFT_X, 0.05, "AR on vLLM, diffusion on CUDA graphs; drafters trained on Qwen's own outputs (Speculators)\n"
             "H100 PCIe, batch 1, mean latency  ·  exec accuracy on 1,000 test queries", fontsize=7.8, color=MUTED,
             va="bottom", linespacing=1.5)
    fig.text(LEFT_X, 0.018, REPO, fontsize=7.8, color=INK_2, va="bottom")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.18, top=0.85)
    out = os.path.join(HERE, "plots", "hero.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
