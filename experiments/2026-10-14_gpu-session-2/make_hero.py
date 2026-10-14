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
QWEN, EAGLE, DFLASH = ("sql-ar-qwen25-coder-0.5b-ar-vllm", "sql-ar-qwen25-coder-0.5b-ar-vllm-eagle3",
                       "sql-ar-qwen25-coder-0.5b-ar-vllm-dflash")
DIFF_400 = "ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs"
REPO = "github.com/ThoreKoritzius/diffusion-llm"
LEFT_X = 0.06  # shared left edge of title, subtitle and footer


def point(name):
    r = json.load(open(os.path.join(HERE, "data", "bench", f"{name}.summary.json")))
    return r["lat_mean_ms"], r["exec"] * 100


def label(ax, xy, name, metrics, where, big=False):
    """Model name (bold, ink) over its numbers (regular, secondary ink), placed beside the marker."""
    fs, fs2 = (12.5, 10) if big else (9.5, 8.5)
    pad = 20 if big else 11
    step = fs + 3  # points between the two lines
    if where in ("above", "below"):
        sign, va = (1, "bottom") if where == "above" else (-1, "top")
        first, second = ((0, pad + step), (0, pad)) if where == "above" else ((0, -pad), (0, -pad - step))
        place = [(name, first, "center", va), (metrics, second, "center", va)]
    else:
        dx, ha = (pad, "left") if where == "right" else (-pad, "right")
        place = [(name, (dx, 1), ha, "bottom"), (metrics, (dx, -1), ha, "top")]
    for (text, off, ha, va), (size, weight, color) in zip(place, [(fs, "bold", INK), (fs2, "normal", INK_2)]):
        ax.annotate(text, xy, xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=size,
                    fontweight=weight, color=color)


def metrics(name):
    x, y = point(name)
    return f"{x:.0f} ms  ·  {y:.0f}% exec"


def dot(ax, name, color, size, ring=None):
    x, y = point(name)
    if ring:  # an AR model with a diffusion drafter: AR fill inside a diffusion ring
        ax.scatter(x, y, s=size * 2.1, color=ring, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(x, y, s=size * 0.75, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    else:
        ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=2.5, zorder=4)
    return x, y


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(100, 18)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(49, 82)

    # Qwen2.5-0.5B: plain -> EAGLE-3 -> diffusion drafter (same model, same outputs up to bf16 near-ties)
    xq, yq = dot(ax, QWEN, COLOR["ar"], 170)
    dot(ax, EAGLE, COLOR["ar"], 110)
    xf, yf = dot(ax, DFLASH, COLOR["ar"], 170, ring=COLOR["diffusion"])
    ax.annotate("", xy=(xf, yf), xytext=(xq, yq), zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=INK, lw=1.5, shrinkA=9, shrinkB=12, mutation_scale=14,
        connectionstyle="arc3,rad=-0.28"))
    ax.annotate(f"{xq / xf:.1f}× faster", ((xq * xf) ** 0.5, yq), xytext=(0, 50), textcoords="offset points",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
    ax.annotate("same model, diffusion drafter", ((xq * xf) ** 0.5, yq), xytext=(0, 35),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK_2)
    label(ax, (xq, yq), "Qwen2.5-0.5B AR", metrics(QWEN), "left")
    label(ax, point(EAGLE), "+ EAGLE-3 drafter", metrics(EAGLE), "below")
    label(ax, (xf, yf), "+ DFlash drafter", metrics(DFLASH), "right")

    dot(ax, DIFF_400, COLOR["diffusion"], 170)
    label(ax, point(DIFF_400), "Ettin-400M diffusion", metrics(DIFF_400), "right")

    # the matched pair
    (xa, ya), (xd, yd) = point(HERO_AR), point(HERO_DIFF)
    ax.annotate("", xy=(xd, yd), xytext=(xa, ya), zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=INK, lw=2.2, shrinkA=17, shrinkB=17, mutation_scale=20))
    mid = ((xa * xd) ** 0.5, (ya + yd) / 2)
    ax.annotate(f"{xa / xd:.1f}× faster", mid, xytext=(0, -16), textcoords="offset points", ha="center",
                va="top", fontsize=22, fontweight="bold", color=INK)
    ax.annotate("same size, tied accuracy", mid, xytext=(0, -46), textcoords="offset points", ha="center",
                va="top", fontsize=10.5, color=INK_2)
    for title, model, arm, name in HERO:
        dot(ax, name, COLOR[arm], 440)
        label(ax, point(name), title, f"{model}\n{metrics(name)}", "below", big=True)

    ax.set_xticks([100, 70, 50, 30, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("latency per query  →  faster", fontsize=10, color=INK_2, labelpad=10)
    ax.set_ylabel("execution accuracy", fontsize=10, color=INK_2, labelpad=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0, which="both", pad=6)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["diffusion"],
                          label="diffusion"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["ar"], label="autoregressive"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=COLOR["ar"],
                          markeredgecolor=COLOR["diffusion"], markeredgewidth=2.5,
                          label="autoregressive + diffusion drafter")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2, handletextpad=0.4,
              borderaxespad=0.2)

    fig.text(LEFT_X, 0.955, "Diffusion makes text-to-SQL faster, two ways", fontsize=18.5, fontweight="bold",
             color=INK, va="top")
    fig.text(LEFT_X, 0.905, f"Same-size diffusion model: {xa / xd:.1f}× faster   ·   Qwen2.5 with a diffusion drafter: {xq / xf:.1f}× faster",
             fontsize=11, color=INK_2, va="top")
    fig.text(LEFT_X, 0.05, "AR on vLLM, diffusion on CUDA graphs; drafters trained on Qwen's own outputs (Speculators)\n"
             "H100 PCIe, batch 1, mean latency  ·  exec accuracy on 1,000 test queries", fontsize=7.8, color=MUTED,
             va="bottom", linespacing=1.6)
    fig.text(LEFT_X, 0.018, REPO, fontsize=7.8, color=INK_2, va="bottom")
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.2, top=0.85)
    out = os.path.join(HERE, "plots", "hero.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
