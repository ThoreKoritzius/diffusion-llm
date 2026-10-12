"""Square GPU chart (accuracy vs. speed) with only fair comparisons:
- every point is a model on its best available engine (AR: vLLM; diffusion: CUDA graphs), H100, batch 1;
- the highlighted pair is Ettin-150M decoder (AR) vs. encoder (diffusion): same size, same pretraining data, same
  SQL data, both optimized (Ettin AR runs in vLLM through vllm_plugin/); larger models are shown for context.

    python experiments/2026-10-12_gpu-session/make_gpu_square.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_headline_chart import COLOR, GRID, HERE, INK, INK_2, MUTED, SURFACE, load  # noqa: E402

HERO_AR, HERO_DIFF = "ettin150-ar-10ep-vllm", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs"
# (label, arm, gpu summary): the matched Ettin-150M pair, emphasised
HERO = [("Autoregressive", "Ettin-150M decoder", "ar", HERO_AR),
        ("Diffusion", "Ettin-150M encoder", "diffusion", HERO_DIFF)]
# (label, arm, gpu summary, label offset (dx, dy, ha, va)): other models, for context
CONTEXT = [
    ("Qwen2.5-0.5B AR · 3× larger", "ar", "sql-ar-qwen25-coder-0.5b-ar-vllm", (0, 12, "center", "bottom")),
    ("Ettin-400M diffusion", "diffusion", "ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (-12, 0, "right", "center")),
    ("ModernBERT-150M diffusion", "diffusion", "modernbert-isolated-distilled-bf16-cudagraphs",
     (0, 12, "center", "bottom")),
]
# accuracy from the 1,000-query CPU fp32 runs (same queries for every model); latency from the H100 runs
ACCURACY = {
    HERO_AR: "ettin150-ar-10ep",
    HERO_DIFF: "ettin150-diffusion-10ep-distilled-w64",
    "sql-ar-qwen25-coder-0.5b-ar-vllm": "qwen25-0.5b-ar",
    "ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs": "ettin400-diffusion-10ep-distilled-w64",
    "modernbert-isolated-distilled-bf16-cudagraphs": "modernbert-distilled-w64",
}
REPO = "github.com/ThoreKoritzius/diffusion-llm"
LEFT_X = 0.06  # shared left edge of title, subtitle and footer


def point(name):
    return load("gpu", name)["lat_mean_ms"], load("n1000", ACCURACY[name])["exec"] * 100


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(110, 17)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(44, 77)

    (xa, ya), (xd, yd) = point(HERO_AR), point(HERO_DIFF)

    for label, arm, name, (dx, dy, ha, va) in CONTEXT:
        x, y = point(name)
        ax.scatter(x, y, s=140, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9, color=INK_2, linespacing=1.4)

    ax.annotate("", xy=(xd, yd), xytext=(xa, ya), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, shrinkA=16, shrinkB=16, mutation_scale=18))
    mid = ((xa * xd) ** 0.5, (ya + yd) / 2)
    ax.annotate(f"{xa / xd:.1f}× faster", mid, xytext=(0, -16), textcoords="offset points", ha="center",
                va="top", fontsize=19, fontweight="bold", color=INK)
    ax.annotate("at similar accuracy", mid, xytext=(0, -44), textcoords="offset points", ha="center",
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
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["diffusion"], label="diffusion"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=COLOR["ar"], label="autoregressive")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.text(LEFT_X, 0.95, f"Diffusion LLM: {xa / xd:.1f}× faster than autoregressive", fontsize=18,
             fontweight="bold", color=INK, va="top")
    fig.text(LEFT_X, 0.902, "Same 150M model family, same training data  ·  text-to-SQL  ·  H100", fontsize=11,
             color=INK_2, va="top")
    fig.text(LEFT_X, 0.05, "Ettin-150M encoder (diffusion) vs. decoder (AR)  ·  AR on vLLM, diffusion on CUDA graphs\n"
             "Latency: H100, batch 1  ·  accuracy: 1,000 test queries", fontsize=7.8, color=MUTED, va="bottom",
             linespacing=1.5)
    fig.text(LEFT_X, 0.018, REPO, fontsize=7.8, color=INK_2, va="bottom")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.18, top=0.85)
    out = os.path.join(HERE, "plots", "gpu_square.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
