"""Square GPU-only headline chart: accuracy vs. speed, up-and-right is better.

    python experiments/2026-10-12_gpu-session/make_gpu_square.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_headline_chart import COLOR, GRID, HERE, INK, INK_2, MUTED, SURFACE, load  # noqa: E402

# (label, arm, square marker = larger reference model, summary, label offset (dx, dy, ha))
POINTS = [
    ("Qwen2.5-0.5B AR\n3.3× larger · vLLM", "ar", True, "sql-ar-qwen25-coder-0.5b-ar-vllm", (0, 14, "center")),
    ("SmolLM2-135M AR\nvLLM", "ar", False, "smollm135-ar-vllm", (0, -26, "center")),
    ("Ettin-150M AR\nHF compiled", "ar", False, "ettin150-ar-10ep-bf16-compiled", (0, -26, "center")),
    ("ModernBERT-150M\ndiffusion", "diffusion", False, "modernbert-isolated-distilled-bf16-cudagraphs", (0, 14, "center")),
    ("Ettin-150M\ndiffusion", "diffusion", False, "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (0, -26, "center")),
]
PAIRS = [  # (AR, diffusion, linestyle, footnote under the speedup)
    ("ettin150-ar-10ep-bf16-compiled", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs", "solid",
     "AR on HF: no vLLM support"),
    ("smollm135-ar-vllm", "modernbert-isolated-distilled-bf16-cudagraphs", "dashed", "both on their fastest engine"),
]


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for ar_name, diff_name, style, note in PAIRS:
        a, d = load("gpu", ar_name), load("gpu", diff_name)
        if a is None or d is None:
            continue
        (xa, ya), (xd, yd) = (a["lat_mean_ms"], a["exec"] * 100), (d["lat_mean_ms"], d["exec"] * 100)
        ax.annotate("", xy=(xd, yd), xytext=(xa, ya), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4, linestyle=style, shrinkA=11, shrinkB=11,
                                    mutation_scale=13))
        mid = ((xa * xd) ** 0.5, (ya + yd) / 2)
        ax.annotate(f"{xa / xd:.0f}× faster", mid, xytext=(0, 9), textcoords="offset points", ha="center",
                    va="bottom", fontsize=11, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
        ax.annotate(note, mid, xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=8,
                    color=MUTED, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))

    for label, arm, big, name, (dx, dy, ha) in POINTS:
        s = load("gpu", name)
        if s is None:
            continue
        x, y = s["lat_mean_ms"], s["exec"] * 100
        ax.scatter(x, y, s=260 if big else 190, marker="s" if big else "o", color=COLOR[arm], edgecolor=SURFACE,
                   linewidth=2.5, zorder=4)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="bottom" if dy > 0 else "top", fontsize=9.5, color=INK_2, linespacing=1.3)

    ax.set_xscale("log")
    ax.set_xlim(700, 10)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(46, 80)
    ax.set_xticks([500, 200, 100, 50, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("latency per query on an H100, batch 1  →  faster", fontsize=10, color=INK_2, labelpad=8)
    ax.set_ylabel("execution accuracy", fontsize=10, color=INK_2, labelpad=8)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0, which="both")
    ax.text(0.98, 0.98, "better ↗", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=MUTED)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["diffusion"], label="diffusion"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["ar"],
                          label="autoregressive (AR)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10, labelcolor=INK_2)
    fig.suptitle("Diffusion answers SQL questions\nin ~25 ms on a GPU", x=0.04, y=0.975, ha="left", va="top",
                 fontsize=17, fontweight="bold", color=INK, linespacing=1.15)
    fig.text(0.04, 0.855, "Same 256 test queries · each model on its fastest engine · arrows link\n"
             "comparable models (solid: same size and pretraining; dashed: similar size)",
             fontsize=9, color=MUTED, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.1, top=0.765)
    out = os.path.join(HERE, "plots", "gpu_square.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
