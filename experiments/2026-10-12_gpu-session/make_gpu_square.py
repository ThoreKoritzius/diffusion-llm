"""Square GPU chart (accuracy vs. speed) with only fair comparisons:
- every point is a model on its best available engine (AR: vLLM; diffusion: CUDA graphs), H100, batch 1;
- the speed bracket compares ~150M models on speed only (speed depends on size/architecture, not pretraining);
- Ettin-150M AR (no vLLM support, so no fair speed) is shown as an accuracy reference line: same pretraining and
  data as the Ettin diffusion point.

    python experiments/2026-10-12_gpu-session/make_gpu_square.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_headline_chart import COLOR, GRID, HERE, INK, INK_2, MUTED, SURFACE, load  # noqa: E402

# (label, arm, marker, gpu summary, label offset (dx, dy, ha)); marker: "o" model on its best engine,
# "s" larger reference model, "hollow" not on an optimized engine (no vLLM support)
POINTS = [
    ("Qwen2.5-0.5B AR · vLLM\n3.3× larger", "ar", "s", "sql-ar-qwen25-coder-0.5b-ar-vllm", (14, 0, "left")),
    ("SmolLM2-135M AR · vLLM", "ar", "o", "smollm135-ar-vllm", (0, -16, "center")),
    ("Ettin-150M AR · torch.compile\n(no vLLM support)", "ar", "hollow", "ettin150-ar-10ep-bf16-compiled-graphs",
     (0, 16, "center")),
    ("ModernBERT-150M diffusion", "diffusion", "o", "modernbert-isolated-distilled-bf16-cudagraphs", (0, 14, "center")),
    ("Ettin-150M diffusion", "diffusion", "o", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (14, -2, "left")),
]
# (AR, diffusion, label, label offset (dx, dy, ha))
ARROWS = [
    ("ettin150-ar-10ep-bf16-compiled-graphs", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     "same size, same pretraining\n{speed:.0f}× faster* · {acc:+.1f} pts", (0, 10, "center")),
    ("smollm135-ar-vllm", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     "same size, both on best engine\n{speed:.0f}× faster", (16, -4, "left")),
]


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(260, 12)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(34, 80)

    for ar_name, diff_name, text, (ldx, ldy, lha) in ARROWS:
        a, d = load("gpu", ar_name), load("gpu", diff_name)
        (xa, ya), (xd, yd) = (a["lat_mean_ms"], a["exec"] * 100), (d["lat_mean_ms"], d["exec"] * 100)
        ax.annotate("", xy=(xd, yd), xytext=(xa, ya), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.5, shrinkA=11, shrinkB=11, mutation_scale=14))
        ax.annotate(text.format(speed=xa / xd, acc=yd - ya), ((xa * xd) ** 0.5, (ya + yd) / 2), xytext=(ldx, ldy),
                    textcoords="offset points", ha=lha, va="bottom" if ldy > 0 else "top", fontsize=9.5, color=INK,
                    fontweight="bold", linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))

    for label, arm, marker, name, (dx, dy, ha) in POINTS:
        s = load("gpu", name)
        x, y = s["lat_mean_ms"], s["exec"] * 100
        if marker == "hollow":
            ax.scatter(x, y, s=200, marker="o", facecolor=SURFACE, edgecolor=COLOR[arm], linewidth=2.5, zorder=4)
        else:
            ax.scatter(x, y, s=260 if marker == "s" else 200, marker=marker, color=COLOR[arm], edgecolor=SURFACE,
                       linewidth=2.5, zorder=4)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center" if dy == 0 else ("bottom" if dy > 0 else "top"), fontsize=9.5, color=INK_2,
                    linespacing=1.3)

    ax.set_xticks([200, 100, 50, 30, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("mean latency per query on an H100, batch 1  →  faster", fontsize=10, color=INK_2, labelpad=8)
    ax.set_ylabel("execution accuracy", fontsize=10, color=INK_2, labelpad=8)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0, which="both")
    ax.text(0.98, 0.98, "better ↗", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["diffusion"],
                          label="diffusion · hand-written CUDA graphs"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["ar"],
                          label="autoregressive (AR) · vLLM"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor=SURFACE,
                          markeredgecolor=COLOR["ar"], markeredgewidth=2, label="AR · no vLLM support (torch.compile)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.suptitle("Diffusion vs. autoregressive (AR)\nat ~150M parameters", x=0.04, y=0.975, ha="left", va="top",
                 fontsize=17, fontweight="bold", color=INK, linespacing=1.15)
    fig.text(0.04, 0.855, "Text-to-SQL · same 256 test queries · H100, batch 1 · AR on vLLM where supported;\n"
             "diffusion on hand-written CUDA graphs. *Ettin AR has no vLLM support, so its speed is not\n"
             "fully optimized; the SmolLM2 arrow is the fair speed comparison.",
             fontsize=9, color=MUTED, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.1, top=0.765)
    out = os.path.join(HERE, "plots", "gpu_square.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
