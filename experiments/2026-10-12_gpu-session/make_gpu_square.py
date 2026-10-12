"""Square GPU chart (accuracy vs. speed) with only fair comparisons:
- every point is a model on its best available engine (AR: vLLM; diffusion: CUDA graphs), H100, batch 1;
- the arrow connects the Ettin-150M pair: same size, same pretraining data, same SQL data, both optimized
  (Ettin AR runs in vLLM through vllm_plugin/).

    python experiments/2026-10-12_gpu-session/make_gpu_square.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_headline_chart import COLOR, GRID, HERE, INK, INK_2, MUTED, SURFACE, load  # noqa: E402

# (label, arm, marker, gpu summary, label offset (dx, dy, ha)); marker: "o" ~150M model, "s" larger model
POINTS = [
    ("Qwen2.5-0.5B AR (494M)", "ar", "s", "sql-ar-qwen25-coder-0.5b-ar-vllm", (14, 0, "left")),
    ("SmolLM2-135M AR", "ar", "o", "smollm135-ar-vllm", (14, 0, "left")),
    ("Ettin-150M AR", "ar", "o", "ettin150-ar-10ep-vllm", (-14, -4, "right")),
    ("Ettin-400M diffusion", "diffusion", "s", "ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (-14, 4, "right")),
    ("ModernBERT-150M diffusion", "diffusion", "o", "modernbert-isolated-distilled-bf16-cudagraphs", (0, 14, "center")),
    ("Ettin-150M diffusion", "diffusion", "o", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (14, -6, "left")),
]
# (AR, diffusion, label, label offset (dx, dy, ha))
ARROWS = [
    ("ettin150-ar-10ep-vllm", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     "same size, same pretraining\n{speed:.1f}× faster · {acc:+.1f} pts", (0, -12, "center")),
]


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(100, 16)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(38, 78)

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
        ax.scatter(x, y, s=260 if marker == "s" else 200, marker=marker, color=COLOR[arm], edgecolor=SURFACE,
                   linewidth=2.5, zorder=4)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center" if dy == 0 else ("bottom" if dy > 0 else "top"), fontsize=9.5, color=INK_2,
                    linespacing=1.3)

    ax.set_xticks([100, 70, 50, 30, 20])
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
    ax.text(0.98, 0.02, "better ↗", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["diffusion"],
                          label="diffusion · hand-written CUDA graphs"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=COLOR["ar"],
                          label="autoregressive (AR) · vLLM"),
               plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=MUTED, label="larger model (~400–500M)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.suptitle("Diffusion vs. autoregressive (AR)\non a GPU", x=0.04, y=0.975, ha="left", va="top",
                 fontsize=17, fontweight="bold", color=INK, linespacing=1.15)
    fig.text(0.04, 0.855, "Text-to-SQL · same 256 test queries · H100, batch 1 · AR on vLLM; diffusion on\n"
             "hand-written CUDA graphs. The arrow joins the Ettin pair: same size, same pretraining data,\n"
             "same SQL training data, each on its fastest engine.",
             fontsize=9, color=MUTED, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.1, top=0.765)
    out = os.path.join(HERE, "plots", "gpu_square.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
