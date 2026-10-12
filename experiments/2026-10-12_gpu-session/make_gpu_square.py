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

# (label, arm, square marker = larger reference model, gpu summary, label offset (dx, dy, ha))
POINTS = [
    ("Qwen2.5-0.5B AR · vLLM\n3.3× larger", "ar", True, "sql-ar-qwen25-coder-0.5b-ar-vllm", (14, 0, "left")),
    ("SmolLM2-135M AR · vLLM", "ar", False, "smollm135-ar-vllm", (0, 14, "center")),
    ("ModernBERT-150M diffusion", "diffusion", False, "modernbert-isolated-distilled-bf16-cudagraphs", (0, 14, "center")),
    ("Ettin-150M diffusion", "diffusion", False, "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs",
     (0, -16, "center")),
]
SPEED_PAIR = ("smollm135-ar-vllm", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs")
AR_REFERENCE = ("ettin150-ar-10ep-bf16-compiled", "Ettin-150M AR accuracy (same pretraining and data\n"
                "as Ettin diffusion; no vLLM support, so no fair speed)")


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(160, 12)  # reversed: faster to the right, so up-and-right is better
    ax.set_ylim(30, 80)

    ref = load("gpu", AR_REFERENCE[0])
    ax.axhline(ref["exec"] * 100, color=COLOR["ar"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(150, ref["exec"] * 100 + 0.8, AR_REFERENCE[1] + f": {ref['exec'] * 100:.0f}%", fontsize=8.3,
            color=INK_2, va="bottom", linespacing=1.3)

    for label, arm, big, name, (dx, dy, ha) in POINTS:
        s = load("gpu", name)
        x, y = s["lat_mean_ms"], s["exec"] * 100
        ax.scatter(x, y, s=260 if big else 200, marker="s" if big else "o", color=COLOR[arm], edgecolor=SURFACE,
                   linewidth=2.5, zorder=4)
        ax.annotate(f"{label}\n{x:.0f} ms · {y:.0f}%", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center" if dy == 0 else ("bottom" if dy > 0 else "top"), fontsize=9.5, color=INK_2,
                    linespacing=1.3)

    a, d = (load("gpu", n) for n in SPEED_PAIR)
    yb = 33.5
    ax.annotate("", xy=(d["lat_mean_ms"], yb), xytext=(a["lat_mean_ms"], yb),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=14, shrinkA=0, shrinkB=0))
    for x in (a["lat_mean_ms"], d["lat_mean_ms"]):
        ax.plot([x, x], [yb - 0.9, yb + 0.9], color=INK, lw=1.5)
    ax.text((a["lat_mean_ms"] * d["lat_mean_ms"]) ** 0.5, yb + 1.3,
            f"{a['lat_mean_ms'] / d['lat_mean_ms']:.0f}× faster at ~150M parameters",
            ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")

    ax.set_xticks([150, 100, 50, 30, 20])
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
                          label="autoregressive (AR) · vLLM")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.suptitle("Diffusion is ~3× faster than\nvLLM-served AR models", x=0.04, y=0.975, ha="left", va="top",
                 fontsize=17, fontweight="bold", color=INK, linespacing=1.15)
    fig.text(0.04, 0.855, "Text-to-SQL · same 256 test queries · each model on its best available engine.\n"
             "AR gets a production engine (vLLM); diffusion has none yet, so it is the underdog here.",
             fontsize=9, color=MUTED, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.1, top=0.765)
    out = os.path.join(HERE, "plots", "gpu_square.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
