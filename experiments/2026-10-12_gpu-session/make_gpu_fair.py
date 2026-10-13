"""Square GPU chart that only makes fair comparisons:
  top    - speed at ~150M parameters, each model on its best available engine (H100, batch 1)
  bottom - accuracy at 150M with identical pretraining and training data (Ettin decoder vs. encoder)

    python experiments/2026-10-12_gpu-session/make_gpu_fair.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_headline_chart import COLOR, GRID, HERE, INK, INK_2, MUTED, SURFACE, load  # noqa: E402

SPEED = [  # (label, arm, gpu summary)
    ("SmolLM2-135M AR\nvLLM", "ar", "smollm135-ar-vllm"),
    ("Ettin-150M diffusion\nCUDA graphs", "diffusion", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs"),
    ("ModernBERT-150M diffusion\nCUDA graphs", "diffusion", "modernbert-isolated-distilled-bf16-cudagraphs"),
]
QUALITY = {  # arm: (label, cpu summary) -- Ettin pair, 10 epochs, same data
    "ar": ("Ettin-150M AR", "ar-10ep-torch"),
    "diffusion": ("Ettin-150M diffusion", "diffusion-iso-10ep-torch"),
}
METRICS = [("exec", "execution\naccuracy"), ("exact", "exact\nmatch"), ("valid", "valid\nSQL")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)


def main():
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(7.2, 7.2), facecolor=SURFACE)
    top = fig.add_axes((0.30, 0.565, 0.62, 0.20))
    bottom = fig.add_axes((0.12, 0.165, 0.80, 0.235))

    # --- speed: horizontal bars, mean latency
    rows = [(label, arm, load("gpu", name)) for label, arm, name in SPEED]
    ys = list(range(len(rows)))[::-1]
    for y, (label, arm, s) in zip(ys, rows):
        top.barh(y, s["lat_mean_ms"], height=0.56, color=COLOR[arm], zorder=3)
        top.text(s["lat_mean_ms"] + 1.2, y, f"{s['lat_mean_ms']:.0f} ms", va="center", fontsize=10, color=INK,
                 fontweight="bold")
    top.set_yticks(ys)
    top.set_yticklabels([label for label, _, _ in rows], fontsize=9.5, color=INK_2, linespacing=1.3)
    top.set_xlim(0, 90)
    top.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} ms"))
    top.grid(True, axis="x", color=GRID, linewidth=1, zorder=0)
    style(top)
    ar_ms = rows[0][2]["lat_mean_ms"]
    fast_ms = min(s["lat_mean_ms"] for _, arm, s in rows if arm == "diffusion")
    top.text(1.0, 1.04, f"diffusion {ar_ms / fast_ms:.0f}× faster", transform=top.transAxes, ha="right",
             fontsize=11, color=INK, fontweight="bold")
    fig.text(0.04, 0.835, "Speed · ~150M parameters, each on its best engine",
             fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.04, 0.812, "mean latency per query, H100, batch 1, same 256 queries", fontsize=9, color=MUTED)

    # --- quality: grouped bars, Ettin pair
    ar, diff = (load("cpu", QUALITY[k][1]) for k in ("ar", "diffusion"))
    width = 0.36
    for i, (key, label) in enumerate(METRICS):
        for offset, arm, s in ((-width / 2, "ar", ar), (width / 2, "diffusion", diff)):
            v = s[key] * 100
            bottom.bar(i + offset, v, width=width * 0.92, color=COLOR[arm], zorder=3)
            bottom.text(i + offset, v + 1.2, f"{v:.0f}%", ha="center", fontsize=9.5, color=INK)
    bottom.set_xticks(range(len(METRICS)))
    bottom.set_xticklabels([label for _, label in METRICS], fontsize=9.5, color=INK_2, linespacing=1.3)
    bottom.set_ylim(0, 92)
    bottom.set_yticks([0, 25, 50, 75])
    bottom.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    bottom.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    style(bottom)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["ar"], label=QUALITY["ar"][0]),
               plt.Rectangle((0, 0), 1, 1, color=COLOR["diffusion"], label=QUALITY["diffusion"][0])]
    bottom.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
                  bbox_to_anchor=(0.0, 1.06))
    fig.text(0.04, 0.465, "Accuracy · same size, same pretraining, same training data",
             fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.04, 0.442, "Ettin decoder (AR) vs. Ettin encoder (diffusion), 10 epochs, 256 queries",
             fontsize=9, color=MUTED)

    fig.text(0.04, 0.955, "Diffusion vs. autoregressive for text-to-SQL", fontsize=16, fontweight="bold",
             color=INK)
    fig.text(0.04, 0.915, "3× faster at matched size, with comparable accuracy", fontsize=12, color=INK_2)
    fig.text(0.04, 0.016, "Speed depends on architecture and size, not pretraining data, so the speed panel uses "
             "SmolLM2, a\nvLLM-supported AR model of the same size (Ettin AR has no vLLM support). Diffusion runs on "
             "hand-written\nCUDA graphs; no optimized serving engine exists for masked diffusion yet.",
             fontsize=7.8, color=MUTED, linespacing=1.4)
    out = os.path.join(HERE, "plots", "gpu_fair.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
