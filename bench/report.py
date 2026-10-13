"""Summarize benchmark runs into a markdown table and an accuracy-vs-latency plot.

    python -m bench.report bench/results/cpu
"""

import glob
import json
import os
import sys


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "bench/results/cpu"
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(folder, "*.summary.json")))]
    runs.sort(key=lambda r: (r["arm"], r["lat_p50_ms"]))

    lines = ["| run | exec | valid | exact | fwd passes | p50 ms | p95 ms | mean ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['name']} | {r['exec']:.3f} | {r['valid']:.3f} | {r['exact']:.3f} | {r['avg_forward']:.1f} "
                     f"| {r['lat_p50_ms']:.0f} | {r['lat_p95_ms']:.0f} | {r['lat_mean_ms']:.0f} |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(folder, "summary.md"), "w") as f:
        f.write(table + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"diffusion": "#1f6feb", "ar": "#d1495b"}
    for r in runs:
        ax.scatter(r["lat_p50_ms"], r["exec"], color=colors[r["arm"]], s=40, zorder=3)
        ax.annotate(r["name"], (r["lat_p50_ms"], r["exec"]), fontsize=7, xytext=(5, 3), textcoords="offset points")
    for arm, c in colors.items():
        ax.scatter([], [], color=c, label=arm)
    ax.set_xlabel("p50 latency per query (ms, batch 1)")
    ax.set_ylabel("execution accuracy")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f"{runs[0]['host']} · {runs[0]['threads']} threads · n={runs[0]['n']}" if runs else "")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "frontier.png"), dpi=150)


if __name__ == "__main__":
    main()
