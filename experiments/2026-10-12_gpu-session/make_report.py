"""Build the comparison tables and accuracy-vs-latency plots from data/{gpu,cpu}/*.summary.json.

    python experiments/2026-10-12_gpu-session/make_report.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, arm, params, summary file) -- one row per configuration; same 256 examples everywhere
GPU = [
    ("Qwen2.5-0.5B AR · vLLM", "ar", "494M", "sql-ar-qwen25-coder-0.5b-ar-vllm"),
    ("Qwen2.5-0.5B AR · HF static cache + compile, bf16", "ar", "494M", "sql-ar-qwen25-coder-0.5b-ar-bf16-compiled"),
    ("Qwen2.5-0.5B AR · HF eager, fp32", "ar", "494M", "sql-ar-qwen25-coder-0.5b-ar-fp32-eager"),
    ("Ettin-150M AR, 10 ep · HF static cache + compile, bf16", "ar", "150M", "ettin150-ar-10ep-bf16-compiled"),
    ("Ettin-150M AR, 3 ep · HF static cache + compile, bf16", "ar", "150M", "ettin150-ar-bf16-compiled"),
    ("ModernBERT diffusion, distilled · CUDA graphs, bf16", "diffusion", "150M", "modernbert-isolated-distilled-bf16-cudagraphs"),
    ("ModernBERT diffusion, distilled · PyTorch eager + prompt cache, bf16", "diffusion", "150M", "modernbert-isolated-distilled-bf16-cached"),
    ("Ettin-150M diffusion, 10 ep, distilled · CUDA graphs, bf16", "diffusion", "150M", "ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs"),
    ("Ettin-150M diffusion, 10 ep · CUDA graphs, bf16", "diffusion", "150M", "ettin150-diffusion-iso-10ep-bf16-cudagraphs"),
    ("ModernBERT diffusion, production model · CUDA graphs, bf16", "diffusion", "150M", "diffusion-sql-modernbert-bf16-cudagraphs"),
    ("ModernBERT diffusion, production model · PyTorch eager, bf16", "diffusion", "150M", "diffusion-sql-modernbert-bf16"),
]
CPU = [
    ("Qwen2.5-0.5B AR · llama.cpp Q8_0", "ar", "494M", "qwen25-ar-llamacpp-q8_0"),
    ("Qwen2.5-0.5B AR · llama.cpp f16", "ar", "494M", "qwen25-ar-llamacpp-f16"),
    ("Qwen2.5-0.5B AR · ONNX Runtime fp32 (KV cache)", "ar", "494M", "ar-onnx-qwen2.5-0.5b"),
    ("Ettin-150M AR, 10 ep · PyTorch fp32 (KV cache)", "ar", "150M", "ar-10ep-torch"),
    ("Ettin-150M AR, 3 ep · PyTorch fp32 (KV cache)", "ar", "150M", "ettin150-ar-torch"),
    ("ModernBERT diffusion, distilled · ONNX + prompt cache, int8", "diffusion", "150M", "modernbert-isolated-distilled-onnx-cached-int8"),
    ("ModernBERT diffusion, distilled · ONNX + prompt cache, fp32", "diffusion", "150M", "modernbert-isolated-distilled-onnx-cached"),
    ("Ettin-150M diffusion, 10 ep, distilled · ONNX + prompt cache, int8", "diffusion", "150M", "ettin150-iso-10ep-distilled-onnx-cached-int8"),
    ("Ettin-150M diffusion, 10 ep, distilled · ONNX + prompt cache, fp32", "diffusion", "150M", "ettin150-iso-10ep-distilled-onnx-cached"),
    ("Ettin-150M diffusion, 10 ep · ONNX + prompt cache, fp32", "diffusion", "150M", "diffusion-iso-10ep-onnx-cached"),
    ("Ettin-150M diffusion, 3 ep · PyTorch fp32", "diffusion", "150M", "ettin150-diffusion-torch-w64"),
    ("ModernBERT diffusion, production config (window 128) · ONNX fp32", "diffusion", "150M", "diffusion-onnx-w128-s16-c0.9"),
]


def load(kind, rows):
    out = []
    for label, arm, params, name in rows:
        s = json.load(open(os.path.join(HERE, "data", kind, f"{name}.summary.json")))
        out.append((label, arm, params, s))
    return out


def table(rows):
    lines = ["| configuration | params | exec | valid | exact | forward passes | p50 ms | p95 ms | mean ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, arm, params, s in rows:
        lines.append(f"| {'**AR**' if arm == 'ar' else '**Diffusion**'} {label} | {params} | {s['exec']:.3f} | "
                     f"{s['valid']:.3f} | {s['exact']:.3f} | {s['avg_forward']:.1f} | {s['lat_p50_ms']:.0f} | "
                     f"{s['lat_p95_ms']:.0f} | {s['lat_mean_ms']:.0f} |")
    return "\n".join(lines)


def short(label: str) -> str:
    model, _, engine = label.partition(" · ")
    model = (model.replace("Qwen2.5-0.5B AR", "Qwen-0.5B").replace("Ettin-150M AR", "Ettin-AR")
             .replace("Ettin-150M diffusion", "Ettin-diff").replace("ModernBERT diffusion", "MB-diff")
             .replace(", production model", " prod").replace(", production config (window 128)", " prod w128")
             .replace(", distilled", " dist").replace(", 10 ep", " 10ep").replace(", 3 ep", " 3ep"))
    engine = (engine.replace("HF static cache + compile", "HF compiled").replace("PyTorch eager", "eager")
              .replace("ONNX Runtime", "ONNX").replace(" (KV cache)", "").replace(" + prompt cache", "+cache"))
    return f"{model} · {engine}"


def plot(rows, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"ar": "#d1495b", "diffusion": "#1f6feb"}
    for label, arm, params, s in rows:
        marker = "o" if params == "150M" else "s"
        ax.scatter(s["lat_mean_ms"], s["exec"], color=colors[arm], marker=marker, s=46, zorder=3)
        ax.annotate(short(label), (s["lat_mean_ms"], s["exec"]), fontsize=6.5,
                    xytext=(5, -9) if "production" in label else (5, 3), textcoords="offset points", color="#333")
    ax.scatter([], [], color=colors["ar"], label="AR")
    ax.scatter([], [], color=colors["diffusion"], label="diffusion")
    ax.scatter([], [], color="#888", marker="o", label="150M params")
    ax.scatter([], [], color="#888", marker="s", label="494M params")
    ax.set_xscale("log")
    ax.set_xlabel("mean latency per query, batch 1 (ms, log scale)")
    ax.set_ylabel("execution accuracy")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    gpu, cpu = load("gpu", GPU), load("cpu", CPU)
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    plot(gpu, "GPU: NVIDIA H100 PCIe, batch 1, n=256", os.path.join(HERE, "plots", "gpu_frontier.png"))
    plot(cpu, "CPU: Apple Silicon, 3 threads, batch 1, n=256", os.path.join(HERE, "plots", "cpu_frontier.png"))
    md = f"### GPU\n\n{table(gpu)}\n\n### CPU\n\n{table(cpu)}\n"
    open(os.path.join(HERE, "data", "tables.md"), "w").write(md)
    print(md)
