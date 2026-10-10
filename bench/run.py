"""Run one backend over the eval slice; write per-example predictions and a summary.

    python -m bench.run --arm diffusion --engine onnx --model checkpoints/diffusion-sql-modernbert --threads 3
    python -m bench.run --arm ar --engine onnx --model checkpoints/sql-ar-qwen25-coder-0.5b-ar --threads 3

Latency is wall-clock per example at batch 1 (tokenization + generation + decode), after warmup.
"""

import argparse
import json
import os
import platform
import statistics
import time

import torch

from bench.common import load_eval_set, score


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["diffusion", "ar"], required=True)
    ap.add_argument("--engine", choices=["torch", "onnx", "vllm"], default="torch")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="Run name (default: derived from arm/engine/settings)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3, help="CPU threads (production uses 3)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16, help="diffusion: max denoising steps")
    ap.add_argument("--conf-stop", type=float, default=0.9, help="diffusion: adaptive early stop (0 = off)")
    ap.add_argument("--window", type=int, default=128, help="diffusion: SQL window size")
    ap.add_argument("--window-head", action="store_true", help="diffusion/onnx: MLM head on window positions only")
    ap.add_argument("--int8", action="store_true", help="onnx: dynamic int8 weight quantization")
    ap.add_argument("--prompt-isolation", action="store_true",
                    help="diffusion/torch: prompt cannot attend to the SQL window (precondition for prompt caching)")
    ap.add_argument("--verify", type=int, default=0,
                    help="if greedy SQL does not compile against the schema, try up to K sampled candidates")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"], help="torch engines")
    ap.add_argument("--ar-compile", action="store_true", help="ar/torch: static KV cache + torch.compile (GPU)")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    if args.arm == "diffusion":
        from bench.backends import DiffusionBackend

        backend = DiffusionBackend(args.model, args.engine, args.device, args.threads, args.window, args.steps,
                                   args.conf_stop or None, args.window_head, args.int8, args.prompt_isolation,
                                   args.dtype)
        default_name = (f"diffusion-{args.engine}-w{args.window}-s{args.steps}-c{args.conf_stop}"
                        f"{'-whead' if args.window_head else ''}{'-int8' if args.int8 else ''}"
                        f"{'-isolated' if backend.prompt_isolation else ''}")
    else:
        from bench.backends import ARBackend

        backend = ARBackend(args.model, args.engine, args.device, args.threads, int8=args.int8, dtype=args.dtype,
                            compile=args.ar_compile)
        default_name = f"ar-{args.engine}{'-int8' if args.int8 else ''}{'-compiled' if args.ar_compile else ''}"
    if args.dtype != "float32" and args.engine == "torch":
        default_name += f"-{args.dtype}"
    if args.verify:
        from bench.backends import VerifiedBackend

        backend = VerifiedBackend(backend, args.verify)
        default_name += f"-verify{args.verify}"
    name = args.name or default_name
    device = getattr(backend, "device", torch.device("cpu"))

    data = load_eval_set(args.n)
    for ex in data[: args.warmup]:
        backend.generate(ex["prompt"], ex["context"])

    rows = []
    for i, ex in enumerate(data):
        sync(device)
        t0 = time.perf_counter()
        pred, n_forward = backend.generate(ex["prompt"], ex["context"])
        sync(device)
        dt = time.perf_counter() - t0
        rows.append({"i": i, "prompt_tokens": getattr(backend, "prompt_tokens", None), "pred": pred, "gold": ex["sql"], "n_forward": n_forward, "latency_ms": dt * 1000,
                     **score(ex["context"], ex["sql"], pred)})
        if (i + 1) % 32 == 0:
            print(f"[{name}] {i + 1}/{len(data)}", flush=True)

    lat = sorted(r["latency_ms"] for r in rows)
    gold_ok = [r for r in rows if r["gold_ok"]]
    summary = {
        "name": name, "arm": args.arm, "engine": args.engine, "model": os.path.basename(os.path.normpath(args.model)),
        "params_m": None, "n": len(rows), "threads": args.threads, "device": str(device),
        "host": f"{platform.system()} {platform.machine()}",
        "exact": sum(r["exact"] for r in rows) / len(rows),
        "valid": sum(r["valid"] for r in rows) / len(rows),
        # execution accuracy only over examples whose gold SQL runs in SQLite
        "exec": sum(r["exec"] for r in gold_ok) / max(1, len(gold_ok)), "n_exec": len(gold_ok),
        "avg_forward": statistics.mean(r["n_forward"] for r in rows),
        "lat_mean_ms": statistics.mean(lat), "lat_p50_ms": lat[len(lat) // 2], "lat_p95_ms": lat[int(len(lat) * 0.95) - 1],
        "settings": {k: v for k, v in vars(args).items() if k not in ("out", "name")},
    }
    model = getattr(backend, "model", None)
    if isinstance(model, torch.nn.Module):
        summary["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(args.out, f"{name}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "settings"}))


if __name__ == "__main__":
    main()
