"""How well does the DFlash drafter keep up with Qwen on CPU? Tokens per verifier pass, wall time, and whether the
output matches plain Qwen, for plain Qwen and for speculative decoding with 1..K candidate blocks per pass.

    python scripts/drafter_probe.py --model-dir checkpoints/sql-ar-qwen25-coder-0.5b-ar \
        --drafter-dir checkpoints/spec-qwen25/dflash --candidates 1 2 3 --out results/drafter-probe
Prompt sets: bench/prompts/vague_50.jsonl (short, vague questions over unseen schemas) and the first --n-test
queries of the gretelai test split (the drafter's training distribution).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwen_engines import QwenEngine  # noqa: E402


def load_sets(n_test: int):
    vague = [json.loads(line) for line in open(os.path.join(os.path.dirname(__file__), "..", "bench", "prompts",
                                                            "vague_50.jsonl"))]
    from bench.common import load_eval_set

    test = [{"prompt": r["prompt"], "context": r["context"]} for r in load_eval_set(n_test)]
    return {"vague": vague, "test": test}


def run(eng, row, speculative, candidates):
    last = None
    for last in eng.generate(row["prompt"], row["context"], speculative=speculative, candidates=candidates):
        pass
    return {"sql": eng.decode(last["tokens"]), "tokens": len(last["tokens"]), "passes": last["forward"],
            "seconds": last["elapsed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--drafter-dir", required=True)
    ap.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/drafter-probe")
    args = ap.parse_args()

    eng = QwenEngine(args.model_dir, drafter_dir=args.drafter_dir, threads=args.threads)
    os.makedirs(args.out, exist_ok=True)
    configs = [("qwen", False, 1)] + [(f"dflash_k{k}", True, k) for k in args.candidates]
    rows = []
    for set_name, prompts in load_sets(args.n_test).items():
        run(eng, prompts[0], True, 1)  # warm-up
        for i, row in enumerate(prompts):
            ref = None
            for name, spec, k in configs:
                r = run(eng, row, spec, k)
                ref = ref or r["sql"]
                rows.append({"set": set_name, "i": i, "config": name, **r, "same": r["sql"] == ref})
        print(set_name, "done", flush=True)
    with open(os.path.join(args.out, "runs.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    lines = ["| set | config | tokens / pass | passes | mean s | speedup | same as Qwen |", "|---|---|---:|---:|---:|---:|---:|"]
    for set_name in ("vague", "test"):
        base = sum(r["seconds"] for r in rows if r["set"] == set_name and r["config"] == "qwen")
        for name, _, _ in configs:
            rs = [r for r in rows if r["set"] == set_name and r["config"] == name]
            tok, passes, sec = (sum(r[k] for r in rs) for k in ("tokens", "passes", "seconds"))
            lines.append(f"| {set_name} | {name} | {tok / passes:.2f} | {passes / len(rs):.1f} | {sec / len(rs):.2f} | "
                         f"{base / sec:.2f}x | {sum(r['same'] for r in rs)}/{len(rs)} |")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f} s")
