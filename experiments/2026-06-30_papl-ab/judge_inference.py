"""Grade + compare inference-improvement configs (refinement / repair) against
the baseline, on semantic accuracy, parse validity, and avg steps.

Grades every pred_*.jsonl in the given dir with the GPT-5.4-mini judge (reusing
an existing graded_*.jsonl instead of re-grading), then prints a config table
and writes data/judge_inference_summary.json.

Usage:  OPENAI_API_KEY=... python3 judge_inference.py <pred_dir>
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from judge_sql import grade_one, JUDGE_MODEL  # noqa: E402
from sql_repair import is_valid  # noqa: E402

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "predictions_inference")
DATA_DIR = os.path.join(HERE, "data")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "8"))


def load_or_grade(pred_path):
    graded_path = pred_path.replace("pred_", "graded_")
    if os.path.exists(graded_path):
        return [json.loads(l) for l in open(graded_path)]
    rows = [json.loads(l) for l in open(pred_path)]
    graded = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(grade_one, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            graded[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"    {os.path.basename(pred_path)}: {done}/{len(rows)}", flush=True)
    with open(graded_path, "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    return graded


def stats(graded):
    n = len(graded)
    valid = [g for g in graded if g["verdict"] != "error"]
    return {
        "n": n,
        "avg_steps": sum(g.get("steps", 0) for g in graded) / n,
        "parse_valid": sum(is_valid(g["pred"]) for g in graded) / n,
        "exact": sum(g["exact_match"] for g in graded) / n,
        "semantic": sum(g["verdict"] == "equivalent" for g in valid) / len(valid) if valid else 0.0,
    }


def main():
    preds = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*.jsonl")))
    if not preds:
        print(f"no pred_*.jsonl in {PRED_DIR}")
        sys.exit(1)
    print(f"[judge-inf] model={JUDGE_MODEL} files={[os.path.basename(p) for p in preds]}")
    out = {}
    for p in preds:
        cfg = os.path.basename(p)[len("pred_"):-len(".jsonl")]
        out[cfg] = stats(load_or_grade(p))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "judge_inference_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    hdr = f"{'config':>26} | {'steps':>6} | {'valid%':>6} | {'exact':>6} | {'semantic':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for cfg, s in sorted(out.items(), key=lambda kv: kv[1]["semantic"]):
        print(f"{cfg:>26} | {s['avg_steps']:>6.2f} | {s['parse_valid']:>6.1%} | "
              f"{s['exact']:>6.3f} | {s['semantic']:>8.3f}")
    print(f"\n[judge-inf] wrote data/judge_inference_summary.json")


if __name__ == "__main__":
    main()
