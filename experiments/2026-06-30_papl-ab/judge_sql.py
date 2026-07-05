"""LLM-as-judge grading of text-to-SQL predictions (OpenAI GPT-5.4-mini).

Grades each prediction for *semantic* equivalence to the gold SQL given the
schema (catches correct SQL that exact-match misses: aliases, column order,
whitespace, equivalent predicates) and annotates a failure category so failures
can be clustered.

Inputs:  <ab_dir>/predictions/pred_<model>.jsonl   (from src/dump_predictions.py)
Outputs: <ab_dir>/predictions/graded_<model>.jsonl (per-row verdicts)
         <ab_dir>/data/judge_summary.json          (per-model + comparison)

Requires OPENAI_API_KEY and `pip install openai`. Model is configurable via
JUDGE_MODEL (default gpt-5.4-mini).

Usage:
    OPENAI_API_KEY=... python3 judge_sql.py <ab_dir> [--models base,tau_0.3]
"""
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
AB_DIR = None
for a in sys.argv[1:]:
    if not a.startswith("-"):
        AB_DIR = a
if AB_DIR is None:
    # default: newest papl_ab_* under repo root (two levels up)
    cands = sorted(glob.glob(os.path.join(HERE, "..", "..", "papl_ab_*")))
    AB_DIR = cands[-1] if cands else "."
PRED_DIR = os.path.join(AB_DIR, "predictions")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "8"))
want = None
for i, a in enumerate(sys.argv):
    if a == "--models" and i + 1 < len(sys.argv):
        want = sys.argv[i + 1].split(",")

client = OpenAI()  # reads OPENAI_API_KEY

FAILURE_CATEGORIES = [
    "none", "wrong_table", "wrong_or_missing_column", "missing_or_wrong_filter",
    "wrong_aggregate_or_groupby", "wrong_join", "wrong_order_or_limit",
    "hallucinated_identifier", "syntax_error", "truncated_or_empty", "other",
]

SCHEMA = {
    "name": "sql_judgement",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["equivalent", "not_equivalent"],
                        "description": "equivalent = predicted SQL returns the same result as gold for any valid DB under the schema"},
            "failure_category": {"type": "string", "enum": FAILURE_CATEGORIES,
                                 "description": "'none' iff equivalent; else the single most important reason it differs"},
            "confidence": {"type": "number", "description": "0-1 confidence in the verdict"},
            "rationale": {"type": "string", "description": "one short sentence"},
        },
        "required": ["verdict", "failure_category", "confidence", "rationale"],
    },
}

SYSTEM = (
    "You are a strict SQL grader. Given a database schema, a natural-language question, "
    "a gold SQL query, and a predicted SQL query, decide whether the predicted query is "
    "SEMANTICALLY EQUIVALENT to the gold query for the question — i.e. it would return the "
    "same result set on any database matching the schema. Ignore cosmetic differences "
    "(whitespace, casing, alias names, column/JOIN order, equivalent predicates, LIMIT vs "
    "TOP). Judge meaning, not text. If the prediction is empty, truncated, or invalid SQL, "
    "it is not_equivalent. Pick the single most important failure_category when not equivalent."
)


def build_user(row):
    return (
        f"# Schema\n{row['context'] or '(none provided)'}\n\n"
        f"# Question\n{row['prompt']}\n\n"
        f"# Gold SQL\n{row['gold']}\n\n"
        f"# Predicted SQL\n{row['pred'] or '(empty)'}\n"
    )


def grade_one(row, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_user(row)}],
                response_format={"type": "json_schema", "json_schema": SCHEMA},
            )
            data = json.loads(resp.choices[0].message.content)
            return {**row, **data}
        except Exception as e:  # noqa: BLE001 — bulk job, keep going on transient errors
            if attempt == retries - 1:
                return {**row, "verdict": "error", "failure_category": "other",
                        "confidence": 0.0, "rationale": f"judge error: {e}"}
            time.sleep(2 ** attempt)


def grade_file(pred_path):
    name = os.path.basename(pred_path)[len("pred_"):-len(".jsonl")]
    rows = [json.loads(l) for l in open(pred_path)]
    graded = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(grade_one, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            graded[futs[fut]] = fut.result()
            done += 1
            if done % 25 == 0:
                print(f"    {name}: graded {done}/{len(rows)}", flush=True)
    out_path = os.path.join(PRED_DIR, f"graded_{name}.jsonl")
    with open(out_path, "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")

    n = len(graded)
    errors = sum(g["verdict"] == "error" for g in graded)
    valid = [g for g in graded if g["verdict"] != "error"]
    sem = sum(g["verdict"] == "equivalent" for g in valid)
    em = sum(g["exact_match"] for g in graded)
    # Judge validation: on exact-string-match rows, judge should say equivalent.
    em_rows = [g for g in valid if g["exact_match"]]
    em_agree = sum(g["verdict"] == "equivalent" for g in em_rows)
    fails = Counter(g["failure_category"] for g in valid if g["verdict"] == "not_equivalent")
    summary = {
        "model": name, "n": n, "judge_errors": errors,
        "exact_match_acc": em / n if n else 0.0,
        "semantic_acc": sem / len(valid) if valid else 0.0,
        "semantic_minus_exact": (sem / len(valid) if valid else 0.0) - (em / n if n else 0.0),
        "judge_validation_agree_on_exact_correct": (em_agree / len(em_rows)) if em_rows else None,
        "failure_histogram": dict(fails.most_common()),
    }
    print(f"[judge] {name}: exact={summary['exact_match_acc']:.3f} "
          f"semantic={summary['semantic_acc']:.3f} "
          f"(+{summary['semantic_minus_exact']:.3f}) "
          f"val={summary['judge_validation_agree_on_exact_correct']}")
    return summary


def main():
    preds = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*.jsonl")))
    if want:
        preds = [p for p in preds if os.path.basename(p)[5:-6] in want]
    if not preds:
        print(f"no pred_*.jsonl in {PRED_DIR} — run src/dump_predictions.py first")
        sys.exit(1)
    print(f"[judge] model={JUDGE_MODEL} files={[os.path.basename(p) for p in preds]}")
    summaries = [grade_file(p) for p in preds]

    out = {"judge_model": JUDGE_MODEL, "models": {s["model"]: s for s in summaries}}
    with open(os.path.join(DATA_DIR, "judge_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Comparison tables
    print("\n================ SEMANTIC vs EXACT (judge=%s) ================" % JUDGE_MODEL)
    print(f"{'model':>8} | {'exact':>6} | {'semantic':>8} | {'Δ':>6} | {'judge-val':>9}")
    for s in summaries:
        v = s["judge_validation_agree_on_exact_correct"]
        print(f"{s['model']:>8} | {s['exact_match_acc']:>6.3f} | {s['semantic_acc']:>8.3f} | "
              f"{s['semantic_minus_exact']:>+6.3f} | {('%.2f' % v) if v is not None else '   n/a':>9}")

    print("\n================ FAILURE MODES (share of graded) ================")
    cats = sorted({c for s in summaries for c in s["failure_histogram"]})
    print(f"{'category':>26} | " + " | ".join(f"{s['model']:>8}" for s in summaries))
    for c in cats:
        cells = []
        for s in summaries:
            cnt = s["failure_histogram"].get(c, 0)
            cells.append(f"{cnt/ s['n']:.3f}" if s['n'] else "0.000")
        print(f"{c:>26} | " + " | ".join(f"{x:>8}" for x in cells))
    print(f"\n[judge] wrote {os.path.join(DATA_DIR, 'judge_summary.json')}")


if __name__ == "__main__":
    main()
