"""Short, vague versions of training questions, for drafter data that looks like what people type into the playground.

An instruction model rewrites each sampled train question ("List the names of customers who placed more than 5 orders
in 2023") into a casual one ("who orders a lot?") over the same schema. The verifier answers these itself in
scripts/spec_data.py (--extra), so the drafter learns to predict the verifier on vague prompts too; the verifier and
the test split are untouched.

    python scripts/vague_prompts.py --n 40000 --out checkpoints/vague_prompts.jsonl
"""

import argparse
import json
import os
import random

STYLES = [
    "very short, 2 to 5 words",
    "a casual question, all lowercase",
    "a casual question with one small typo",
    "an ambiguous superlative such as 'best', 'top' or 'worst', without saying by which measure",
    "a vague request that names no columns",
    "a question a manager would ask, without any numbers or thresholds",
]
SYSTEM = ("You rewrite database questions into what a non-technical person would type into a search box. Keep the "
          "same topic and intent, but be vague: drop exact column names, numbers, dates and output details. Reply "
          "with the rewritten question only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--max-row", type=int, default=99800, help="sample from train rows [0, max-row)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    rng = random.Random(args.seed)
    rows = [ds[i] for i in rng.sample(range(min(len(ds), args.max_row)), min(args.n, args.max_row))]
    chats = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": f"Style: {rng.choice(STYLES)}.\nQuestion: {r['sql_prompt']}"}] for r in rows]

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=1024,
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.3")))
    outs = llm.chat(chats, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=40, seed=args.seed))
    kept = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            q = o.outputs[0].text.strip().splitlines()[0].strip().strip('"').strip() if o.outputs[0].text.strip() else ""
            if not q or len(q) > 200:
                continue
            f.write(json.dumps({"prompt": q, "context": r["sql_context"] or "", "source": r["sql_prompt"]}) + "\n")
            kept += 1
    print(f"wrote {kept} vague prompts to {args.out}")


if __name__ == "__main__":
    main()
