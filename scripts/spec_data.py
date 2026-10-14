"""On-policy training data for a speculative-decoding drafter (Speculators format: input_ids + loss_mask).

The verifier (an AR SQL model from src/train_ar.py) answers training prompts greedily in vLLM; each row is the prompt
in the serving layout (bench/backends.py ARBackend.prompt_ids) followed by the verifier's own output, with the loss
on the output only. Uses training rows only (never the test split).

    python scripts/spec_data.py --model checkpoints/sql-ar-qwen25-coder-0.5b-ar --n 30000 --out data.jsonl

--extra adds rows from a {prompt, context} JSONL (e.g. scripts/vague_prompts.py), answered the same way.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.common import TAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--offset", type=int, default=0, help="first train row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", help="JSONL of {prompt, context} rows to add")
    ap.add_argument("--max-new-tokens", type=int, default=130)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tid = {t: tok.convert_tokens_to_ids(t) for t in TAGS}
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    ds = ds.select(range(args.offset, min(len(ds), args.offset + args.n)))

    def prompt_ids(prompt, context):  # same as ARBackend.prompt_ids
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(context, add_special_tokens=False)["input_ids"]
        budget = 512 - 128 - 12
        if len(p) + len(c) > budget:
            c = c[: max(0, budget - len(p))]
            p = p[:budget]
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        return bos + [tid["<PROMPT>"]] + p + [tid["</PROMPT>"], tid["<CONTEXT>"]] + c + [tid["</CONTEXT>"], tid["<SQL>"]]

    prompts = [prompt_ids(ex["sql_prompt"], ex["sql_context"] or "") for ex in ds]
    if args.extra:
        extra = [json.loads(line) for line in open(args.extra)]
        prompts += [prompt_ids(r["prompt"], r["context"]) for r in extra]
        print(f"+ {len(extra)} rows from {args.extra}")
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.4")))
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            stop_token_ids=[tid["</SQL>"], tok.eos_token_id])
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    with open(args.out, "w") as f:
        for p, o in zip(prompts, outs):
            gen = list(o.outputs[0].token_ids)  # includes the stop token, so the drafter learns where SQL ends
            f.write(json.dumps({"input_ids": p + gen, "loss_mask": [0] * len(p) + [1] * len(gen)}) + "\n")
    print(f"wrote {len(prompts)} rows to {args.out}")


if __name__ == "__main__":
    main()
