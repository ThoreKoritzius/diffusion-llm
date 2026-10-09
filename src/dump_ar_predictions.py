"""Dump autoregressive SQL predictions for exact/execution/LLM-judge grading."""

import contextlib
import json
import os
import re
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sql_benchmark import execution_match

MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/sql-ar-qwen25-coder-0.5b-ar")
OUT_DIR = os.environ.get("DUMP_OUT", "benchmarks/ar_qwen25")
N = int(os.environ.get("DUMP_N", "256"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "128"))
DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
)
OUT_NAME = os.environ.get("OUT_NAME", "ar_qwen25")

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.model_max_length = MAX_LEN
TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
BOS_IDS = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def build_prompt_ids(prompt: str, context: str):
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(context, add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - SQL_WINDOW - 12
    if len(p) + len(c) > budget:
        c = c[: max(0, budget - len(p))]
        p = p[:budget]
    return (
        BOS_IDS
        + [TAG_IDS["<PROMPT>"]]
        + p
        + [TAG_IDS["</PROMPT>"], TAG_IDS["<CONTEXT>"]]
        + c
        + [TAG_IDS["</CONTEXT>"], TAG_IDS["<SQL>"]]
    )


@torch.inference_mode()
def predict(model, ex):
    prompt = build_prompt_ids(ex["sql_prompt"], ex.get("sql_context", ""))
    ids = torch.tensor([prompt], dtype=torch.long, device=DEVICE)
    attn = torch.ones_like(ids)
    close_sql = TAG_IDS["</SQL>"]
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        out = model.generate(
            ids,
            attention_mask=attn,
            max_new_tokens=SQL_WINDOW + 2,
            do_sample=False,
            num_beams=1,
            eos_token_id=[EOS_ID, close_sql],
            pad_token_id=PAD_ID,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    gen = out[0, ids.shape[1] :].tolist()
    cut = gen.index(close_sql) if close_sql in gen else len(gen)
    return tokenizer.decode(gen[:cut], skip_special_tokens=True), len(gen), latency


def main():
    print(f"[dump-ar] model={MODEL_DIR} device={DEVICE} n={N} -> {OUT_DIR}", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(DEVICE).eval()
    ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    examples = [ds[i] for i in range(min(N, len(ds)))]

    pred_path = os.path.join(OUT_DIR, f"pred_{OUT_NAME}.jsonl")
    summary_path = os.path.join(OUT_DIR, f"summary_{OUT_NAME}.json")
    exact = exec_hits = exec_valid = total_new = 0
    total_latency = 0.0
    with open(pred_path, "w") as f:
        for i, ex in enumerate(examples):
            pred, new_tokens, latency = predict(model, ex)
            gold = ex.get("sql", "")
            em = normalize_sql(pred) == normalize_sql(gold)
            valid, exm = execution_match(ex.get("sql_context", ""), gold, pred)
            exact += int(em)
            exec_hits += int(exm)
            exec_valid += int(valid)
            total_new += new_tokens
            total_latency += latency
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "prompt": ex["sql_prompt"],
                        "context": ex.get("sql_context", ""),
                        "gold": gold,
                        "pred": pred,
                        "exact_match": em,
                        "execution_match": exm,
                        "execution_valid": valid,
                        "steps": new_tokens,
                        "latency_s": latency,
                    }
                )
                + "\n"
            )
            if (i + 1) % 25 == 0:
                print(
                    f"    {i+1}/{len(examples)} exact={exact/(i+1):.3f} "
                    f"exec={exec_hits/(i+1):.3f} valid={exec_valid/(i+1):.3f}",
                    flush=True,
                )
    n = max(1, len(examples))
    summary = {
        "model": OUT_NAME,
        "n": len(examples),
        "exact_match_acc": exact / n,
        "execution_match_acc": exec_hits / n,
        "execution_valid_acc": exec_valid / n,
        "avg_new_tokens": total_new / n,
        "avg_latency_ms": total_latency * 1000.0 / n,
        "predictions": pred_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dump-ar] summary={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
