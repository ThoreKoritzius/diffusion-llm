"""Dump Qwen masked-diffusion SQL predictions for exact/execution/judge grading."""

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

from denoising import denoise_steps
from sql_benchmark import execution_match

MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/sql-diffusion-qwen25-coder-0.5b-checkpoint-3126")
OUT_DIR = os.environ.get("DUMP_OUT", "benchmarks/diffusion_qwen25_256")
N = int(os.environ.get("DUMP_N", "256"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "128"))
GEN_STEPS = int(os.environ.get("GEN_STEPS", "24"))
CONF_STOP = os.environ.get("CONF_STOP")
CONF_STOP = None if CONF_STOP in (None, "", "none", "None") else float(CONF_STOP)
DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
)
OUT_NAME = os.environ.get("OUT_NAME", "diffusion_qwen25")

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.model_max_length = MAX_LEN
TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
SEP_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id


def encode_text(prompt: str, context: str, sql: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    sql_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))

    budget = MAX_LEN - SQL_WINDOW - 9
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]

    head = [CLS_ID] if CLS_ID is not None else []
    ids = (
        head
        + [TAG_IDS["<PROMPT>"]]
        + prompt_ids
        + [TAG_IDS["</PROMPT>"], TAG_IDS["<CONTEXT>"]]
        + context_ids
        + [TAG_IDS["</CONTEXT>"], TAG_IDS["<SQL>"]]
    )
    sql_start = len(ids)
    ids = ids + sql_ids + [TAG_IDS["</SQL>"], SEP_ID]
    sql_end = sql_start + SQL_WINDOW
    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {"input_ids": ids, "attention_mask": attention, "sql_start": sql_start, "sql_end": sql_end}


def make_bidirectional(model):
    base = model.model if hasattr(model, "model") else model

    def _padding_only_bias(attention_mask, input_tensor, kv_length=None):
        b, q = input_tensor.shape[0], input_tensor.shape[1]
        k = int(kv_length or (attention_mask.shape[-1] if attention_mask is not None else q))
        dtype, device = input_tensor.dtype, input_tensor.device
        bias = torch.zeros((b, 1, q, k), dtype=dtype, device=device)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask
            pad = attention_mask[:, :k] == 0
            bias = bias.masked_fill(pad[:, None, None, :], torch.finfo(dtype).min)
        return bias

    def _bidir_mask(attention_mask, input_tensor, *args, **kwargs):
        return _padding_only_bias(attention_mask, input_tensor)

    patched = False
    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = _bidir_mask
        patched = True
    else:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

        def _create_padding_mask(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values=None,
            position_ids=None,
            **kwargs,
        ):
            kv_length = attention_mask.shape[-1] if attention_mask is not None else input_embeds.shape[1]
            return _padding_only_bias(attention_mask, input_embeds, kv_length=kv_length)

        qwen2_modeling.create_causal_mask = _create_padding_mask
        patched = base.__class__.__module__ == qwen2_modeling.__name__

    if not patched:
        raise RuntimeError("Could not patch Qwen causal mask for bidirectional diffusion inference.")
    model.config._attn_implementation = "eager"
    model.config.is_causal = False
    return model


@torch.inference_mode()
def predict(model, ex):
    enc = encode_text(ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", ""))
    ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=DEVICE)
    attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=DEVICE)
    lo, hi = enc["sql_start"], enc["sql_end"]
    ids[0, lo:hi] = MASK_ID
    steps = 0
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        for _ in denoise_steps(
            model,
            ids,
            attn,
            list(range(lo, hi)),
            MASK_ID,
            n_steps=GEN_STEPS,
            forbid_token_ids=list(TAG_IDS.values()),
            confidence_stop=CONF_STOP,
        ):
            steps += 1
            if (ids[0, lo:hi] == MASK_ID).sum().item() == 0:
                break
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    out_ids = [t for t in ids[0, lo:hi].tolist() if t not in (PAD_ID, MASK_ID)]
    return tokenizer.decode(out_ids, skip_special_tokens=True), steps, latency


def main():
    print(
        f"[dump-diffusion-qwen] model={MODEL_DIR} device={DEVICE} n={N} "
        f"steps={GEN_STEPS} conf_stop={CONF_STOP} -> {OUT_DIR}",
        flush=True,
    )
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    model = make_bidirectional(model)

    ds = load_dataset("gretelai/synthetic_text_to_sql")["test"]
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    examples = [ds[i] for i in range(min(N, len(ds)))]

    pred_path = os.path.join(OUT_DIR, f"pred_{OUT_NAME}.jsonl")
    summary_path = os.path.join(OUT_DIR, f"summary_{OUT_NAME}.json")
    exact = exec_hits = exec_valid = total_steps = 0
    total_latency = 0.0
    with open(pred_path, "w") as f:
        for i, ex in enumerate(examples):
            pred, steps, latency = predict(model, ex)
            gold = ex.get("sql", "")
            em = normalize_sql(pred) == normalize_sql(gold)
            valid, exm = execution_match(ex.get("sql_context", ""), gold, pred)
            exact += int(em)
            exec_hits += int(exm)
            exec_valid += int(valid)
            total_steps += steps
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
                        "steps": steps,
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
        "avg_steps": total_steps / n,
        "avg_latency_ms": total_latency * 1000.0 / n,
        "predictions": pred_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dump-diffusion-qwen] summary={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
