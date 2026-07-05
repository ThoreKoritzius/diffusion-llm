#!/usr/bin/env python3
"""compare_strategies.py - verify DOS dependency-ordering vs confidence ordering.

Runs generation-based exact-match on the text-to-SQL test split using the
trained ModernBERT diffusion model, comparing the legacy confidence-based
unmasking against the new dependency-oriented (DOS) ordering at several
`dep_alpha` blends. Reports exact-match accuracy and average denoising steps.

The model is loaded with eager attention so `output_attentions` returns real
attention matrices (sdpa/flash kernels do not expose them).

Usage:
  python3 compare_strategies.py --n 64 --steps 24 \
      --model-dir diffusion-sql-modernbert
"""
import argparse
import random
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from denoising import denoise_steps

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 512
SQL_WINDOW = 128
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']


def build_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.model_max_length = MAX_LEN
    # The trained tokenizer already contains the tags, but adding again is a
    # no-op if present and guarantees the ids exist when pointing at the base.
    tok.add_special_tokens({'additional_special_tokens': TAGS})
    return tok


def encode_text(tok, ids_map, prompt: str, context: str, sql: str):
    """Mirror of train.py.encode_text (token-level construction)."""
    CLS_ID = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    SEP_ID = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    PAD_ID = tok.pad_token_id

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tok(context, add_special_tokens=False)["input_ids"]
    sql_ids = tok(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    sql_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))

    budget = MAX_LEN - SQL_WINDOW - 9
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]

    ids = [CLS_ID, ids_map['<PROMPT>']] + prompt_ids + [
        ids_map['</PROMPT>'], ids_map['<CONTEXT>']] + context_ids + [
        ids_map['</CONTEXT>'], ids_map['<SQL>']]
    sql_start = len(ids)
    ids = ids + sql_ids + [ids_map['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW

    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {"input_ids": ids, "attention_mask": attention,
            "sql_start": sql_start, "sql_end": sql_end}


def normalize_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def evaluate(model, tok, ids_map, examples, *, strategy, dep_alpha, dep_layers,
             n_steps, mask_id, pad_id, tag_ids, device, confidence_stop=None,
             dep_layer_index=None):
    hits, total_steps_used, n = 0, 0, 0
    for ex in examples:
        p, c, s = ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", "")
        enc = encode_text(tok, ids_map, p, c, s)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
        lo, hi = enc["sql_start"], enc["sql_end"]
        ids[0, lo:hi] = mask_id

        # Count committing steps via on_step_stats (one call per commit), so the
        # early-stop final no-op yield does not inflate the step count.
        stats = []
        for _ in denoise_steps(
            model, ids, attn, list(range(lo, hi)), mask_id,
            n_steps=n_steps, forbid_token_ids=tag_ids,
            strategy=strategy, dep_alpha=dep_alpha, dep_layers=dep_layers,
            dep_layer_index=dep_layer_index,
            confidence_stop=confidence_stop, on_step_stats=stats.append,
        ):
            pass
        out_ids = [t for t in ids[0, lo:hi].tolist() if t not in (pad_id, mask_id)]
        pred_sql = tok.decode(out_ids, skip_special_tokens=True)
        if normalize_sql(pred_sql) == normalize_sql(s):
            hits += 1
        total_steps_used += len(stats)
        n += 1
    return hits / max(1, n), total_steps_used / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="diffusion-sql-modernbert")
    ap.add_argument("--n", type=int, default=64, help="num test examples")
    ap.add_argument("--offset", type=int, default=0, help="start index into test split")
    ap.add_argument("--steps", type=int, default=24, help="denoising steps")
    ap.add_argument("--dep-layers", type=int, default=1)
    ap.add_argument("--confidence-stop", type=float, default=None,
                    help="adaptive early-stop threshold (e.g. 0.9); omit/0 to disable "
                         "and run fixed --steps")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0",
                    help="comma-sep dep_alpha values for the dependency strategy")
    ap.add_argument("--dep-layer-index", type=int, default=None,
                    help="single attention layer to read for dependency (0=shallowest, "
                         "-1=last); default averages the last --dep-layers layers")
    ap.add_argument("--layer-sweep", default="",
                    help="comma-sep layer indices to sweep with FAITHFUL DOS config "
                         "(standalone dep_alpha=1.0); overrides --alphas when set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tok = build_tokenizer(args.model_dir)
    # eager attention is required so output_attentions returns real matrices.
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_dir, attn_implementation="eager").to(device).eval()
    if model.get_input_embeddings().num_embeddings < len(tok):
        model.resize_token_embeddings(len(tok))

    ids_map = {t: tok.convert_tokens_to_ids(t) for t in TAGS}
    tag_ids = list(ids_map.values())
    mask_id = tok.mask_token_id
    pad_id = tok.pad_token_id

    ds = load_dataset("gretelai/synthetic_text_to_sql")
    test = ds["test"].filter(lambda ex: ex["sql_prompt"].strip() != "")
    start = min(args.offset, max(0, len(test) - 1))
    examples = [test[i] for i in range(start, min(start + args.n, len(test)))]

    conf_stop = args.confidence_stop if args.confidence_stop else None
    print(f"device={device}  model={args.model_dir}  n={len(examples)}  "
          f"steps={args.steps}  dep_layers={args.dep_layers}  "
          f"confidence_stop={conf_stop}\n")

    results = []

    t0 = time.time()
    acc, avg_steps = evaluate(
        model, tok, ids_map, examples, strategy="confidence", dep_alpha=0.0,
        dep_layers=args.dep_layers, n_steps=args.steps, mask_id=mask_id,
        pad_id=pad_id, tag_ids=tag_ids, device=device, confidence_stop=conf_stop)
    dt = time.time() - t0
    results.append(("confidence (baseline)", acc, avg_steps, dt))
    print(f"{'confidence (baseline)':28s} acc={acc:.3f}  avg_steps={avg_steps:.1f}  ({dt:.0f}s)")

    layer_sweep = [int(x) for x in args.layer_sweep.split(",") if x.strip()]
    if layer_sweep:
        # Faithful DOS: standalone dependency ordering (dep_alpha=1.0), reading a
        # single tuned attention layer, swept across the given layer indices.
        n_layers = model.config.num_hidden_layers
        print(f"(faithful DOS: standalone dep_alpha=1.0, model has {n_layers} layers)\n")
        for li in layer_sweep:
            t0 = time.time()
            acc, avg_steps = evaluate(
                model, tok, ids_map, examples, strategy="dependency", dep_alpha=1.0,
                dep_layers=args.dep_layers, dep_layer_index=li, n_steps=args.steps,
                mask_id=mask_id, pad_id=pad_id, tag_ids=tag_ids, device=device,
                confidence_stop=conf_stop)
            dt = time.time() - t0
            label = f"DOS layer={li:>3}"
            results.append((label, acc, avg_steps, dt))
            print(f"{label:28s} acc={acc:.3f}  avg_steps={avg_steps:.1f}  ({dt:.0f}s)")
    else:
        for a in [float(x) for x in args.alphas.split(",") if x.strip()]:
            t0 = time.time()
            acc, avg_steps = evaluate(
                model, tok, ids_map, examples, strategy="dependency", dep_alpha=a,
                dep_layers=args.dep_layers, dep_layer_index=args.dep_layer_index,
                n_steps=args.steps, mask_id=mask_id, pad_id=pad_id, tag_ids=tag_ids,
                device=device, confidence_stop=conf_stop)
            dt = time.time() - t0
            label = f"dependency a={a:.2f}"
            results.append((label, acc, avg_steps, dt))
            print(f"{label:28s} acc={acc:.3f}  avg_steps={avg_steps:.1f}  ({dt:.0f}s)")

    base = results[0][1]
    base_ms = results[0][3] / max(1, len(examples)) * 1000.0
    best = max(results, key=lambda r: r[1])
    print("\n=== summary ===")
    for label, acc, avg_steps, dt in results:
        delta = acc - base
        ms = dt / max(1, len(examples)) * 1000.0
        ms_delta = (ms - base_ms) / max(1e-9, base_ms) * 100.0
        tag = "  <-- best" if label == best[0] else ""
        print(f"  {label:28s} acc={acc:.3f}  delta={delta:+.3f}  "
              f"avg_steps={avg_steps:4.1f}  "
              f"latency={ms:7.1f}ms/ex ({ms_delta:+.0f}% vs baseline){tag}")
    if best[0].startswith("confidence"):
        print("\nDOS did NOT beat the confidence baseline on this slice.")
    else:
        print(f"\nDOS best: {best[0]}  ({best[1]:.3f} vs {base:.3f}, "
              f"{best[1] - base:+.3f}).")


if __name__ == "__main__":
    main()
