"""
PAPL finetune of the LLaDA-style SQL masked-diffusion checkpoint.

Background
----------
The base model (src/train.py) is trained on *uniform random* Bernoulli masks:
every masked position is weighted only by 1/t. At inference, however, we decode
with a *confidence planner* (denoising.py): the most confident predictions are
committed first and freeze as context for everything decoded later. Standard
discrete-diffusion training is blind to that planner, so the tokens that anchor
the trajectory (the early, high-confidence commits) are trained no harder than
any other token. A wrong early commit then poisons the rest of the trace.

PAPL (Planner-Aware Path Learning, arXiv:2509.23405) corrects the objective so
training matches the planner actually used at inference. This script implements
the practical confidence-reweighting form of that correction:

    ce_papl = ce + tau * stopgrad(exp(-ce)) * ce

`exp(-ce)` is ~1 when the model is already confident (low CE) and ~0 when it is
unsure. So confident tokens -- exactly the ones the planner commits first -- get
up to (1 + tau)x loss weight. This (a) makes early commits more reliable, which
makes the whole decoding trace more robust to ordering, and (b) is a
certainty-forcing pressure: confidence on committable tokens rises, so the
`confidence_stop` adaptive early-exit fires sooner and inference uses fewer
forward passes. `tau = 0` exactly recovers the original LLaDA loss.

This continues training from the existing checkpoint with a small LR for a
couple of epochs -- it is a finetune, not a from-scratch run. Eval logs both
generation exact-match AND average steps-to-converge (with confidence_stop on)
so the speed win is directly visible.

Usage
-----
    TAU=0.3 FT_EPOCHS=2 FT_LR=2e-5 python src/finetune_papl.py

Env knobs (all optional):
    CKPT_DIR        checkpoint to finetune (default diffusion-sql-modernbert)
    OUTPUT_DIR      where to save (default diffusion-sql-modernbert-papl)
    TAU             PAPL strength (default 0.3; 0 == plain LLaDA loss)
    FT_EPOCHS       finetune epochs (default 2)
    FT_LR           learning rate (default 2e-5)
    TRAIN_SIZE      train examples (default 100000)
    CONF_STOP       confidence early-stop threshold for eval (default 0.9)
    MAX_TRAIN_STEPS >0 caps steps for a smoke test (default 0 == use epochs)
    EVAL_STEPS      eval cadence (default 500)
"""

import contextlib
import json
import math
import os
import random
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import wandb

from augment import augment_example
from denoising import denoise_steps

# 1. Config
CKPT_DIR = os.environ.get("CKPT_DIR", "diffusion-sql-modernbert")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "diffusion-sql-modernbert-papl")
TAU = float(os.environ.get("TAU", "0.3"))         # PAPL strength; 0 == plain LLaDA
NUM_EPOCHS = float(os.environ.get("FT_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("FT_LR", "2e-5"))  # small: continuing a converged model
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "100000"))
CONF_STOP = float(os.environ.get("CONF_STOP", "0.9"))   # eval early-stop threshold
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))

# Fixed by the base checkpoint -- must match how it was trained.
MAX_LEN = 512
SQL_WINDOW = 128
T_EPS = 1e-3
VAL_SIZE = 500
FILTER_JOINS = False

# Default to multi-worker collation on CUDA (Linux/fork), but 0 workers off-CUDA:
# macOS spawns workers and cannot pickle the closure-based collator. Override
# explicitly with DATALOADER_WORKERS if needed.
DATALOADER_WORKERS = int(os.environ.get(
    "DATALOADER_WORKERS", "12" if torch.cuda.is_available() else "0"
))
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"

GEN_EVAL_SIZE = int(os.environ.get("GEN_EVAL_SIZE", "32"))  # examples for generation-based eval
GEN_EVAL_STEPS = 24    # max denoising steps during eval (budget)

# Batch sized for a 96 GB GH200 by default; far too large for a Mac's unified
# memory, so default to 16 off-CUDA. Override with BATCH_SIZE.
DEFAULT_BATCH = "128" if torch.cuda.is_available() else "16"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ["WANDB_PROJECT"] = "sql-diffusion"
if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
    os.environ["WANDB_MODE"] = "offline"
    print("[wandb] no WANDB_API_KEY found -> logging offline")
wandb.init(project="sql-diffusion", name="modernbert-papl-ft", config={
    "ckpt_dir": CKPT_DIR,
    "tau": TAU,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "train_size": TRAIN_SIZE,
    "conf_stop": CONF_STOP,
    "max_len": MAX_LEN,
    "sql_window": SQL_WINDOW,
    "t_eps": T_EPS,
})


# 2. Dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql")
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")
if FILTER_JOINS:
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda ex: "join" not in ex["sql"].lower())


# 3. Tokenizer + model -- loaded FROM the checkpoint. The special tags and the
# resized embeddings were saved with it, so we do not re-add or re-resize.
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
tokenizer.model_max_length = MAX_LEN
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
missing = [t for t in TAGS if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
if missing:
    raise RuntimeError(f"checkpoint tokenizer missing special tags {missing}; wrong CKPT_DIR?")

TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
SEP_ID = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id

model = AutoModelForMaskedLM.from_pretrained(CKPT_DIR)
if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))


# 4. Token-level input construction (identical to train.py so the checkpoint
# sees exactly the layout it was trained on).
def encode_text(prompt: str, context: str, sql: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    sql_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))

    budget = MAX_LEN - SQL_WINDOW - 9
    if len(prompt_ids) + len(context_ids) > budget:
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]

    ids = (
        [CLS_ID, TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
        TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'],
        TAG_IDS['<SQL>']]
    )
    sql_start = len(ids)
    ids = ids + sql_ids + [TAG_IDS['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW

    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "sql_start": sql_start,
        "sql_end": sql_end,
    }


# 5. Collator: augment + encode + continuous-t Bernoulli masking (same as base).
def make_collator(augment: bool):
    def collate(features):
        rows = []
        for f in features:
            p, c, s = f["sql_prompt"], f.get("sql_context", ""), f.get("sql", "")
            if augment:
                p, c, s = augment_example(p, c, s)
            rows.append(encode_text(p, c, s))

        input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        attention = torch.tensor([r["attention_mask"] for r in rows], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        B = input_ids.shape[0]

        t = torch.rand(B) * (1.0 - T_EPS) + T_EPS  # t ~ U(eps, 1]
        for i, r in enumerate(rows):
            s_, e_ = r["sql_start"], r["sql_end"]
            span = e_ - s_
            masked = torch.rand(span) < t[i]
            if not masked.any():
                masked[torch.randint(span, (1,))] = True
            idx = torch.nonzero(masked, as_tuple=True)[0] + s_
            labels[i, idx] = input_ids[i, idx]
            input_ids[i, idx] = MASK_ID

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "loss_weights": 1.0 / t,
        }
    return collate


train_collator = make_collator(augment=True)
eval_collator = make_collator(augment=False)


# 6. Trainer with the PAPL-reweighted ELBO loss.
class PAPLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        B, L, V = logits.shape
        ce = F.cross_entropy(
            logits.view(-1, V).float(), labels.view(-1),
            reduction="none", ignore_index=-100,
        ).view(B, L)
        # PAPL planner-aware reweight: upweight tokens the model is already
        # confident about (low CE) -- these are the ones the confidence planner
        # commits first at inference. stopgrad on the weight so it scales the
        # loss without backpropagating through the scaling itself. Non-masked
        # positions have ce == 0, so the term vanishes there automatically.
        if TAU > 0:
            papl_w = torch.exp(-ce.detach())          # (0, 1], ~1 when confident
            ce = ce + TAU * papl_w * ce
        # LLaDA objective: (1/t) * sum of (reweighted) masked CE, per-span-normalized.
        per_example = ce.sum(dim=1) * weights / SQL_WINDOW
        loss = per_example.mean()
        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        original = self.data_collator
        self.data_collator = eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original


# 7. Generation eval: exact match + average steps-to-converge.
def normalize_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def generation_eval(model, raw_examples, transform=None, n_steps=GEN_EVAL_STEPS):
    """Returns (exact_match, avg_steps_to_converge).

    Runs the real confidence planner with adaptive early-stop (CONF_STOP). The
    step count is the number of forward passes until the SQL window has no masks
    left -- the direct measure of inference speed PAPL is meant to improve.
    """
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(0)
    hits = 0
    total_steps = 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )
    for ex in raw_examples:
        p, c, s = ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", "")
        if transform is not None:
            p, c, s = transform(p, c, s, rng=rng)
        enc = encode_text(p, c, s)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
        lo, hi = enc["sql_start"], enc["sql_end"]
        ids[0, lo:hi] = MASK_ID
        steps_used = 0
        with autocast_ctx:
            for _ in denoise_steps(
                model, ids, attn, list(range(lo, hi)), MASK_ID,
                n_steps=n_steps, forbid_token_ids=list(TAG_IDS.values()),
                confidence_stop=CONF_STOP,
            ):
                steps_used += 1
                if (ids[0, lo:hi] == MASK_ID).sum().item() == 0:
                    break
        total_steps += steps_used
        out_ids = [tid for tid in ids[0, lo:hi].tolist() if tid not in (PAD_ID, MASK_ID)]
        pred_sql = tokenizer.decode(out_ids, skip_special_tokens=True)
        if normalize_sql(pred_sql) == normalize_sql(s):
            hits += 1
    n = max(1, len(raw_examples))
    return hits / n, total_steps / n


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, raw_examples):
        self.raw_examples = raw_examples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        em_clean, steps_clean = generation_eval(model, self.raw_examples)
        em_aug, steps_aug = generation_eval(model, self.raw_examples, transform=augment_example)
        print(
            f"[gen-eval] exact_match={em_clean:.3f} ({steps_clean:.1f} steps) "
            f"exact_match_aug={em_aug:.3f} ({steps_aug:.1f} steps) "
            f"n={len(self.raw_examples)}"
        )
        wandb.log({
            "eval/generation_exact_match": em_clean,
            "eval/generation_exact_match_aug": em_aug,
            "eval/avg_steps": steps_clean,
            "eval/avg_steps_aug": steps_aug,
        }, step=state.global_step)
        model.train()


# 8. Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=DATALOADER_WORKERS > 0,
    dataloader_drop_last=True,
    torch_compile=USE_TORCH_COMPILE,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
    logging_strategy="steps",
    report_to=["wandb"],
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
)


# 9. Data slices (collator encodes on the fly)
train_ds = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
val_ds = dataset["test"].select(range(min(VAL_SIZE, len(dataset["test"]))))
gen_eval_examples = [dataset["test"][i] for i in range(min(GEN_EVAL_SIZE, len(dataset["test"])))]


# 10. Baseline eval before any PAPL step, so the before/after delta is logged.
print(f"[papl] finetuning {CKPT_DIR} -> {OUTPUT_DIR} | tau={TAU} lr={LEARNING_RATE} epochs={NUM_EPOCHS}")
_dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(_dev)
em0, st0 = generation_eval(model, gen_eval_examples)
print(f"[papl] baseline (pre-finetune): exact_match={em0:.3f} avg_steps={st0:.1f}")
wandb.log({"eval/generation_exact_match": em0, "eval/avg_steps": st0}, step=0)


# 11. Finetune
trainer = PAPLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=train_collator,
    processing_class=tokenizer,
    callbacks=[GenerationEvalCallback(gen_eval_examples)],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 12. Final eval + machine-readable results dump, so an A/B driver can compare
# runs without scraping stdout. Includes the pre-finetune baseline (== base
# checkpoint) so each run is self-contained.
em_final, st_final = generation_eval(model, gen_eval_examples)
em_final_aug, st_final_aug = generation_eval(model, gen_eval_examples, transform=augment_example)
results = {
    "tau": TAU,
    "ckpt_dir": CKPT_DIR,
    "output_dir": OUTPUT_DIR,
    "train_size": TRAIN_SIZE,
    "max_train_steps": MAX_TRAIN_STEPS,
    "learning_rate": LEARNING_RATE,
    "gen_eval_size": GEN_EVAL_SIZE,
    "conf_stop": CONF_STOP,
    "baseline_exact_match": em0,
    "baseline_avg_steps": st0,
    "final_exact_match": em_final,
    "final_avg_steps": st_final,
    "final_exact_match_aug": em_final_aug,
    "final_avg_steps_aug": st_final_aug,
}
results_path = os.path.join(OUTPUT_DIR, "papl_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"[papl] results -> {results_path}")
print(
    f"[papl] tau={TAU}: exact_match {em0:.3f} -> {em_final:.3f} | "
    f"avg_steps {st0:.1f} -> {st_final:.1f}"
)

wandb.finish()
