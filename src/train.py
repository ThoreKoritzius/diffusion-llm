"""
SQL Masked-Diffusion Training (LLaDA-style)

Key design (vs. the previous version):
- ModernBERT-base backbone
- Token-level input construction (no decode/re-tokenize round trip).
- Continuous mask ratio t ~ U(eps, 1] with Bernoulli per-token masking,
  matching the absorbing-state discrete diffusion forward process.
- Cross-entropy weighted by 1/t (the discrete diffusion ELBO), so the
  fully-masked regime that generation starts from is properly trained.
- SQL span padded to a fixed window with [PAD]; pads are maskable and
  predictable, which is how the model learns output length.
- On-the-fly augmentation (spaced/mixed-case identifiers, colloquial
  prompts) so the model is robust to messy real-world schemas.
- Generation-based eval (exact match via confidence-based denoising) on
  both clean and augmented inputs, not just masked-LM loss.
"""

import contextlib
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

# 1. Hyperparameters and Config
MODEL_NAME = "answerdotai/ModernBERT-base"
NUM_EPOCHS = 10
BATCH_SIZE = 128             # safe on a 96 GB GH200 for ModernBERT-base @ seq 512
LEARNING_RATE = 1e-4         # ~sqrt-scaled from 3e-5@bs32 for the 4x larger batch
MAX_LEN = 512
SQL_WINDOW = 128
TRAIN_SIZE = 100_000
VAL_SIZE = 500
T_EPS = 1e-3          # lower bound for mask ratio t
FILTER_JOINS = False

# Throughput knobs. The collator tokenizes + augments per-example in Python, so
# without worker processes the GH200 sits idle waiting on the CPU.
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", "12"))
# Off by default: inductor mis-handles ModernBERT's flash-attn unpadding under
# bf16 (Float/BFloat16 dtype mismatch). flash-attn already gives the big speedup.
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
# >0 caps training to N steps (overrides epochs) — used for the pre-run smoke test.
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))

GEN_EVAL_SIZE = 32    # examples for generation-based exact-match eval
GEN_EVAL_STEPS = 24   # denoising steps during eval

OUTPUT_DIR = "diffusion-sql-modernbert"

# Hopper perf: TF32 matmuls + avoid the fast-tokenizer fork warning/deadlock
# once dataloader workers are forking the module-level tokenizer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ["WANDB_PROJECT"] = "sql-diffusion"
# Don't let a missing API key block the run on a rented box: fall back to
# offline logging (sync later with `wandb sync`) instead of hanging on a prompt.
if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
    os.environ["WANDB_MODE"] = "offline"
    print("[wandb] no WANDB_API_KEY found -> logging offline")
wandb.init(project="sql-diffusion", name="modernbert-llada-aug", config={
    "model": MODEL_NAME,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_len": MAX_LEN,
    "sql_window": SQL_WINDOW,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "t_eps": T_EPS,
    "filter_joins": FILTER_JOINS,
    "augmentation": True,
    "learning_rate": LEARNING_RATE,
    "dataloader_workers": DATALOADER_WORKERS,
    "torch_compile": USE_TORCH_COMPILE,
})


# 2. Load and Filter Dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql")
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")
if FILTER_JOINS:
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda ex: "join" not in ex["sql"].lower())


# 3. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LEN
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
tokenizer.add_special_tokens({'additional_special_tokens': TAGS})

TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
SEP_ID = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id


# 4. Token-level Input Construction
def encode_text(prompt: str, context: str, sql: str):
    """Builds:
        [CLS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL> sql+pads </SQL> [SEP] [PAD]...
    The SQL span is padded to SQL_WINDOW with PAD tokens that are part of the
    diffusion target (the model learns to predict PAD => output length).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    sql_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))

    # 9 fixed tokens: CLS, 6 tags, SEP — leave the rest for prompt+context
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


# 5. Model Setup
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))


# 6. Collator: Augment + Encode + Continuous-t Bernoulli Masking
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


# 7. Trainer with 1/t-weighted ELBO Loss
class DiffusionTrainer(Trainer):
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
        # LLaDA objective: (1/t) * sum of masked-token CE, normalized by span length
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


# 8. Generation-based Eval (exact match, clean + augmented)
def normalize_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def generation_exact_match(model, raw_examples, transform=None, n_steps=GEN_EVAL_STEPS) -> float:
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(0)
    hits = 0
    # Match training precision: weights are fp32 but flash-attn needs fp16/bf16,
    # so run the denoising forwards under bf16 autocast on CUDA.
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
        with autocast_ctx:
            for _ in denoise_steps(
                model, ids, attn, list(range(lo, hi)), MASK_ID,
                n_steps=n_steps, forbid_token_ids=list(TAG_IDS.values()),
            ):
                pass
        out_ids = [tid for tid in ids[0, lo:hi].tolist() if tid not in (PAD_ID, MASK_ID)]
        pred_sql = tokenizer.decode(out_ids, skip_special_tokens=True)
        if normalize_sql(pred_sql) == normalize_sql(s):
            hits += 1
    return hits / max(1, len(raw_examples))


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, raw_examples):
        self.raw_examples = raw_examples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        em_clean = generation_exact_match(model, self.raw_examples)
        em_aug = generation_exact_match(model, self.raw_examples, transform=augment_example)
        print(f"[gen-eval] exact_match={em_clean:.3f} exact_match_aug={em_aug:.3f} (n={len(self.raw_examples)})")
        wandb.log({
            "eval/generation_exact_match": em_clean,
            "eval/generation_exact_match_aug": em_aug,
        }, step=state.global_step)
        model.train()


# 9. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    # Keep the GH200 fed: parallel collation + pinned, prefetched, fixed-shape batches.
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=DATALOADER_WORKERS > 0,
    dataloader_drop_last=True,            # constant batch shape -> stable torch.compile
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


# 10. Dataset Slices (collator encodes on the fly, so these stay raw)
train_ds = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
val_ds = dataset["test"].select(range(min(VAL_SIZE, len(dataset["test"]))))
gen_eval_examples = [dataset["test"][i] for i in range(min(GEN_EVAL_SIZE, len(dataset["test"])))]

# 11. Train
trainer = DiffusionTrainer(
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

wandb.finish()
