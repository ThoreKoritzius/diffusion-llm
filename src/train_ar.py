"""
SQL Autoregressive SFT baseline (Qwen3.5-0.8B-Base).

This is the AR control for the diffusion-vs-AR comparison the PAPL report flagged
as missing ("no AR baseline or latency measurement"). It fine-tunes a causal
decoder on the *same* gretelai text2sql data and the *same* <PROMPT>/<CONTEXT>/<SQL>
tag scheme as the diffusion arm, so the only variable is AR vs. masked-diffusion.

Differences from the diffusion arm (src/train_diffusion_qwen.py):
- causal LM, standard next-token loss, loss masked to the SQL completion only;
- no fixed SQL_WINDOW pad target — AR learns output length via EOS;
- eval is greedy `model.generate` with wall-clock + forward-pass (== new tokens)
  latency, which is the number the diffusion arm has to beat with parallel decode.

Env knobs mirror src/train.py: MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM,
LEARNING_RATE, MAX_TRAIN_STEPS (smoke test), EVAL_STEPS, TRAIN_SIZE, OUTPUT_DIR.
"""

import contextlib
import os
import random
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import wandb

from augment import augment_example
from sql_benchmark import execution_match, normalize_sql

# 1. Config
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-0.8B-Base")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))   # low on purpose: 10ep on
                                                       # templated SQL collapsed
                                                       # the diffusion model.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))  # per-device; 0.8B @ seq 512
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))   # effective batch 64
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))  # SFT-scale
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "128"))  # max SQL tokens (parity)
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "100000"))
VAL_SIZE = int(os.environ.get("VAL_SIZE", "500"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", "8"))
GEN_EVAL_SIZE = int(os.environ.get("GEN_EVAL_SIZE", "32"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "sql-ar-qwen0.8b")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ["WANDB_PROJECT"] = "sql-diffusion"
if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
    os.environ["WANDB_MODE"] = "offline"
    print("[wandb] no WANDB_API_KEY found -> logging offline")
wandb.init(project="sql-diffusion", name=f"ar-{MODEL_NAME.split('/')[-1]}", config={
    "model": MODEL_NAME, "arm": "autoregressive", "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
    "sql_window": SQL_WINDOW, "train_size": TRAIN_SIZE, "lr": LEARNING_RATE,
})

# 2. Data
dataset = load_dataset("gretelai/synthetic_text_to_sql")
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")

# 3. Tokenizer + tags (same scheme as the diffusion arm for a fair comparison)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LEN
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
tokenizer.add_special_tokens({'additional_special_tokens': TAGS})
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # Qwen ships no pad; reuse eos
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
BOS_IDS = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def build_prompt_ids(prompt: str, context: str):
    """Everything up to and including <SQL> — the causal conditioning prefix."""
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(context, add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - SQL_WINDOW - 12
    if len(p) + len(c) > budget:
        c = c[: max(0, budget - len(p))]
        p = p[:budget]
    return (
        BOS_IDS + [TAG_IDS['<PROMPT>']] + p + [TAG_IDS['</PROMPT>'],
        TAG_IDS['<CONTEXT>']] + c + [TAG_IDS['</CONTEXT>'], TAG_IDS['<SQL>']]
    )


def build_completion_ids(sql: str):
    s = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
    return s + [TAG_IDS['</SQL>'], EOS_ID]


# 4. Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.resize_token_embeddings(len(tokenizer))
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False


# 5. Collator: build prompt+completion, mask loss to the completion, right-pad.
def make_collator(augment: bool):
    def collate(features):
        seqs, labels = [], []
        for f in features:
            p, c, s = f["sql_prompt"], f.get("sql_context", ""), f.get("sql", "")
            if augment:
                p, c, s = augment_example(p, c, s)
            prompt = build_prompt_ids(p, c)
            comp = build_completion_ids(s)
            ids = (prompt + comp)[:MAX_LEN]
            lab = ([-100] * len(prompt) + comp)[:MAX_LEN]
            seqs.append(ids)
            labels.append(lab)
        width = max(len(x) for x in seqs)
        input_ids, attn, lab_out = [], [], []
        for ids, lab in zip(seqs, labels):
            pad = width - len(ids)
            input_ids.append(ids + [PAD_ID] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            lab_out.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lab_out, dtype=torch.long),
        }
    return collate


train_collator = make_collator(augment=True)
eval_collator = make_collator(augment=False)


class ARTrainer(Trainer):
    def get_eval_dataloader(self, eval_dataset=None):
        original = self.data_collator
        self.data_collator = eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original


@torch.no_grad()
def generation_eval(model, raw_examples, transform=None):
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(0)
    close_sql = TAG_IDS['</SQL>']
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )
    hits, exec_hits, valid_exec, total_new, total_time = 0, 0, 0, 0, 0.0
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    for ex in raw_examples:
        p, c, s = ex["sql_prompt"], ex.get("sql_context", ""), ex.get("sql", "")
        if transform is not None:
            p, c, s = transform(p, c, s, rng=rng)
        prompt = build_prompt_ids(p, c)
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        t0 = time.perf_counter()
        with autocast_ctx:
            out = model.generate(
                ids, attention_mask=attn, max_new_tokens=SQL_WINDOW + 2,
                do_sample=False, num_beams=1,
                eos_token_id=[EOS_ID, close_sql], pad_token_id=PAD_ID,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        gen = out[0, ids.shape[1]:].tolist()
        total_new += len(gen)
        cut = gen.index(close_sql) if close_sql in gen else len(gen)
        pred_sql = tokenizer.decode(gen[:cut], skip_special_tokens=True)
        if normalize_sql(pred_sql) == normalize_sql(s):
            hits += 1
        pred_valid, pred_exec_match = execution_match(c, s, pred_sql)
        valid_exec += int(pred_valid)
        exec_hits += int(pred_exec_match)
    model.config.use_cache = prev_cache
    n = max(1, len(raw_examples))
    return {
        "exact_match": hits / n,
        "execution_match": exec_hits / n,
        "execution_valid": valid_exec / n,
        "avg_new_tokens": total_new / n,   # == forward passes for AR greedy
        "avg_forward_steps": total_new / n,
        "avg_latency_s": total_time / n,
    }


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, raw_examples):
        self.raw_examples = raw_examples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        clean = generation_eval(model, self.raw_examples)
        aug = generation_eval(model, self.raw_examples, transform=augment_example)
        print(f"[gen-eval] em={clean['exact_match']:.3f} exec={clean['execution_match']:.3f} "
              f"valid={clean['execution_valid']:.3f} em_aug={aug['exact_match']:.3f} "
              f"| new_tokens={clean['avg_new_tokens']:.1f} lat={clean['avg_latency_s']*1e3:.0f}ms")
        wandb.log({
            "eval/generation_exact_match": clean["exact_match"],
            "eval/generation_exact_match_aug": aug["exact_match"],
            "eval/execution_match": clean["execution_match"],
            "eval/execution_valid": clean["execution_valid"],
            "eval/ar_avg_new_tokens": clean["avg_new_tokens"],
            "eval/avg_forward_steps": clean["avg_forward_steps"],
            "eval/ar_avg_latency_ms": clean["avg_latency_s"] * 1e3,
            "eval/avg_latency_ms": clean["avg_latency_s"] * 1e3,
        }, step=state.global_step)
        model.train()


# 7. Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=DATALOADER_WORKERS > 0,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
    report_to=["wandb"],
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
)

train_ds = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
val_ds = dataset["test"].select(range(min(VAL_SIZE, len(dataset["test"]))))
gen_eval_examples = [dataset["test"][i] for i in range(min(GEN_EVAL_SIZE, len(dataset["test"])))]

trainer = ARTrainer(
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
