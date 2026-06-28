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
import math
import os
import random
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
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
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "256"))
SQL_BLOCK_SIZE = int(os.environ.get("SQL_BLOCK_SIZE", "32"))
EOS_TOKEN_BIAS = float(os.environ.get("EOS_TOKEN_BIAS", "1.5"))
BLOCK_TRAINING_STYLE = os.environ.get("BLOCK_TRAINING_STYLE", "prefix").strip().lower()
if BLOCK_TRAINING_STYLE not in {"prefix", "full"}:
    raise ValueError("BLOCK_TRAINING_STYLE must be 'prefix' or 'full'")
SQL_GENERATION_MODE = os.environ.get("SQL_GENERATION_MODE", "block").strip().lower()
if SQL_GENERATION_MODE not in {"block", "fixed"}:
    raise ValueError("SQL_GENERATION_MODE must be 'block' or 'fixed'")
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "100000"))
VAL_SIZE = int(os.environ.get("VAL_SIZE", "500"))
EMPTY_CONTEXT_RATE = float(os.environ.get("EMPTY_CONTEXT_RATE", "0.05"))
CUSTOM_SQL_TRAIN_JSONL = os.environ.get("CUSTOM_SQL_TRAIN_JSONL", "").strip()
CUSTOM_SQL_EVAL_JSONL = os.environ.get("CUSTOM_SQL_EVAL_JSONL", "").strip()
CUSTOM_SQL_MIX_RATE = float(os.environ.get("CUSTOM_SQL_MIX_RATE", "0.20"))
CUSTOM_SQL_EVAL_SIZE = int(os.environ.get("CUSTOM_SQL_EVAL_SIZE", "120"))
CUSTOM_SQL_ONLY = os.environ.get("CUSTOM_SQL_ONLY", "0") == "1"
ENABLE_AUGMENTATION = os.environ.get("ENABLE_AUGMENTATION", "1") != "0"
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

GEN_EVAL_SIZE = int(os.environ.get("GEN_EVAL_SIZE", "32"))     # examples for generation-based exact-match eval
GEN_EVAL_STEPS = int(os.environ.get("GEN_EVAL_STEPS", "24"))   # denoising steps during eval

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "diffusion-sql-modernbert")

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
    "sql_block_size": SQL_BLOCK_SIZE,
    "sql_generation_mode": SQL_GENERATION_MODE,
    "block_training_style": BLOCK_TRAINING_STYLE,
    "eos_token_bias": EOS_TOKEN_BIAS,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "empty_context_rate": EMPTY_CONTEXT_RATE,
    "custom_sql_train_jsonl": CUSTOM_SQL_TRAIN_JSONL,
    "custom_sql_eval_jsonl": CUSTOM_SQL_EVAL_JSONL,
    "custom_sql_mix_rate": CUSTOM_SQL_MIX_RATE,
    "custom_sql_only": CUSTOM_SQL_ONLY,
    "t_eps": T_EPS,
    "filter_joins": FILTER_JOINS,
    "augmentation": ENABLE_AUGMENTATION,
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

custom_dataset = None
if CUSTOM_SQL_TRAIN_JSONL:
    data_files = {"train": CUSTOM_SQL_TRAIN_JSONL}
    if CUSTOM_SQL_EVAL_JSONL:
        data_files["test"] = CUSTOM_SQL_EVAL_JSONL
    custom_dataset = load_dataset("json", data_files=data_files)
    for split in custom_dataset.keys():
        custom_dataset[split] = custom_dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")


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
    """Build the token-level diffusion example.

    block mode, prefix style (new default for dynamic length):
        [CLS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL> prefix next_block [SEP] [PAD]...
        Only next_block is maskable. This matches inference, which repeatedly
        conditions on the already-generated SQL prefix and denoises the next
        masked block, including </SQL> when the query ends.

    fixed mode (legacy):
        [CLS] <PROMPT> p </PROMPT> <CONTEXT> c </CONTEXT> <SQL> sql+pads </SQL> [SEP] [PAD]...
        PADs inside the fixed SQL window are prediction targets, matching older
        checkpoints.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if SQL_GENERATION_MODE == "fixed":
        sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][:SQL_WINDOW]
        target_ids = sql_ids + [PAD_ID] * (SQL_WINDOW - len(sql_ids))
        prefix_ids = []
        suffix_ids = [TAG_IDS['</SQL>'], SEP_ID]
    elif BLOCK_TRAINING_STYLE == "prefix":
        full_sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][: max(1, SQL_WINDOW - 1)]
        full_target = full_sql_ids + [TAG_IDS['</SQL>']]
        # Sample the exact inference state: a known SQL prefix followed by one
        # unknown block. Bias toward prefix=0 so fully-empty generation is
        # trained often, but also cover later continuation/termination blocks.
        if len(full_target) <= 1 or random.random() < 0.35:
            prefix_len = 0
        else:
            prefix_len = random.randrange(0, len(full_target))
        prefix_ids = full_target[:prefix_len]
        target_ids = full_target[prefix_len : prefix_len + max(1, SQL_BLOCK_SIZE)]
        suffix_ids = [SEP_ID]
    else:
        sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"][: max(1, SQL_WINDOW - 1)]
        prefix_ids = []
        target_ids = sql_ids + [TAG_IDS['</SQL>']]
        suffix_ids = [SEP_ID]

    # Leave room for CLS, prompt/context tags, <SQL>, target span, and suffix.
    fixed_overhead = 1 + 5 + 1 + len(suffix_ids)
    budget = MAX_LEN - len(prefix_ids) - len(target_ids) - fixed_overhead
    if len(prompt_ids) + len(context_ids) > budget:
        budget = max(0, budget)
        context_ids = context_ids[: max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget]

    ids = (
        [CLS_ID, TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
        TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'],
        TAG_IDS['<SQL>']]
    )
    ids = ids + prefix_ids
    sql_start = len(ids)
    ids = ids + target_ids + suffix_ids
    sql_end = sql_start + len(target_ids)

    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "sql_start": sql_start,
        "sql_end": sql_end,
        "sql_span_len": sql_end - sql_start,
    }


# 5. Model Setup
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.config.diffusion_sql_mode = SQL_GENERATION_MODE
model.config.diffusion_sql_window = SQL_WINDOW
model.config.diffusion_sql_block_size = SQL_BLOCK_SIZE
model.config.diffusion_block_training_style = BLOCK_TRAINING_STYLE
model.config.diffusion_eos_token_bias = EOS_TOKEN_BIAS


# 6. Collator: Augment + Encode + Continuous-t Bernoulli Masking
def make_collator(augment: bool):
    def collate(features):
        rows = []
        for f in features:
            p, c, s = f["sql_prompt"], f.get("sql_context", ""), f.get("sql", "")
            if augment:
                p, c, s = augment_example(p, c, s)
                if c and random.random() < EMPTY_CONTEXT_RATE:
                    c = ""
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
            "span_lengths": torch.tensor([r["sql_span_len"] for r in rows], dtype=torch.float32),
        }
    return collate


train_collator = make_collator(augment=ENABLE_AUGMENTATION)
eval_collator = make_collator(augment=False)


# 7. Trainer with 1/t-weighted ELBO Loss
class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weights")
        span_lengths = inputs.pop("span_lengths")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        B, L, V = logits.shape
        ce = F.cross_entropy(
            logits.view(-1, V).float(), labels.view(-1),
            reduction="none", ignore_index=-100,
        ).view(B, L)
        # LLaDA objective: (1/t) * sum of masked-token CE, normalized by target span length.
        per_example = ce.sum(dim=1) * weights / span_lengths.clamp_min(1.0).to(ce.device)
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


def trim_generated_sql(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").replace("</SQL>", " ")).strip()
    semi = s.find(";")
    if semi >= 0:
        s = s[:semi]
    return s.strip()


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
        if SQL_GENERATION_MODE == "block":
            prompt_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            context_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            budget = MAX_LEN - SQL_WINDOW - 8
            if len(prompt_ids) + len(context_ids) > budget:
                context_ids = context_ids[: max(0, budget - len(prompt_ids))]
                prompt_ids = prompt_ids[:budget]
            head = (
                [CLS_ID, TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
                TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'],
                TAG_IDS['<SQL>']]
            )
            generated_ids = []
            block_len = max(1, min(SQL_BLOCK_SIZE, SQL_WINDOW))
            max_blocks = int(math.ceil(SQL_WINDOW / block_len))
            end_sql_id = TAG_IDS['</SQL>']
            forbid_ids = set(tokenizer.all_special_ids or [])
            forbid_ids.update(TAG_IDS.values())
            forbid_ids.discard(end_sql_id)
            forbid_ids.discard(MASK_ID)
            with autocast_ctx:
                for _block in range(max_blocks):
                    this_block_len = min(block_len, SQL_WINDOW - len(generated_ids))
                    if this_block_len <= 0:
                        break
                    ids_list = head + generated_ids + [MASK_ID] * this_block_len + [SEP_ID]
                    attn_list = [1] * len(ids_list)
                    if len(ids_list) < MAX_LEN:
                        attn_list += [0] * (MAX_LEN - len(ids_list))
                        ids_list += [PAD_ID] * (MAX_LEN - len(ids_list))
                    ids = torch.tensor([ids_list[:MAX_LEN]], dtype=torch.long, device=device)
                    attn = torch.tensor([attn_list[:MAX_LEN]], dtype=torch.long, device=device)
                    start = len(head) + len(generated_ids)
                    positions = list(range(start, min(start + this_block_len, MAX_LEN - 1)))
                    for _ in denoise_steps(
                        model, ids, attn, positions, MASK_ID,
                        n_steps=min(n_steps, len(positions)),
                        forbid_token_ids=list(forbid_ids),
                        bias_token_ids={end_sql_id: EOS_TOKEN_BIAS} if EOS_TOKEN_BIAS else None,
                    ):
                        pass
                    block_out = [int(tid) for tid in ids[0, start:start + len(positions)].tolist()]
                    if end_sql_id in block_out:
                        generated_ids.extend(block_out[:block_out.index(end_sql_id)])
                        break
                    generated_ids.extend(block_out)
            out_ids = [tid for tid in generated_ids if tid not in (PAD_ID, MASK_ID)]
            pred_sql = trim_generated_sql(tokenizer.decode(out_ids, skip_special_tokens=True))
        else:
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
            pred_sql = trim_generated_sql(tokenizer.decode(out_ids, skip_special_tokens=True))
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
if custom_dataset is not None and CUSTOM_SQL_ONLY:
    custom_train_size = min(len(custom_dataset["train"]), TRAIN_SIZE)
    train_ds = custom_dataset["train"].shuffle(seed=42).select(range(custom_train_size))
elif custom_dataset is not None:
    custom_train_size = min(len(custom_dataset["train"]), max(1, int(TRAIN_SIZE * CUSTOM_SQL_MIX_RATE)))
    base_train_size = max(1, TRAIN_SIZE - custom_train_size)
    train_parts = [
        dataset["train"].select(range(min(base_train_size, len(dataset["train"])))),
        custom_dataset["train"].shuffle(seed=42).select(range(custom_train_size)),
    ]
    train_ds = concatenate_datasets(train_parts).shuffle(seed=42)
else:
    train_ds = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))

if custom_dataset is not None and CUSTOM_SQL_ONLY and "test" in custom_dataset:
    val_ds = custom_dataset["test"].select(range(min(max(1, VAL_SIZE), len(custom_dataset["test"]))))
else:
    val_ds = dataset["test"].select(range(min(VAL_SIZE, len(dataset["test"]))))

if custom_dataset is not None and not CUSTOM_SQL_ONLY and "test" in custom_dataset:
    custom_eval_size = min(CUSTOM_SQL_EVAL_SIZE, len(custom_dataset["test"]))
    if custom_eval_size > 0:
        val_ds = concatenate_datasets([
            val_ds,
            custom_dataset["test"].select(range(custom_eval_size)),
        ])

gen_eval_examples = [val_ds[i] for i in range(min(GEN_EVAL_SIZE, len(val_ds)))]

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
