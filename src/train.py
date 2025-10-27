"""
SQL Diffusion-Style Masked Language Model Training Script

- Loads synthetic text-to-SQL dataset.
- Prepares masked LM training on the SQL span of each input using Roberta.
- Custom masking/probabilities for "diffusion-style" training.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
)
import wandb

# 1. Hyperparameters and Config
N_STEPS =       10
NUM_EPOCHS =    10
BATCH_SIZE =    32
MAX_LEN =       512
TRAIN_SIZE =    10_000
VAL_SIZE =      100
SQL_WINDOW =    128

os.environ["WANDB_PROJECT"] = "sql-diffusion"
wandb.init(project="sql-diffusion", name="roberta-base-diffusion-style", config={
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "n_steps": N_STEPS,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE
})

# Masking schedule: linearly spaced from 1/N_STEPS .. 1.0
mask_probs = [(i + 1) / N_STEPS for i in range(N_STEPS - 1, -1, -1)]


# 2. Load and Filter Dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql")
# Remove empty prompts
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")
# Filter for simple SQL (exclude JOINs for now)
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: "join" not in ex["sql"].lower())

# 3. Tokenizer Setup
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer.model_max_length = MAX_LEN
special_tokens_dict = {'additional_special_tokens': [
    '<PROMPT>', '</PROMPT>',
    '<CONTEXT>', '</CONTEXT>',
    '<SQL>', '</SQL>'
]}
tokenizer.add_special_tokens(special_tokens_dict)

# 4. Input Formatting
def build_text(example):
    """Constructs input like:
        <PROMPT>...</PROMPT> <CONTEXT>...</CONTEXT> <SQL>...</SQL>
    """
    prompt = example["sql_prompt"]
    context = example.get("sql_context", "")
    sql = example.get("sql", "")
    sql_ids = tokenizer(sql, add_special_tokens=False)["input_ids"]
    sql_len = len(sql_ids)
    if sql_len < SQL_WINDOW:
        sql_ids = sql_ids + [tokenizer.pad_token_id]*(SQL_WINDOW - sql_len)
    else:
        sql_ids = sql_ids[:SQL_WINDOW]
    text = f"<PROMPT>{prompt}</PROMPT> <CONTEXT>{context}</CONTEXT> <SQL>{tokenizer.decode(sql_ids)}</SQL>"
    return {"text": text}


# 5. Tokenization & SQL Span Indexing
def tokenize_function(example):
    """
    Tokenize the formatted text.
    Identify token span for the SQL field between <SQL>...</SQL>.
    """
    tok = tokenizer(
        example["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offsets = tok.pop("offset_mapping")
    input_ids = tok["input_ids"]

    # Try to find special tokens directly (single-token assumption preferred)
    sql_start_id = tokenizer.convert_tokens_to_ids("<SQL>")
    sql_end_id = tokenizer.convert_tokens_to_ids("</SQL>")
    try:
        sql_open = input_ids.index(sql_start_id) + 1
        sql_close = input_ids.index(sql_end_id)
    except ValueError:
        # Fallback: use char offsets if tokens not directly matchable
        text = example["text"]
        open_char = text.find("<SQL>")
        close_char = text.find("</SQL>")
        if open_char == -1 or close_char == -1:
            sql_open, sql_close = 0, 0
        else:
            content_start_char = open_char + len("<SQL>")
            content_end_char = close_char
            token_start = next((i for i, (s, e) in enumerate(offsets) if e > content_start_char), None)
            tokens_covering = [i for i, (s, e) in enumerate(offsets) if s < content_end_char and e > content_start_char]
            if token_start is None or not tokens_covering:
                sql_open, sql_close = 0, 0
            else:
                sql_open = token_start
                sql_close = tokens_covering[-1] + 1  # exclusive end

    tok["sql_start"] = int(sql_open)
    tok["sql_end"] = int(sql_close)
    return tok


# 6. Prepare Dataset
tokenized = dataset.map(build_text)
for i in range(10):
    print(tokenized["train"][i]["text"])
    print("=" * 80)
tokenized = tokenized.map(tokenize_function, remove_columns=["text"])


# 7. Model Setup
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.resize_token_embeddings(len(tokenizer))


# 8. Custom Data Collator: Diffusion-Style Masking on SQL Span
def diffusion_collator(features):
    """
    For each example:
        - Randomly select a mask rate (from mask_probs).
        - Mask (randomly within the SQL span tokens) N tokens at the chosen rate.
        - Set labels = token at masked spots, -100 elsewhere (ignored for loss).
    """
    if "sql_start" not in features[0] or "sql_end" not in features[0]:
        raise KeyError("sql_start/sql_end missing; ensure remove_unused_columns=False")

    batch_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch_attention = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    labels = batch_input_ids.clone()
    B, L = batch_input_ids.shape

    sql_starts = [int(f["sql_start"]) for f in features]
    sql_ends = [int(f["sql_end"]) for f in features]

    mask_positions = torch.zeros_like(batch_input_ids, dtype=torch.bool)
    probs = torch.rand(len(features))  # random number [0, 1) per sample
    mask_rate_indices = (probs * len(mask_probs)).long()
    ps = [mask_probs[i] for i in mask_rate_indices]

    for i in range(B):
        s, e = sql_starts[i], sql_ends[i]
        if e > s:
            length = e - s
            num_to_mask = max(1, int(round(ps[i] * length)))
            idxs = torch.randperm(length)[:num_to_mask] + s
            mask_positions[i, idxs] = True

    # Mask positions outside SQL should never be selected.
    batch_input_ids[mask_positions] = tokenizer.mask_token_id
    labels[~mask_positions] = -100    # Ignore non-masked for loss

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": labels,
    }


# 9. Training Arguments
training_args = TrainingArguments(
    output_dir="diffusion-sql",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    save_total_limit=10,
    logging_steps=10,
    remove_unused_columns=False,
    logging_strategy="steps",
    report_to=["wandb"],
    eval_strategy="steps",
    eval_steps=100,
)


# 10. Dataset Slices (small subset for quick test)
small_train = tokenized["train"].select(range(TRAIN_SIZE))
small_val = tokenized["test"].select(range(VAL_SIZE))

# 11. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_val,
    data_collator=diffusion_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("diffusion-sql")
tokenizer.save_pretrained("diffusion-sql")

wandb.finish()
