"""
SQL Masked-Diffusion adaptation of a causal decoder (Qwen3.5-0.8B-Base) — v2.

Diffusion arm of the AR-vs-diffusion comparison. LLaDA-style absorbing-state
masked diffuser on the same data / tag scheme / 1-t ELBO as src/train.py, so
the eval harness (src/denoising.py) works unchanged.

v2 — fixes for the confounds diagnosed in the first run (pad=eos aliasing,
output-length collapse at step ~2000, causal->bidirectional adaptation shock):

  1. DISTINCT PAD. v1 aliased window-pad to EOS, so 76%% of the 128-token
     window taught "emit EOS" and the model couldn't separate "query ends
     here" from "filler" -> periodic all-pad output collapse. v2 adds a real
     <|pad|> token.
  2. WINDOW 128 -> 64. Median gold SQL is 27 tokens, p90=54. Halves the pad
     fraction (76%% -> ~52%%), doubles supervision density, halves decode cost.
  3. PAD-DOWNWEIGHTED LOSS (PAD_LOSS_WEIGHT, default 0.1). Pads are still
     predicted (that is how length is learned) but no longer dominate the
     objective — so eval_loss tracks SQL quality again instead of padding.
  4. CAUSAL->BIDIRECTIONAL ANNEALING (DiffuLLaMA-style). Instead of hard-
     switching the mask, future positions start at an additive bias of
     CAUSAL_NEG (effectively causal) and decay linearly to 0 over the first
     ANNEAL_FRAC of training. Removes the distribution shock of suddenly
     un-hiding the right context from weights pretrained on a triangle.
  5. SHIFTED PREDICTION (Dream-style). An AR head at position i was pretrained
     to predict token i+1. We keep that alignment: the target for masked
     position i is read from the logits at position i-1, in training AND in
     decoding (via a thin adapter around the model for denoise_steps).
  6. SDPA attention instead of eager. The padding/anneal mask is a plain
     additive 4D float mask, which SDPA accepts — ~2-3x faster than eager.
     (Returning None would be WRONG: SDPA falls back to is_causal=True.)
  7. Gentler optimization: lr 1e-5 (was 2e-5), warmup 10%%. New-token rows are
     mean-initialized by transformers' default mean_resizing on resize.
  8. Eval decodes with confidence_stop=CONF_STOP (default 0.9, the frontier
     sweet spot from the PAPL report) and a 12-step cap, logging avg forward
     passes + wall-clock latency — the numbers that must beat the AR arm.

NOTE: the saved checkpoint does NOT carry the mask patch or the shift. Any
consumer must re-apply make_bidirectional() + the shifted-logits adapter
(exactly as generation_eval does) or it will silently run causal/unshifted.

Still direct-to-SQL adaptation (no general-corpus diffusion pass first), so
treat the result as a lower bound on a full Dream/DiffuLLaMA-style adaptation.
"""

import contextlib
import os
import random
import time
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn.functional as F
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
from denoising import denoise_steps
from sql_benchmark import execution_match, normalize_sql

# 1. Config
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-0.8B-Base")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))    # sdpa + window 64 -> roomier
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))     # effective batch 64
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
SQL_WINDOW = int(os.environ.get("SQL_WINDOW", "64"))    # median gold 27, p90 54
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "100000"))
VAL_SIZE = int(os.environ.get("VAL_SIZE", "500"))
T_EPS = float(os.environ.get("T_EPS", "1e-3"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", "8"))
GEN_EVAL_SIZE = int(os.environ.get("GEN_EVAL_SIZE", "32"))
GEN_EVAL_STEPS = int(os.environ.get("GEN_EVAL_STEPS", "12"))   # cap; conf-stop adapts below it
CONF_STOP = float(os.environ.get("CONF_STOP", "0.9"))          # 0 disables early stop
PAD_LOSS_WEIGHT = float(os.environ.get("PAD_LOSS_WEIGHT", "0.1"))
ANNEAL_FRAC = float(os.environ.get("ANNEAL_FRAC", "0.3"))      # fraction of run to reach bidir
CAUSAL_NEG = -20.0   # additive bias on future positions at alpha=1 (e^-20 ~ 0: causal)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints/sql-diffusion-qwen0.8b")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ["WANDB_PROJECT"] = "sql-diffusion"
if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
    os.environ["WANDB_MODE"] = "offline"
    print("[wandb] no WANDB_API_KEY found -> logging offline")
wandb.init(project="sql-diffusion", name=f"diffusion-v2-{MODEL_NAME.split('/')[-1]}", config={
    "model": MODEL_NAME, "arm": "masked-diffusion-v2", "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
    "sql_window": SQL_WINDOW, "train_size": TRAIN_SIZE, "t_eps": T_EPS,
    "lr": LEARNING_RATE, "pad_loss_weight": PAD_LOSS_WEIGHT,
    "anneal_frac": ANNEAL_FRAC, "conf_stop": CONF_STOP,
    "shifted_prediction": True, "distinct_pad": True, "attn": "sdpa",
})

# 2. Data
dataset = load_dataset("gretelai/synthetic_text_to_sql")
for split in dataset.keys():
    dataset[split] = dataset[split].filter(lambda ex: ex["sql_prompt"].strip() != "")

# 3. Tokenizer: tags + mask + a DISTINCT pad (fix #1 — v1 aliased pad to EOS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LEN
ORIG_VOCAB = len(tokenizer)
TAGS = ['<PROMPT>', '</PROMPT>', '<CONTEXT>', '</CONTEXT>', '<SQL>', '</SQL>']
tokenizer.add_special_tokens({'additional_special_tokens': TAGS})
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '<|mask|>'})
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})  # unconditional: never reuse EOS
TAG_IDS = {t: tokenizer.convert_tokens_to_ids(t) for t in TAGS}
CLS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
SEP_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id
assert PAD_ID != tokenizer.eos_token_id, "pad must be distinct from eos (v1 confound)"


# 4. Token-level input construction (same scheme as src/train.py, window=64)
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
        head + [TAG_IDS['<PROMPT>']] + prompt_ids + [TAG_IDS['</PROMPT>'],
        TAG_IDS['<CONTEXT>']] + context_ids + [TAG_IDS['</CONTEXT>'],
        TAG_IDS['<SQL>']]
    )
    sql_start = len(ids)
    ids = ids + sql_ids + [TAG_IDS['</SQL>'], SEP_ID]
    sql_end = sql_start + SQL_WINDOW

    attention = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return {"input_ids": ids, "attention_mask": attention,
            "sql_start": sql_start, "sql_end": sql_end}


# 5. Model + annealed bidirectional conversion (fixes #4, #6)
# alpha=1 -> effectively causal (additive CAUSAL_NEG on future positions);
# alpha=0 -> fully bidirectional. MaskAnnealCallback decays it during training.
ANNEAL = {"alpha": 1.0}


def make_bidirectional(model):
    """Replace the causal mask with padding bias + alpha-annealed causal bias."""
    base = model.model if hasattr(model, "model") else model

    def _annealed_bias(attention_mask, input_tensor, kv_length=None):
        b, q = input_tensor.shape[0], input_tensor.shape[1]
        k = int(kv_length or (attention_mask.shape[-1] if attention_mask is not None else q))
        dtype, device = input_tensor.dtype, input_tensor.device
        bias = torch.zeros((b, 1, q, k), dtype=dtype, device=device)
        alpha = ANNEAL["alpha"]
        if alpha > 0.0 and q == k:
            # Soft causal bias: CAUSAL_NEG (not -inf) scaled by alpha, so the
            # right context fades IN smoothly instead of appearing at once.
            future = torch.triu(torch.ones(q, k, dtype=torch.bool, device=device), diagonal=1)
            bias = bias + future[None, None] * (alpha * CAUSAL_NEG)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask
            pad = attention_mask[:, :k] == 0
            bias = bias.masked_fill(pad[:, None, None, :], torch.finfo(dtype).min)
        return bias

    def _bidir_mask(attention_mask, input_tensor, *args, **kwargs):
        return _annealed_bias(attention_mask, input_tensor)

    patched = False
    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = _bidir_mask
        patched = True
    else:
        # Transformers 4.57 Qwen2 calls a module-level create_causal_mask directly
        # from Qwen2Model.forward, so patch that imported symbol for this model.
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

            def _create_annealed_mask(
                config, input_embeds, attention_mask, cache_position,
                past_key_values=None, position_ids=None, **kwargs,
            ):
                kv_length = attention_mask.shape[-1] if attention_mask is not None else input_embeds.shape[1]
                return _annealed_bias(attention_mask, input_embeds, kv_length=kv_length)

            qwen2_modeling.create_causal_mask = _create_annealed_mask
            patched = base.__class__.__module__ == qwen2_modeling.__name__
        except Exception:
            patched = False

    if not patched:
        raise RuntimeError(
            "Could not patch the backbone causal mask. Inspect the model forward "
            "path and add a padding-only mask hook before training diffusion."
        )
    model.config.is_causal = False
    return model


@torch.no_grad()
def verify_bidirectional(model):
    """Hard-fail if attention is still causal AT alpha=0: changing a LATER token
    must move an EARLIER position's logits. Catches a silently-unpatched mask."""
    model.eval()
    device = next(model.parameters()).device
    prev = ANNEAL["alpha"]
    ANNEAL["alpha"] = 0.0
    try:
        L = 8
        ids = torch.randint(0, 1000, (1, L), device=device)
        attn = torch.ones((1, L), dtype=torch.long, device=device)
        a = model(input_ids=ids, attention_mask=attn).logits[0, 2].float().clone()
        ids2 = ids.clone(); ids2[0, 6] = (ids2[0, 6] + 7) % 1000
        b = model(input_ids=ids2, attention_mask=attn).logits[0, 2].float()
        if torch.allclose(a, b, atol=1e-4):
            raise RuntimeError(
                "Attention is still CAUSAL after make_bidirectional() at alpha=0 — "
                "position 2's logits did not react to a change at position 6. The "
                "mask patch did not take; do not train (you'd get a causal model "
                "wearing a diffusion objective).")
    finally:
        ANNEAL["alpha"] = prev
    print("[bidir] verified: attention is bidirectional at alpha=0.")


# SDPA takes our additive 4D float mask and is ~2-3x faster than eager (fix #6).
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
except Exception as e:
    print(f"[attn] sdpa load failed ({e}); falling back to eager")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
# mean_resizing (transformers >=4.46 default) initializes the new rows to the
# embedding mean — much gentler than random init for <|mask|>/<|pad|>/tags.
model.resize_token_embeddings(len(tokenizer))
model = make_bidirectional(model)
verify_bidirectional(model)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False


class MaskAnnealCallback(TrainerCallback):
    """Linear alpha 1 -> 0 over the first ANNEAL_FRAC of training (fix #4)."""

    def on_step_begin(self, args, state, control, **kwargs):
        if ANNEAL_FRAC <= 0:
            ANNEAL["alpha"] = 0.0
            return
        total = max(1, state.max_steps)
        ANNEAL["alpha"] = max(0.0, 1.0 - state.global_step / (ANNEAL_FRAC * total))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["mask_anneal_alpha"] = round(ANNEAL["alpha"], 4)


# 6. Collator + ELBO loss with shifted prediction and pad down-weighting
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

        t = torch.rand(B) * (1.0 - T_EPS) + T_EPS
        for i, r in enumerate(rows):
            s_, e_ = r["sql_start"], r["sql_end"]
            span = e_ - s_
            masked = torch.rand(span) < t[i]
            if not masked.any():
                masked[torch.randint(span, (1,))] = True
            idx = torch.nonzero(masked, as_tuple=True)[0] + s_
            labels[i, idx] = input_ids[i, idx]
            input_ids[i, idx] = MASK_ID

        return {"input_ids": input_ids, "attention_mask": attention,
                "labels": labels, "loss_weights": 1.0 / t}
    return collate


train_collator = make_collator(augment=True)
eval_collator = make_collator(augment=False)


class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        B, L, V = logits.shape
        # Dream-style shift (fix #5): the AR-pretrained head at position i-1
        # predicts token i, so score masked position i with logits[:, i-1].
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        ce = F.cross_entropy(
            shifted_logits.reshape(-1, V).float(), shifted_labels.reshape(-1),
            reduction="none", ignore_index=-100,
        ).view(B, L - 1)
        # Pad down-weighting (fix #3): pads still teach length, but at 0.1x so
        # the objective (and eval_loss) tracks SQL tokens, not padding.
        tok_w = torch.where(
            shifted_labels == PAD_ID,
            torch.full_like(ce, PAD_LOSS_WEIGHT),
            torch.ones_like(ce),
        )
        per_example = (ce * tok_w).sum(dim=1) * weights.to(ce.device) / SQL_WINDOW
        loss = per_example.mean()
        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        original = self.data_collator
        self.data_collator = eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original


# 7. Generation eval — shifted adapter + confidence-stop decoding (fixes #5, #8)
class ShiftedForDenoise:
    """denoise_steps reads logits AT masked position i; with the Dream shift the
    prediction for i lives at i-1, so re-align by shifting logits right by one.
    Any checkpoint consumer needs this same adapter."""

    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return SimpleNamespace(logits=shifted)


@torch.no_grad()
def generation_eval(model, raw_examples, transform=None, n_steps=GEN_EVAL_STEPS):
    model.eval()
    device = next(model.parameters()).device
    denoiser = ShiftedForDenoise(model)
    rng = random.Random(0)
    hits, exec_hits, valid_exec, total_steps, total_time = 0, 0, 0, 0, 0.0
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
        used_steps = 0
        t0 = time.perf_counter()
        with autocast_ctx:
            for used_steps, _, _ in denoise_steps(
                denoiser, ids, attn, list(range(lo, hi)), MASK_ID,
                n_steps=n_steps, forbid_token_ids=list(TAG_IDS.values()),
                confidence_stop=CONF_STOP if CONF_STOP > 0 else None,
            ):
                pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_steps += used_steps + 1
        out_ids = [tid for tid in ids[0, lo:hi].tolist() if tid not in (PAD_ID, MASK_ID)]
        pred_sql = tokenizer.decode(out_ids, skip_special_tokens=True)
        if normalize_sql(pred_sql) == normalize_sql(s):
            hits += 1
        pred_valid, pred_exec_match = execution_match(c, s, pred_sql)
        valid_exec += int(pred_valid)
        exec_hits += int(pred_exec_match)
    n = max(1, len(raw_examples))
    return {
        "exact_match": hits / n,
        "execution_match": exec_hits / n,
        "execution_valid": valid_exec / n,
        "avg_forward_steps": total_steps / n,
        "avg_latency_s": total_time / n,
    }


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, raw_examples):
        self.raw_examples = raw_examples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        prev_cache = getattr(model.config, "use_cache", None)
        model.config.use_cache = False
        # NB: early in training alpha > 0, so generation runs partially causal —
        # numbers before the anneal completes are progress signal, not the model.
        clean = generation_eval(model, self.raw_examples)
        aug = generation_eval(model, self.raw_examples, transform=augment_example)
        model.config.use_cache = prev_cache
        print(f"[gen-eval] em={clean['exact_match']:.3f} exec={clean['execution_match']:.3f} "
              f"valid={clean['execution_valid']:.3f} em_aug={aug['exact_match']:.3f} "
              f"| steps={clean['avg_forward_steps']:.1f} lat={clean['avg_latency_s']*1e3:.0f}ms "
              f"alpha={ANNEAL['alpha']:.2f}")
        wandb.log({
            "eval/generation_exact_match": clean["exact_match"],
            "eval/generation_exact_match_aug": aug["exact_match"],
            "eval/execution_match": clean["execution_match"],
            "eval/execution_valid": clean["execution_valid"],
            "eval/diffusion_avg_steps": clean["avg_forward_steps"],
            "eval/avg_forward_steps": clean["avg_forward_steps"],
            "eval/diffusion_avg_latency_ms": clean["avg_latency_s"] * 1e3,
            "eval/avg_latency_ms": clean["avg_latency_s"] * 1e3,
            "eval/mask_anneal_alpha": ANNEAL["alpha"],
        }, step=state.global_step)
        model.train()


# 8. Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=DATALOADER_WORKERS > 0,
    dataloader_drop_last=True,
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

trainer = DiffusionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=train_collator,
    processing_class=tokenizer,
    callbacks=[MaskAnnealCallback(), GenerationEvalCallback(gen_eval_examples)],
)

trainer.train()
ANNEAL["alpha"] = 0.0  # saved model is meant to run fully bidirectional
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
wandb.finish()
