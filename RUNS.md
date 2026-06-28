# GH200 Run Notes

## Main 3h Experiment

Train the new block-mode checkpoint:

```bash
export WANDB_API_KEY=...   # optional
bash run_gh200.sh
```

Recommended upload to the remote box:

```bash
rsync -az --progress \
  --exclude '.git/' \
  --exclude '.hf_cache/' \
  --exclude 'diffusion-sql-modernbert/' \
  --exclude 'diffusion-sql-modernbert-block*/' \
  --exclude 'model/' \
  --exclude 'generated_gifs/' \
  /Users/thore/Documents/projects/diffusion-prod/diffusion-llm/ \
  user@host:/workspace/diffusion-llm/
```

This sends source code plus `data/custom_sql_*.jsonl`, but skips old local
checkpoints and caches. The remote run will download/cache the base model and
GretelAI dataset under `.hf_cache/`.

Defaults used by `run_gh200.sh`:

```bash
SQL_GENERATION_MODE=block
SQL_WINDOW=256
TRAIN_SIZE=100000
VAL_SIZE=500
EMPTY_CONTEXT_RATE=0.05
CUSTOM_SQL_TRAIN_JSONL=data/custom_sql_train.jsonl
CUSTOM_SQL_EVAL_JSONL=data/custom_sql_eval.jsonl
CUSTOM_SQL_MIX_RATE=0.20
CUSTOM_SQL_EVAL_SIZE=120
GEN_EVAL_SIZE=32
GEN_EVAL_STEPS=24
OUTPUT_DIR=diffusion-sql-modernbert-block
```

Effective default train mix:

- 80k `gretelai/synthetic_text_to_sql` examples.
- 20k custom failure-case examples from `data/custom_sql_train.jsonl`.
- 500 GretelAI validation examples plus 120 custom validation examples.
- 5% of augmented training rows have empty context.

The script runs a 20-step smoke test first, then full training, then:

```bash
python3 inspect_eval.py diffusion-sql-modernbert-block 50 24
```

## Useful Variants

Skip smoke test:

```bash
SKIP_SMOKE=1 bash run_gh200.sh
```

Legacy fixed-window control run:

```bash
SQL_GENERATION_MODE=fixed SQL_WINDOW=128 OUTPUT_DIR=diffusion-sql-modernbert-fixed bash run_gh200.sh
```

Faster partial run:

```bash
MAX_TRAIN_STEPS=2500 OUTPUT_DIR=diffusion-sql-modernbert-block-2500 bash run_gh200.sh
```

Run eval only after copying a checkpoint back:

```bash
python3 inspect_eval.py diffusion-sql-modernbert-block 100 24
python3 src/inference.py --model-dir diffusion-sql-modernbert-block
```

## What To Compare

- `inspect_eval.py` exact/soft on sanity + GretelAI test examples.
- W&B `eval/generation_exact_match`.
- Average steps used with adaptive early stop in the playground.
- Qualitative failures: joins, grouping, date logic, aliases, long SQL.

Spider2 should stay a stress test for now. For this model size, the useful target is strong easy/medium SQL plus a small controlled BigQuery-pattern eval.

`EMPTY_CONTEXT_RATE=0.05` randomly drops schema context for a small fraction of
training examples. Keep it low: it teaches prompt-only SQL priors without
training the model to ignore schemas.

`CUSTOM_SQL_MIX_RATE=0.20` means a default `TRAIN_SIZE=100000` run uses roughly
80k GretelAI examples and 20k custom examples. Increase the custom file size
before increasing the rate materially.

Tradeoffs:

- `SQL_GENERATION_MODE=block` is the new variable-length recipe. Use `fixed`
  only for legacy control runs.
- `SQL_WINDOW=256` gives longer SQL room but uses more target budget.
- `CUSTOM_SQL_MIX_RATE` improves known failure cases, but too high can overfit
  to template style.
- `EMPTY_CONTEXT_RATE` helps prompt-only demos, but too high teaches schema
  hallucination.
- `GEN_EVAL_SIZE` and `GEN_EVAL_STEPS` improve signal but slow training eval.
