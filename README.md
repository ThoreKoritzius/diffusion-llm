# Diffusion-LLM

A masked diffusion language model (LLaDA-style) for text-to-SQL generation, built on ModernBERT. This repo includes training and a web playground for iterative denoising inference.

![Diffusion Example](examples/diffusion_example.gif)

## How It Works

### Training (`src/train.py`)

- Backbone: `answerdotai/ModernBERT-base` with an MLM head.
- Inputs are built at the token level:
  `[CLS] <PROMPT> ... </PROMPT> <CONTEXT> ... </CONTEXT> <SQL> sql + [PAD]s </SQL> [SEP]`
  The SQL span is a fixed 128-token window; `[PAD]` tokens inside it are real
  prediction targets, so the model learns output length.
- Forward (noising) process: per example a continuous mask ratio `t ~ U(0, 1]`
  is drawn and each SQL-span token is masked independently with probability `t`.
- Loss: cross-entropy on masked positions weighted by `1/t` (the discrete
  diffusion ELBO). This properly trains the fully-masked regime that
  generation starts from.
- Eval: besides MLM loss, generation-based exact match is computed by actually
  denoising validation examples (`eval/generation_exact_match` in wandb).

### Inference (`src/denoising.py`)

Confidence-based iterative unmasking (MaskGIT/LLaDA decoding):

1. The SQL window starts fully masked.
2. Each step predicts only the currently masked positions.
3. The most confident predictions are committed permanently (cosine schedule:
   few commits early, more as context fills in).
4. Committed tokens are never resampled, so generation converges instead of
   flickering. `top_k > 1` adds Gumbel noise to the commit order for diversity.

## Features

- LLaDA-style masked diffusion training with `1/t`-weighted ELBO loss.
- Confidence-based parallel decoding shared by training eval and both UIs.
- Generation-based exact-match eval logged to wandb during training.
- Public-ready Flask playground (`src/inference.py`) with SSE + polling fallback.
- Redis-backed queue + worker for stable queue position/ETA and run TTL.
- CPU safety controls for public deployment:
  - strict single-flight inference (default 1 active run)
  - bounded queue (default 3)
  - per-IP rate limits
  - request timeout and stale-run cleanup
  - GIF generation disabled by default

## Local Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
python src/train.py
```

Checkpoints and the final model are written to `diffusion-sql-modernbert/`.
Track `eval/generation_exact_match` in wandb for actual generation quality.

### 3. Run inference UI locally

```bash
python src/inference.py --model-dir /absolute/path/to/model
```

Both old (roberta) and new (ModernBERT) checkpoints load via the Auto classes.

## Docker Deployment (CPU-safe public playground)

### Build image

```bash
docker build -t diffusion-llm:latest .
```

### Run container (equivalent to local `python src/inference.py ...`)

```bash
docker run --rm -p 7860:7860 \
  -v /host/model:/models/model:ro \
  diffusion-llm:latest \
  --model-dir /models/model --host 0.0.0.0 --no-open
```

### Run with compose

Set model location:

```bash
export MODEL_DIR=/host/model
```

Start web + worker + redis:

```bash
docker compose up -d --build
```

Web service listens on `:7860` and is designed to be put behind Caddy/Nginx.

## Production Runtime Defaults

- Global request limit: `60/minute` per IP.
- `/start` limit: `5/minute` and `30/hour` per IP.
- Max concurrent runs: `1`.
- Max queued runs: `3`.
- Timeout per run: `90s`.
- Max prompt chars: `1000`.
- Max context chars: `8000`.
- Max steps: `48`.
- Max max_len: `512`.
- Max sql_len: `128`.
- Run TTL cleanup: `900s`.
- GIF generation: disabled unless `--enable-gif` is set.

## Public API Behavior

- `POST /start`
  - `200` with `{run_id, state, queue_position, eta_seconds, queue_token}` when accepted.
  - `400` for validation errors.
  - `429` when rate-limited.
  - `503` when queue is full.
- `GET /run/<run_id>?after=<n>`
  - Incremental run state + snapshot deltas.
- `GET /queue/<run_id>`
  - Queue position and ETA info.
- `GET /stream/<run_id>`
  - Server-sent events: `snapshot`, `status`, `done`.
- `POST /stop/<run_id>`
  - Requests cooperative cancellation.
- `GET /healthz`
  - Liveness/readiness info.

## Relevant CLI Options (`src/inference.py`)

- `--public` (bind `0.0.0.0`, disable browser auto-open)
- `--max-concurrent-runs`
- `--max-queued-runs`
- `--request-timeout-seconds`
- `--max-prompt-chars`
- `--max-context-chars`
- `--max-steps`
- `--max-max-len`
- `--max-sql-len`
- `--enable-gif`
- `--run-ttl-seconds`
- `--worker`
- `--redis-url`

## CPU Tuning Notes

The container sets:

- `TORCH_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `PYTHONUNBUFFERED=1`

This is intentional for predictable CPU load on a Linux CPU host.

For a faster profile (roughly ~2x on many CPU-only hosts), increase worker CPU/threads in `.env`:

```bash
TORCH_NUM_THREADS=4
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
WORKER_CPUS=3.0
WORKER_MEM_LIMIT=4g
```

And lower denoising steps in requests (for example `steps=6` instead of `10`) for near-linear speedups.
