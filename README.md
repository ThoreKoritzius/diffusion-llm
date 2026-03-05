# Diffusion-LLM

A masked language model implementing diffusion-style text denoising for SQL generation. This repo includes training and a web playground for iterative denoising inference.

![Diffusion Example](examples/diffusion_example.gif)

## Features

- Diffusion-style masking and denoising for SQL spans.
- Training pipeline (`src/train.py`) on text-to-SQL data.
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

### 3. Run inference UI locally

```bash
python src/inference.py --model-dir /absolute/path/to/model
```

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
- Max steps: `24`.
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
