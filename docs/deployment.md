# Deployment (CPU-safe public playground)

How the live demo at [diffusion.lethonium.com](https://diffusion.lethonium.com/) is served: Docker image, compose stack, runtime limits, API and CPU tuning. Moved here from the main README.

## Docker

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
- Max steps: `64`.
- Max max_len: `512`.
- Max sql_len: `256`.
- Adaptive early-stop threshold (`CONFIDENCE_STOP`): `0.9` (on; `0` disables).
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
- `--confidence-stop` (adaptive early-stop threshold; default `0.9`, `0` disables)
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

Adaptive early-stopping already cuts average steps automatically; for further
speedups lower the step cap in requests (for example `steps=8`) at a small
quality cost.
