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
5. **Adaptive early stop** (on by default): on each step, any masked position
   whose top-token probability is already `>= confidence_stop` is also committed,
   so easy queries empty the window in far fewer forward passes. The step slider
   becomes a *maximum*; the actual number of steps adapts to difficulty.

### Decoding efficiency (steps vs. autoregressive)

Each diffusion step is one forward pass over the whole sequence but commits many
tokens at once, whereas autoregressive decoding needs roughly one forward pass
per output token. Measured with `inspect_eval.py` on 58 held-out examples
(8 sanity + 50 gretelai test), greedy decoding, 128-token SQL window:

**Fixed steps** (no early stop) — accuracy plateaus around 12–16 steps; beyond
that adds cost without quality, and at 24 steps the model needs *more* passes
than autoregressive would:

| steps | exact | soft (Jaccard ≥ .9) | speedup vs. autoregressive |
|------:|:-----:|:-------------------:|:--------------------------:|
| 4     | 0.50  | 0.50  | 4.7× |
| 8     | 0.55  | 0.60  | 2.3× |
| 12    | 0.65  | 0.65  | 1.6× |
| 16    | 0.60  | 0.60  | 1.2× |
| 24    | 0.65  | 0.65  | 0.8× |

**Adaptive early stop** (cap = 16 steps) — fewer average passes *and* higher
accuracy, because committing confident tokens sooner gives the model better
context to resolve the rest:

| `confidence_stop` | exact | soft | avg steps used |
|:-----------------:|:-----:|:----:|:--------------:|
| off               | 0.328 | 0.345 | 16.0 |
| 0.95              | 0.362 | 0.379 | 10.0 |
| **0.90** (default)| **0.397** | **0.414** | **8.8** |

Easy queries finish in ~4 passes, hard ones use up to the cap. Defaults:
`default_steps = 20` (the cap; early stop adapts down from there),
`CONFIDENCE_STOP = 0.9`. Toggle per run in the playground's Advanced panel, or
globally via `--confidence-stop` / `CONFIDENCE_STOP` (set `0` to disable).

## Features

- LLaDA-style masked diffusion training with `1/t`-weighted ELBO loss.
- Confidence-based parallel decoding shared by training eval and both UIs.
- Adaptive confidence early-stopping: ~1.8–3× fewer forward passes with no
  quality loss (toggleable per run; see Decoding efficiency above).
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
