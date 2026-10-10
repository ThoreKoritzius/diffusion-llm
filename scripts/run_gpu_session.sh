#!/usr/bin/env bash
# One rented-GPU session (H100 or GH200): every experiment in docs/gpu-session.md, unattended and resumable.
#
#   bash scripts/run_gpu_session.sh                       # all jobs
#   JOBS="isolation bench" bash scripts/run_gpu_session.sh  # a subset
#   SKIP_SMOKE=1 bash scripts/run_gpu_session.sh          # skip the per-job smoke tests
#
# Jobs (run in this order; a job whose output checkpoint already exists is skipped, so re-running resumes):
#   setup      venvs + deps (.venv-gpu: transformers 4.x; .venv-tf5: transformers 5 for Qwen3.5;
#              .venv-vllm: vLLM for the AR latency baseline, kept separate because it pins its own torch)
#   isolation  ModernBERT + prompt isolation, 2 epochs from checkpoints/diffusion-sql-modernbert, and a
#              control with the same 2 epochs but full attention (isolates the effect of the mask)
#   ettin150   Ettin-150M encoder as masked diffusion vs. Ettin-150M decoder as AR, same data and epochs
#   ettin400   same at 400M
#   qwen35     Qwen3.5-0.8B-Base AR fine-tune (strongest small AR baseline)
#   bench      quality + GPU latency for every checkpoint that exists (bench/), then a summary table
#
# Needs: checkpoints/diffusion-sql-modernbert (upload it, see docs/gpu-session.md). Optional: WANDB_API_KEY,
# INSTALL_FLASH_ATTN=1 (often compiles for 30+ min; only speeds up the full-attention control run).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

JOBS="${JOBS:-setup isolation ettin150 ettin400 qwen35 bench}"
EPOCHS="${EPOCHS:-3}"                        # epochs for the from-scratch fine-tunes (both arms)
ETTIN_ISOLATION="${ETTIN_ISOLATION:-auto}"   # auto: use the mask for Ettin diffusion only if it held up in job 'isolation'
SESSION_DIR="${SESSION_DIR:-results/gpu-session-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SESSION_DIR/logs" checkpoints

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-12}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ]; then export WANDB_MODE=offline; fi

PY="${PY:-.venv-gpu/bin/python}"
PY5="${PY5:-.venv-tf5/bin/python}"
PYV="${PYV:-.venv-vllm/bin/python}"
# Overrides used for a local CPU dry run of this script (see docs/gpu-session.md):
ETTIN_SIZES="${ETTIN_SIZES:-150 400}"
BENCH_DEVICE="${BENCH_DEVICE:-cuda}"
BENCH_N="${BENCH_N:-256}"
EXTRA_TRAIN_ENV="${EXTRA_TRAIN_ENV:-}"   # e.g. "MAX_TRAIN_STEPS=2 TRAIN_SIZE=16" for a dry run
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SESSION_DIR/session.log"; }
has_job() { [[ " $JOBS " == *" $1 "* ]]; }
done_ckpt() { [ -f "checkpoints/$1/config.json" ] && [ -f "checkpoints/$1/model.safetensors" ]; }

# train <name> <python> <script> [ENV=VAL ...]: smoke test (20 steps) then the full run, logged per job
train() {
  local name=$1 py=$2 script=$3; shift 3
  if done_ckpt "$name"; then log "skip $name (checkpoints/$name exists)"; return; fi
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    log "smoke $name"
    (cd src && env "$@" MAX_TRAIN_STEPS=20 TRAIN_SIZE=2000 VAL_SIZE=32 GEN_EVAL_SIZE=4 EVAL_STEPS=20 \
      WANDB_MODE=disabled OUTPUT_DIR="$REPO_DIR/.smoke/$name" "$REPO_DIR/$py" "$script") \
      > "$SESSION_DIR/logs/smoke_$name.log" 2>&1 || { log "!! smoke $name failed, see logs/smoke_$name.log"; return 0; }
    rm -rf ".smoke/$name"
  fi
  log "train $name"
  local t0=$SECONDS
  # shellcheck disable=SC2086
  (cd src && env "$@" $EXTRA_TRAIN_ENV RUN_NAME="$name" OUTPUT_DIR="$REPO_DIR/checkpoints/$name" "$REPO_DIR/$py" "$script") \
    > "$SESSION_DIR/logs/train_$name.log" 2>&1 || { log "!! train $name failed, see logs/train_$name.log"; return 0; }
  log "done $name in $(( (SECONDS - t0) / 60 )) min"
}

# bench_one <arm> <checkpoint> [extra bench args]: quality + latency on GPU, batch 1
bench_one() {
  local arm=$1 ckpt=$2; shift 2
  done_ckpt "$ckpt" || return 0
  $PY -m bench.run --arm "$arm" --model "checkpoints/$ckpt" --device "$BENCH_DEVICE" --n "$BENCH_N" \
    --name "$ckpt-${BENCH_TAG:-gpu}" --out "$SESSION_DIR/bench" "$@" >> "$SESSION_DIR/logs/bench.log" 2>&1 \
    || log "!! bench $ckpt $* failed (see logs/bench.log)"
}

# ---------------------------------------------------------------------------------------------------------------
if has_job setup; then
  log "=== setup on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ($(uname -m))"
  for v in .venv-gpu .venv-tf5 .venv-vllm; do
    [ -x "$v/bin/python" ] || python3 -m venv "$v"
    $v/bin/pip install -q --upgrade pip
  done
  for v in .venv-gpu .venv-tf5; do
    $v/bin/pip install -q torch --index-url "$TORCH_INDEX"
  done
  .venv-gpu/bin/pip install -q -r requirements.txt -r bench/requirements.txt
  if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    .venv-gpu/bin/pip install -q flash-attn --no-build-isolation || log "flash-attn unavailable (sdpa is used)"
  fi
  # vLLM brings its own torch; x86_64 wheels exist, on GH200 (aarch64) this may fail -> AR uses --ar-compile only
  .venv-vllm/bin/pip install -q vllm "datasets>=4" || log "vLLM unavailable, AR GPU latency uses --ar-compile only"
  .venv-tf5/bin/pip install -q "transformers>=5" "datasets>=4" accelerate wandb sentencepiece \
    flash-linear-attention || log "!! transformers 5 env incomplete (qwen35 job may fail)"
  .venv-tf5/bin/pip install -q causal-conv1d --no-build-isolation || log "causal-conv1d unavailable (slower Qwen3.5)"
  $PY -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" \
    | tee -a "$SESSION_DIR/session.log"
fi

if has_job isolation; then
  [ -f checkpoints/diffusion-sql-modernbert/model.safetensors ] || {
    log "!! checkpoints/diffusion-sql-modernbert missing: upload it first (docs/gpu-session.md)"; exit 1; }
  common=(INIT_CHECKPOINT="$REPO_DIR/checkpoints/diffusion-sql-modernbert" NUM_EPOCHS=2 LEARNING_RATE=2e-5)
  train modernbert-isolated "$PY" train.py "${common[@]}" PROMPT_ISOLATION=1
  train modernbert-ft-control "$PY" train.py "${common[@]}" PROMPT_ISOLATION=0
fi

# Decide whether the Ettin diffusion arm uses the mask: keep it if exec dropped by <= 2 points vs. the control.
ettin_isolation_flag() {
  if [ "$ETTIN_ISOLATION" != "auto" ]; then echo "$ETTIN_ISOLATION"; return; fi
  if ! done_ckpt modernbert-isolated || ! done_ckpt modernbert-ft-control; then echo 1; return; fi
  BENCH_TAG=gate bench_one diffusion modernbert-isolated --window 64
  BENCH_TAG=gate bench_one diffusion modernbert-ft-control --window 64
  $PY - "$SESSION_DIR/bench" <<'PY'
import json, sys
d = sys.argv[1]
iso = json.load(open(f"{d}/modernbert-isolated-gate.summary.json"))["exec"]
ctl = json.load(open(f"{d}/modernbert-ft-control-gate.summary.json"))["exec"]
print(1 if iso >= ctl - 0.02 else 0)
PY
}

if [[ "$JOBS" == *ettin* ]]; then
  ETTIN_ISO=$(ettin_isolation_flag | tail -1)
  log "Ettin diffusion arms: prompt isolation = $ETTIN_ISO"
fi
for size in $ETTIN_SIZES; do
  has_job "ettin$size" || continue
  bs=128; [ "$size" = 400 ] && bs="${ETTIN400_BATCH:-64}"   # no unpadding with sdpa -> halve batch at 400M
  train "ettin$size-diffusion" "$PY" train.py MODEL_NAME="jhu-clsp/ettin-encoder-${size}m" \
    NUM_EPOCHS="$EPOCHS" PROMPT_ISOLATION="$ETTIN_ISO" BATCH_SIZE="$bs"
  train "ettin$size-ar" "$PY" train_ar.py MODEL_NAME="jhu-clsp/ettin-decoder-${size}m" \
    NUM_EPOCHS="$EPOCHS" LEARNING_RATE=5e-5
done

if has_job qwen35; then
  train qwen35-0.8b-ar "$PY5" train_ar.py MODEL_NAME=Qwen/Qwen3.5-0.8B-Base NUM_EPOCHS="$EPOCHS"
fi

if has_job bench; then
  log "=== bench (n=$BENCH_N, batch 1, $BENCH_DEVICE)"
  for c in diffusion-sql-modernbert modernbert-isolated modernbert-ft-control; do
    bench_one diffusion "$c" --window 64
    BENCH_TAG=gpu-bf16 bench_one diffusion "$c" --window 64 --dtype bfloat16
  done
  for size in $ETTIN_SIZES; do
    bench_one diffusion "ettin$size-diffusion" --window 64
    BENCH_TAG=gpu-bf16 bench_one diffusion "ettin$size-diffusion" --window 64 --dtype bfloat16
    bench_one ar "ettin$size-ar"
    if [ "$BENCH_DEVICE" = cuda ]; then
      BENCH_TAG=gpu-compiled bench_one ar "ettin$size-ar" --dtype bfloat16 --ar-compile
      PY="$PYV" BENCH_TAG=gpu-vllm bench_one ar "ettin$size-ar" --engine vllm
    fi
  done
  # Qwen3.5 needs transformers 5 -> its own venv (torch engine; vLLM support for qwen3_5 is version dependent)
  if done_ckpt qwen35-0.8b-ar; then
    PY="$PY5" BENCH_TAG=gpu-bf16 bench_one ar qwen35-0.8b-ar --dtype bfloat16
  fi
  $PY -m bench.report "$SESSION_DIR/bench" | tee "$SESSION_DIR/summary.md"
fi

log "=== finished. Results: $SESSION_DIR (download it + checkpoints/, see docs/gpu-session.md)"
