#!/usr/bin/env bash
# One-shot setup + train for a rented 1x GH200 (aarch64 + Hopper sm_90).
#
# Usage:
#   export WANDB_API_KEY=...        # optional; without it, logs go offline
#   bash run_gh200.sh               # full run (smoke test first, then real train)
#   SKIP_SMOKE=1 bash run_gh200.sh  # skip the smoke test
#   USE_TORCH_COMPILE=0 bash run_gh200.sh   # disable torch.compile if it errors
#   OUTPUT_DIR=diffusion-sql-modernbert-block bash run_gh200.sh
#
# Assumes: an NVIDIA CUDA box with internet and python3. Best on the NGC
# PyTorch container (nvcr.io/nvidia/pytorch:24.10-py3) which already ships an
# aarch64 CUDA torch + flash-attn; on a bare box this installs them itself.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Keep HF + datasets cache on local disk so re-runs don't re-download.
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"  # inductor breaks ModernBERT+flash-attn+bf16
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-12}"
export SQL_GENERATION_MODE="${SQL_GENERATION_MODE:-block}"
export SQL_WINDOW="${SQL_WINDOW:-128}"
export SQL_BLOCK_SIZE="${SQL_BLOCK_SIZE:-32}"
export EOS_TOKEN_BIAS="${EOS_TOKEN_BIAS:-1.5}"
export TRAIN_SIZE="${TRAIN_SIZE:-100000}"
export VAL_SIZE="${VAL_SIZE:-500}"
export EMPTY_CONTEXT_RATE="${EMPTY_CONTEXT_RATE:-0.02}"
export CUSTOM_SQL_TRAIN_JSONL="${CUSTOM_SQL_TRAIN_JSONL:-data/custom_sql_train.jsonl}"
export CUSTOM_SQL_EVAL_JSONL="${CUSTOM_SQL_EVAL_JSONL:-data/custom_sql_eval.jsonl}"
export CUSTOM_SQL_MIX_RATE="${CUSTOM_SQL_MIX_RATE:-0.10}"
export CUSTOM_SQL_EVAL_SIZE="${CUSTOM_SQL_EVAL_SIZE:-120}"
export GEN_EVAL_SIZE="${GEN_EVAL_SIZE:-32}"
export GEN_EVAL_STEPS="${GEN_EVAL_STEPS:-24}"
export OUTPUT_DIR="${OUTPUT_DIR:-diffusion-sql-modernbert-block-v2}"
mkdir -p "$HF_HOME"

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi failed — is this a GPU box?"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
# Use a venv unless we're inside the NGC container (torch already present).
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "=== Installing CUDA PyTorch (aarch64, cu124) ==="
  python3 -m pip install --upgrade pip
  # aarch64 CUDA wheels live on the cu124 index.
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

echo "=== Installing project deps ==="
# torch is already satisfied above, so this won't pull the CPU wheel.
python3 -m pip install -r requirements.txt

# flash-attn gives ModernBERT a big speedup; best-effort, falls back to SDPA.
# Building it from source on aarch64 is slow; set SKIP_FLASH=1 to use SDPA.
if [ "${SKIP_FLASH:-0}" = "1" ]; then
  echo "=== SKIP_FLASH=1 — using SDPA, not installing flash-attn ==="
elif python3 -c "import flash_attn" 2>/dev/null; then
  echo "=== flash-attn already present ==="
else
  echo "=== Attempting flash-attn (best effort; skip on failure) ==="
  python3 -m pip install flash-attn --no-build-isolation || \
    echo "!! flash-attn unavailable — ModernBERT will use SDPA (slower but fine)"
fi

python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| bf16={torch.cuda.is_bf16_supported()} | dev={torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
# 2. Smoke test — exercise dataloader + compile + eval path in a few steps
#    so a config/OOM/compile error fails in ~1 min, not 2 hours in.
# ---------------------------------------------------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "=== Smoke test (20 steps, no wandb) ==="
  MAX_TRAIN_STEPS=20 WANDB_MODE=disabled OUTPUT_DIR="${OUTPUT_DIR}-smoke" python3 src/train.py
  echo "=== Smoke test passed ==="
fi

# ---------------------------------------------------------------------------
# 3. Real run
# ---------------------------------------------------------------------------
echo "=== Starting full training ==="
echo "mode=$SQL_GENERATION_MODE sql_window=$SQL_WINDOW block_size=$SQL_BLOCK_SIZE eos_bias=$EOS_TOKEN_BIAS train_size=$TRAIN_SIZE custom_mix=$CUSTOM_SQL_MIX_RATE empty_context_rate=$EMPTY_CONTEXT_RATE output=$OUTPUT_DIR"
python3 src/train.py 2>&1 | tee "train_$(date +%Y%m%d_%H%M%S).log"

if [ "${RUN_POST_EVAL:-1}" = "1" ]; then
  echo "=== Post-training inspect eval ==="
  python3 inspect_eval.py "$OUTPUT_DIR" "${INSPECT_EVAL_N:-50}" "${INSPECT_EVAL_STEPS:-24}" "${INSPECT_SQL_LEN:-$SQL_WINDOW}" || \
    echo "!! inspect_eval failed; training output is still available"
fi

echo "=== Done. Model in $REPO_DIR/$OUTPUT_DIR ==="
echo "scp that dir back, plus the wandb/ run if you logged offline (wandb sync)."
