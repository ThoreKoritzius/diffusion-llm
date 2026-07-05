#!/usr/bin/env bash
# Setup + train Qwen3.5-0.8B on gretelai text2sql, AR or masked-diffusion arm.
# Same data / tags / eval as the ModernBERT model, so the two arms are comparable.
#
# Usage:
#   export WANDB_API_KEY=...                 # optional; without it -> offline logs
#   bash run_qwen.sh ar                      # autoregressive SFT (do this FIRST — cheapest)
#   bash run_qwen.sh diffusion               # masked-diffusion adaptation
#   bash run_qwen.sh both                    # AR, then diffusion, same data/eval
#   SKIP_SMOKE=1 bash run_qwen.sh ar         # skip the 20-step smoke test
#   MODEL_NAME=Qwen/Qwen2.5-Coder-0.5B bash run_qwen.sh diffusion   # code-specialized seed
#
# Best on the NGC PyTorch container (nvcr.io/nvidia/pytorch:24.10-py3); on a bare
# CUDA box it installs torch + deps itself.
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  ar|diffusion|both) ;;
  *) echo "usage: bash run_qwen.sh {ar|diffusion|both}"; exit 1 ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
mkdir -p "$HF_HOME"

echo "=== requested arm=$ARM model=$MODEL_NAME ==="
echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi failed — is this a GPU box?"; exit 1; }

# --- deps ---
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "=== Installing CUDA PyTorch (cu124) ==="
  python3 -m pip install --upgrade pip
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi
echo "=== Installing project deps ==="
python3 -m pip install -r requirements.txt
# Qwen3.5 support needs a recent transformers; upgrade if the base won't load.
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL_NAME')" 2>/dev/null || {
  echo "=== Upgrading transformers for $MODEL_NAME ==="
  python3 -m pip install -U 'transformers>=4.57.0,<5' accelerate; }

python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| bf16={torch.cuda.is_bf16_supported()} | dev={torch.cuda.get_device_name(0)}")
PY

run_one() {
  local arm="$1"
  local script default_out out_dir
  case "$arm" in
    ar)        script="src/train_ar.py";             default_out="sql-ar-qwen0.8b" ;;
    diffusion) script="src/train_diffusion_qwen.py"; default_out="sql-diffusion-qwen0.8b" ;;
  esac
  out_dir="${OUTPUT_DIR:-$default_out}"

  echo "=== arm=$arm model=$MODEL_NAME -> $out_dir ==="

  # --- smoke test: exercises data + (for diffusion) the bidirectional patch/verify
  #     + eval path in ~1 min, so a config/OOM/mask error fails fast, not hours in. ---
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "=== Smoke test ($arm, 20 steps, no wandb) ==="
    MAX_TRAIN_STEPS=20 TRAIN_SIZE=2000 EVAL_STEPS=20 GEN_EVAL_SIZE=4 \
      OUTPUT_DIR="${out_dir}-smoke" WANDB_MODE=disabled python3 "$script"
    echo "=== Smoke test passed ($arm) ==="
  fi

  # --- real run ---
  echo "=== Starting full training ($arm) ==="
  OUTPUT_DIR="$out_dir" python3 "$script" 2>&1 | tee "train_${arm}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Done ($arm). Model in $REPO_DIR/$out_dir ==="
}

if [ "$ARM" = "both" ]; then
  unset OUTPUT_DIR
  run_one ar
  run_one diffusion
else
  run_one "$ARM"
fi

echo "=== All requested runs complete. scp model dirs back, plus wandb/ if you logged offline (wandb sync). ==="
