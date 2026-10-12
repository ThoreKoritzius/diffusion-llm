#!/usr/bin/env bash
# Fewer-steps training (trajectory masking + certainty forcing, see src/train.py) for the two prompt-isolated models,
# 1 epoch each, in parallel (32 x 4 = effective batch 128 keeps both within 80 GB).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-1}"; mkdir -p "$SESSION_DIR/logs"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SESSION_DIR/session.log"; }
distill() { local base=$1 out=$2
  [ -f "checkpoints/$base/model.safetensors" ] || { log "!! distill: checkpoints/$base missing"; return 0; }
  [ -f "checkpoints/$out/model.safetensors" ] && return 0
  (cd src && INIT_CHECKPOINT="$REPO_DIR/checkpoints/$base" PROMPT_ISOLATION=1 TRAJECTORY_MASKING=1 CERTAINTY_WEIGHT=0.5 \
     NUM_EPOCHS=1 LEARNING_RATE=3e-5 BATCH_SIZE=32 GRAD_ACCUM=4 RUN_NAME="$out" \
     OUTPUT_DIR="$REPO_DIR/checkpoints/$out" "$REPO_DIR/.venv-gpu/bin/python" train.py) \
     > "$SESSION_DIR/logs/train_$out.log" 2>&1 && log "done $out" || log "!! $out failed"
}
log "distill: modernbert-isolated || ettin150-diffusion-iso-10ep (1 epoch each)"
distill modernbert-isolated modernbert-isolated-distilled &
distill ettin150-diffusion-iso-10ep ettin150-diffusion-iso-10ep-distilled &
wait
