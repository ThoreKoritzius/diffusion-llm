#!/usr/bin/env bash
# Phase 2 of the GPU session: longer training (diffusion is reported to keep improving for many epochs on limited
# data, AR saturates) with prompt isolation for the diffusion arm, and the matching AR control. Both continue from
# the 3-epoch Ettin-150M checkpoints for +7 epochs (10 total) and run in parallel (AR leaves most of the GPU idle).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-1}"; mkdir -p "$SESSION_DIR/logs"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SESSION_DIR/session.log"; }

log "phase2: ettin150 diffusion (+7 epochs, prompt isolation) || ettin150 AR (+7 epochs)"
(cd src && INIT_CHECKPOINT="$REPO_DIR/checkpoints/ettin150-diffusion" PROMPT_ISOLATION=1 NUM_EPOCHS=7 \
   LEARNING_RATE=5e-5 BATCH_SIZE=64 GRAD_ACCUM=2 RUN_NAME=ettin150-diffusion-iso-10ep \
   OUTPUT_DIR="$REPO_DIR/checkpoints/ettin150-diffusion-iso-10ep" "$REPO_DIR/.venv-gpu/bin/python" train.py) \
   > "$SESSION_DIR/logs/train_ettin150-diffusion-iso-10ep.log" 2>&1 &
DIFF=$!
(cd src && MODEL_NAME="$REPO_DIR/checkpoints/ettin150-ar" NUM_EPOCHS=7 LEARNING_RATE=2.5e-5 BATCH_SIZE=64 GRAD_ACCUM=1 \
   RUN_NAME=ettin150-ar-10ep OUTPUT_DIR="$REPO_DIR/checkpoints/ettin150-ar-10ep" "$REPO_DIR/.venv-gpu/bin/python" train_ar.py) \
   > "$SESSION_DIR/logs/train_ettin150-ar-10ep.log" 2>&1 &
AR=$!
wait $DIFF && log "done ettin150-diffusion-iso-10ep" || log "!! ettin150-diffusion-iso-10ep failed"
wait $AR && log "done ettin150-ar-10ep" || log "!! ettin150-ar-10ep failed"
