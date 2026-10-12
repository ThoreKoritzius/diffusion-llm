#!/usr/bin/env bash
# Ettin-400M encoder as masked diffusion with the recipe that worked at 150M: prompt isolation, 10 epochs, then one
# epoch of step distillation, then the GPU benchmark (CUDA graphs + eager). Waits for a running tmux session first.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-1}"; L="$SESSION_DIR/session.log"; O="$SESSION_DIR/bench-gpu"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
[ -n "${WAIT_FOR:-}" ] && while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 20; done
P="$REPO_DIR/.venv-gpu/bin/python"
train() { local out=$1; shift
  [ -f "checkpoints/$out/model.safetensors" ] && return 0
  log "train $out"
  (cd src && env "$@" RUN_NAME="$out" OUTPUT_DIR="$REPO_DIR/checkpoints/$out" "$P" train.py) \
    > "$SESSION_DIR/logs/train_$out.log" 2>&1 && log "done $out" || { log "!! $out failed"; return 1; }
}
train ettin400-diffusion-iso-10ep MODEL_NAME=jhu-clsp/ettin-encoder-400m PROMPT_ISOLATION=1 NUM_EPOCHS=10 \
  LEARNING_RATE=5e-5 BATCH_SIZE=32 GRAD_ACCUM=4 || exit 1
train ettin400-diffusion-iso-10ep-distilled INIT_CHECKPOINT="$REPO_DIR/checkpoints/ettin400-diffusion-iso-10ep" \
  PROMPT_ISOLATION=1 TRAJECTORY_MASKING=1 CERTAINTY_WEIGHT=0.5 NUM_EPOCHS=1 LEARNING_RATE=3e-5 BATCH_SIZE=32 GRAD_ACCUM=4
for c in ettin400-diffusion-iso-10ep ettin400-diffusion-iso-10ep-distilled; do
  [ -f "checkpoints/$c/model.safetensors" ] || continue
  "$P" -m bench.run --device cuda --n 256 --arm diffusion --window 64 --dtype bfloat16 --cuda-graphs \
    --model "checkpoints/$c" --name "$c-bf16-cudagraphs" --out "$O" >> "$O/bench.log" 2>&1 && log "bench $c cudagraphs"
  "$P" -m bench.run --device cuda --n 256 --arm diffusion --window 64 --dtype bfloat16 \
    --model "checkpoints/$c" --name "$c-bf16" --out "$O" >> "$O/bench.log" 2>&1 && log "bench $c eager"
done
log "400m: ALL DONE"
