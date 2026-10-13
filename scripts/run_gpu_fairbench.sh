#!/usr/bin/env bash
# Fair GPU latency benchmark: both arms with the same tooling (torch.compile "reduce-overhead" = inductor kernel
# fusion + CUDA graphs, bf16, batch 1), next to AR's best engine (vLLM, run separately). Then optionally starts the
# next training job so the GPU doesn't idle.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-1}"; L="$SESSION_DIR/session.log"; O="$SESSION_DIR/bench-gpu"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false
log() { echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
[ -n "${WAIT_FOR:-}" ] && while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 15; done
P=.venv-gpu/bin/python
B() { local name=$1; shift
  [ -f "$O/$name.summary.json" ] && return 0
  "$P" -m bench.run --device cuda --n 256 --dtype bfloat16 --cuda-graphs --name "$name" --out "$O" "$@" >> "$O/bench.log" 2>&1 \
    && log "fairbench $name" || log "!! fairbench $name failed"; }
log "fairbench: start"
for c in modernbert-isolated-distilled ettin150-diffusion-iso-10ep-distilled; do
  DIFF_COMPILE=1 B "$c-bf16-compiled-graphs" --arm diffusion --window 64 --model "checkpoints/$c"
done
for c in smollm135-ar ettin150-ar-10ep sql-ar-qwen25-coder-0.5b-ar; do
  [ -f "checkpoints/$c/model.safetensors" ] && B "$c-bf16-compiled-graphs" --arm ar --model "checkpoints/$c"
done
log "fairbench: ALL DONE"
[ -n "${THEN:-}" ] && bash -c "$THEN"
