#!/usr/bin/env bash
# GPU benchmark of every trained checkpoint (run on an otherwise idle GPU; timings are batch 1, n=256).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
OUT="${OUT:-results/gpu-session-1/bench-gpu}"; mkdir -p "$OUT"
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false
B() { local py=$1 name=$2; shift 2
  [ -f "$OUT/$name.summary.json" ] && return 0
  "$py" -m bench.run --device cuda --n 256 --name "$name" --out "$OUT" "$@" >> "$OUT/bench.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] $name" || echo "[$(date +%H:%M:%S)] !! $name failed"; }
P=.venv-gpu/bin/python; V=.venv-vllm/bin/python
have() { [ -f "checkpoints/$1/model.safetensors" ]; }

# diffusion: fp32 + bf16, full forward; prompt-isolated checkpoints additionally with the prompt cache
for c in diffusion-sql-modernbert modernbert-ft-control ettin150-diffusion; do
  have $c || continue
  B $P "$c-fp32" --arm diffusion --model checkpoints/$c --window 64
  B $P "$c-bf16" --arm diffusion --model checkpoints/$c --window 64 --dtype bfloat16
done
for c in modernbert-isolated ettin150-diffusion-iso-10ep; do
  have $c || continue
  B $P "$c-fp32" --arm diffusion --model checkpoints/$c --window 64
  B $P "$c-fp32-cached" --arm diffusion --model checkpoints/$c --window 64 --prompt-cache
  B $P "$c-bf16-cached" --arm diffusion --model checkpoints/$c --window 64 --prompt-cache --dtype bfloat16
done
# AR: HF eager fp32, bf16 + static-cache torch.compile, vLLM (best-in-class serving)
for c in ettin150-ar ettin150-ar-10ep sql-ar-qwen25-coder-0.5b-ar; do
  have $c || continue
  B $P "$c-fp32" --arm ar --model checkpoints/$c
  B $P "$c-bf16-compiled" --arm ar --model checkpoints/$c --dtype bfloat16 --ar-compile
  B $V "$c-vllm" --arm ar --engine vllm --model checkpoints/$c
done
$P -m bench.report "$OUT" > "$OUT/summary.md" && cat "$OUT/summary.md"
