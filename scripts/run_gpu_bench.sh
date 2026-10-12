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

# Trimmed to what matters (GPU time is billed): bf16 throughout; diffusion full vs. prompt cache; AR compiled + vLLM.
for c in diffusion-sql-modernbert modernbert-ft-control ettin150-diffusion; do
  have $c && B $P "$c-bf16" --arm diffusion --model checkpoints/$c --window 64 --dtype bfloat16
done
for c in modernbert-isolated modernbert-isolated-distilled ettin150-diffusion-iso-10ep ettin150-diffusion-iso-10ep-distilled; do
  have $c || continue
  B $P "$c-bf16" --arm diffusion --model checkpoints/$c --window 64 --dtype bfloat16
  B $P "$c-bf16-cached" --arm diffusion --model checkpoints/$c --window 64 --prompt-cache --dtype bfloat16
done
have sql-ar-qwen25-coder-0.5b-ar && B $P "sql-ar-qwen25-coder-0.5b-ar-fp32-eager" --arm ar --model checkpoints/sql-ar-qwen25-coder-0.5b-ar
for c in ettin150-ar ettin150-ar-10ep sql-ar-qwen25-coder-0.5b-ar; do
  have $c || continue
  B $P "$c-bf16-compiled" --arm ar --model checkpoints/$c --dtype bfloat16 --ar-compile
  B $V "$c-vllm" --arm ar --engine vllm --model checkpoints/$c
done
$P -m bench.report "$OUT" > "$OUT/summary.md" && cat "$OUT/summary.md"
