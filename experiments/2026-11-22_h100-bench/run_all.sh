#!/usr/bin/env bash
# Stages, run in order (each can be rerun on its own; finished edits are skipped on resume):
#   ./run_all.sh smoke     ~15 min  5 edits through the key vLLM configs + 3 edits through each harness mode
#   ./run_all.sh vllm      ~60-80 min  all 9 vLLM configs on all 261 edits (Qwen3-8B)
#   ./run_all.sh harness   ~45-60 min  DFlash reference, copy+DFlash hybrids, AR-drafter replays (Qwen3-8B)
#   ./run_all.sh analyze   ~1 min   results/Qwen3-8B/summary.md with gates
#   ./run_all.sh core      vllm + harness + analyze
#   ./run_all.sh extra4b   ~40 min optional: Qwen3-4B cross-check against the Mac numbers (setup with WITH_4B=1)
set -uo pipefail
cd "$(dirname "$0")"
H=".venv-torch/bin/python harness.py"
T=Qwen/Qwen3-8B

harness_all() {
  local target=$1 limit=${2:-}
  local L=(); [ -n "$limit" ] && L=(--limit "$limit")
  $H dflash --target "$target" "${L[@]}"
  $H hybrid_oracle --target "$target" "${L[@]}"
  for m in 4 8 16; do $H hybrid_router --router-min $m --target "$target" "${L[@]}"; done
  $H ardraft --ar-drafter Qwen/Qwen3-0.6B --target "$target" "${L[@]}"
  $H ardraft --ar-drafter Qwen/Qwen3-1.7B --target "$target" "${L[@]}"
}

case "${1:-}" in
  smoke)
    LIMIT=5 ./run_vllm_suite.sh $T ar ngram31 draft06b7 eagle3_3 dflash15
    harness_all $T 3
    .venv-torch/bin/python analyze_h100.py Qwen3-8B
    mkdir -p results/smoke && mv results/Qwen3-8B/vllm_*.jsonl results/Qwen3-8B/harness_*.jsonl results/Qwen3-8B/summary.md results/smoke/
    echo "smoke OK if every config printed tau lines and results/smoke/summary.md has sections 1-4"
    ;;
  vllm) ./run_vllm_suite.sh $T ;;
  harness) harness_all $T ;;
  analyze) .venv-torch/bin/python analyze_h100.py Qwen3-8B ;;
  core) ./run_vllm_suite.sh $T; harness_all $T; .venv-torch/bin/python analyze_h100.py Qwen3-8B ;;
  followup)   # ~40 min after core: post-processing, AR repeat on divergent cases, divergence replay, paired timing
    .venv-torch/bin/python postprocess.py Qwen3-8B
    IDS=results/Qwen3-8B/diverge_ids.json ./run_vllm_suite.sh $T ar_repeat
    .venv-torch/bin/python diverge.py --target $T
    .venv-torch/bin/python paired_timing.py --target $T
    .venv-torch/bin/python postprocess.py Qwen3-8B
    ;;
  extra4b) ./run_vllm_suite.sh Qwen/Qwen3-4B; harness_all Qwen/Qwen3-4B; .venv-torch/bin/python analyze_h100.py Qwen3-4B ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
