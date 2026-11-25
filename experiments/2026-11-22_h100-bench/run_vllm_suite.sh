#!/usr/bin/env bash
# One vLLM server per method, batch one, greedy, prefix caching off. Results: results/<target>/vllm_<name>.jsonl
# Usage: ./run_vllm_suite.sh [TARGET] [NAME ...]     e.g. ./run_vllm_suite.sh Qwen/Qwen3-8B ar dflash15
# LIMIT=20 ./run_vllm_suite.sh ...   runs only the first 20 edits (smoke test).
set -uo pipefail
cd "$(dirname "$0")"
source .venv-vllm/bin/activate
TARGET=${1:-Qwen/Qwen3-8B}; shift || true
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
  mapfile -t NAMES < <(python -c "import json,sys; print('\n'.join(json.load(open('configs.json'))[sys.argv[1]]))" "$TARGET")
fi
PORT=8000; BASE=http://127.0.0.1:$PORT
mkdir -p logs
LIMIT_ARG=(); [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")
[ -n "${IDS:-}" ] && LIMIT_ARG+=(--ids "$IDS")

for NAME in "${NAMES[@]}"; do
  SPEC=$(python -c "import json,sys; c=json.load(open('configs.json'))[sys.argv[1]][sys.argv[2]]; print('' if c is None else json.dumps(c))" "$TARGET" "$NAME")
  EXTRA=(); [ -n "$SPEC" ] && EXTRA=(--speculative-config "$SPEC")
  LOG=logs/server_$(basename "$TARGET")_$NAME.log
  echo "=== $NAME: $SPEC"
  vllm serve "$TARGET" --port $PORT --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.85 --attention-backend flash_attn \
    --max-num-batched-tokens 32768 --seed 0 --generation-config vllm "${EXTRA[@]}" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for _ in $(seq 1 180); do
    if curl -sf $BASE/health >/dev/null; then ok=1; break; fi
    if ! kill -0 $PID 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! server for $NAME failed to start; see $LOG"; tail -20 "$LOG"; kill $PID 2>/dev/null; wait $PID 2>/dev/null; continue
  fi
  python bench_vllm.py --name "$NAME" --target "$TARGET" --base-url $BASE "${LIMIT_ARG[@]}" 2>&1 | tee logs/bench_$(basename "$TARGET")_$NAME.log | tail -3
  kill $PID; wait $PID 2>/dev/null
  sleep 5
done
