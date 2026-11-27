#!/usr/bin/env bash
# Text-to-SQL suite (sqlbench.py): one vLLM server per method, 1,000 queries, batch one, greedy, prefix caching off.
# Usage: ./run_sql_suite.sh KEY [METHOD ...]   e.g. ./run_sql_suite.sh qwen3-8b ar dflash eagle7 copyflash63
#   LIMIT=20 ./run_sql_suite.sh qwen3-8b ar   (smoke)
set -uo pipefail
cd "$(dirname "$0")"
source .venv-vllm/bin/activate
KEY=$1; shift
MODEL=$(python -c "import json,sys; print(json.load(open('targets.json'))[sys.argv[1]]['model'])" "$KEY")
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
  mapfile -t NAMES < <(python -c "from common import vllm_configs; import sys; print('\n'.join(vllm_configs(sys.argv[1])))" "$KEY")
fi
PORT=8000; BASE=http://127.0.0.1:$PORT
mkdir -p logs
LIMIT_ARG=(); [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")
for NAME in "${NAMES[@]}"; do
  SPEC=$(python -c "from common import vllm_configs; import json,sys; c=vllm_configs(sys.argv[1], extra=True)[sys.argv[2]]; print('' if c is None else json.dumps(c))" "$KEY" "$NAME")
  EXTRA=(); [ -n "$SPEC" ] && EXTRA=(--speculative-config "$SPEC")
  LOG=logs/server_${KEY}_$NAME.log
  echo "=== $KEY $NAME: $SPEC"
  vllm serve "$MODEL" --port $PORT --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.88 --attention-backend flash_attn \
    --max-num-batched-tokens 32768 --seed 0 --generation-config vllm "${EXTRA[@]}" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for _ in $(seq 1 240); do
    if curl -sf $BASE/health >/dev/null; then ok=1; break; fi
    if ! kill -0 $PID 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! server for $KEY $NAME failed to start; see $LOG"; tail -20 "$LOG"; kill $PID 2>/dev/null; wait $PID 2>/dev/null; continue
  fi
  python sqlbench.py run --key "$KEY" --name "$NAME" --base-url $BASE "${LIMIT_ARG[@]}" 2>&1 \
    | tee -a logs/sql_${KEY}_$NAME.log | tail -1
  kill $PID; wait $PID 2>/dev/null
  sleep 5
done
python sqlbench.py score --key "$KEY"
