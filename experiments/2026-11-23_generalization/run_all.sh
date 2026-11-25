#!/usr/bin/env bash
# Stages (each resumable; finished requests are skipped):
#   ./run_all.sh smoke            ~30 min  every target: ar + dflash on 3 requests per cohort, 2 harness requests per cohort
#   ./run_all.sh target KEY       one target: full vLLM suite on all cohorts, then harness dflash + router8
#   ./run_all.sh core             qwen3-8b, qwen3-4b, llama31-8b (unless SKIP_LLAMA=1), qwen3coder-30b, then summarize
#   ./run_all.sh concurrency      ~40 min  qwen3-8b: ar / ngram31 / dflash / copyflash63 at 1, 4, 16 concurrent clients
#   ./run_all.sh summarize        results/summary.md (+ pre-registered predictions check)
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-torch/bin/python
KEYS=(qwen3-8b qwen3-4b llama31-8b qwen3coder-30b)
[ "${SKIP_LLAMA:-0}" = 1 ] && KEYS=(qwen3-8b qwen3-4b qwen3coder-30b)
COHORT_LIST=(edit261 ceb150 diff104)

harness_target() {  # KEY [LIMIT]
  local L=(); [ -n "${2:-}" ] && L=(--limit "$2")
  for c in "${COHORT_LIST[@]}"; do
    $PY harness.py dflash --key "$1" --cohort "$c" "${L[@]}"
    $PY harness.py hybrid_router --router-min 8 --key "$1" --cohort "$c" "${L[@]}"
  done
}

case "${1:-}" in
  smoke)
    for k in "${KEYS[@]}"; do LIMIT=3 ./run_vllm_suite.sh "$k" ar dflash; harness_target "$k" 2; done
    mkdir -p results/smoke && for k in "${KEYS[@]}"; do [ -d results/$k ] && mv results/$k results/smoke/; done
    echo "smoke OK if every target printed tau lines for ar and dflash in all 3 cohorts and harness lines for each cohort"
    ;;
  target) ./run_vllm_suite.sh "$2"; harness_target "$2" ;;
  core)
    for k in "${KEYS[@]}"; do ./run_vllm_suite.sh "$k"; harness_target "$k"; $PY summarize.py > /dev/null; done
    $PY summarize.py
    ;;
  concurrency)
    mkdir -p logs; source .venv-vllm/bin/activate
    M=$(python -c "import json; print(json.load(open('targets.json'))['qwen3-8b']['model'])")
    for NAME in ar ngram31 dflash copyflash63; do
      SPEC=$(python -c "from common import vllm_configs; import json,sys; c=vllm_configs('qwen3-8b', extra=True)[sys.argv[1]]; print('' if c is None else json.dumps(c))" $NAME)
      EXTRA=(); [ -n "$SPEC" ] && EXTRA=(--speculative-config "$SPEC")
      vllm serve "$M" --port 8000 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 --no-enable-prefix-caching \
        --gpu-memory-utilization 0.88 --attention-backend flash_attn --max-num-batched-tokens 32768 --seed 0 \
        --generation-config vllm "${EXTRA[@]}" > logs/server_concurrency_$NAME.log 2>&1 &
      PID=$!
      for _ in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
      python bench_concurrency.py --key qwen3-8b --name $NAME --levels 1 4 16
      kill $PID; wait $PID 2>/dev/null; sleep 5
    done
    ;;
  summarize) $PY summarize.py ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
