#!/usr/bin/env bash
# GPU session 2 (one H100 80 GB, ~10-12 h): close the gap to Qwen, and test "diffusion accelerates AR".
#
#   bash scripts/run_gpu_session2.sh                          # everything
#   TRACKS="rl spec" bash scripts/run_gpu_session2.sh         # a subset of the training tracks
#   SKIP_SMOKE=1 / BENCH_N=1000 / RL_STEPS=1000 / EPOCHS_MORE=10 ...   (see below)
#
# Three training tracks run in parallel (small models, one GPU), then all latency benchmarks run alone, in
# sequence, so no number is measured while something else is training. Every step skips work whose output exists,
# so re-running resumes.
#
#   epochs  Ettin-400M diffusion +EPOCHS_MORE epochs from ettin400-diffusion-iso-10ep, then 1 epoch distillation
#   rl      execution-reward RL (src/train_rl.py) on ettin400-diffusion-iso-10ep-distilled; again on the
#           longer-trained model when 'epochs' finishes (RL_AFTER_EPOCHS=1)
#   spec    Qwen3-0.6B-Base SFT (same recipe as Qwen2.5: 3 epochs); the verifier is whichever Qwen SFT has the
#           higher exec; on-policy data from it; EAGLE-3 and DFlash drafters (Speculators, offline hidden states)
#   bench   (after the tracks) quality + batch-1 latency: 400M diffusion (+verify 1) for every checkpoint, the
#           Qwen SFTs on vLLM plain / ngram / EAGLE-3 / DFlash
#
# Needs in checkpoints/: ettin400-diffusion-iso-10ep, ettin400-diffusion-iso-10ep-distilled,
# sql-ar-qwen25-coder-0.5b-ar (see docs/gpu-session-2.md). Setup reuses scripts/run_gpu_session.sh (JOBS=setup) and
# adds .venv-spec (Speculators + a vLLM with DFlash).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"

TRACKS="${TRACKS:-epochs rl spec}"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-2}"
BENCH_N="${BENCH_N:-1000}"
EPOCHS_MORE="${EPOCHS_MORE:-10}"
RL_STEPS="${RL_STEPS:-1000}"
RL_AFTER_EPOCHS="${RL_AFTER_EPOCHS:-1}"
SPEC_N="${SPEC_N:-30000}"           # on-policy rows for the drafters
DFLASH_LAYERS="${DFLASH_LAYERS:-3}" # draft depth; the 8B recipe uses 5, a 0.5B target wants a cheaper drafter
DFLASH_BLOCK="${DFLASH_BLOCK:-16}"
mkdir -p "$SESSION_DIR/logs" "$SESSION_DIR/bench" checkpoints
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
[ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ] && export WANDB_MODE=offline
PY=.venv-gpu/bin/python; PYV=.venv-vllm/bin/python; PYS=.venv-spec/bin/python
L="$SESSION_DIR/session.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
has() { [[ " $TRACKS " == *" $1 "* ]]; }
done_ckpt() { [ -f "checkpoints/$1/model.safetensors" ]; }

# train <name> <python> <script> [ENV=VAL ...]: 20-step smoke run, then the real run (skipped if it exists)
train() {
  local name=$1 py=$2 script=$3; shift 3
  done_ckpt "$name" && { log "skip $name (exists)"; return 0; }
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    (cd src && env "$@" MAX_TRAIN_STEPS=20 TRAIN_SIZE=2000 VAL_SIZE=32 GEN_EVAL_SIZE=4 EVAL_STEPS=20 \
      WANDB_MODE=disabled OUTPUT_DIR="$REPO_DIR/.smoke/$name" "$REPO_DIR/$py" "$script") \
      > "$SESSION_DIR/logs/smoke_$name.log" 2>&1 || { log "!! smoke $name failed"; return 1; }
    rm -rf ".smoke/$name"
  fi
  log "train $name"; local t0=$SECONDS
  (cd src && env "$@" RUN_NAME="$name" OUTPUT_DIR="$REPO_DIR/checkpoints/$name" "$REPO_DIR/$py" "$script") \
    > "$SESSION_DIR/logs/train_$name.log" 2>&1 || { log "!! train $name failed"; return 1; }
  log "done $name in $(( (SECONDS - t0) / 60 )) min"
}

rl() {  # rl <from> <name>
  local from=$1 name=$2
  done_ckpt "$name" && { log "skip $name (exists)"; return 0; }
  done_ckpt "$from" || { log "!! rl: $from missing"; return 1; }
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    (cd src && INIT_CHECKPOINT="$REPO_DIR/checkpoints/$from" OUTPUT_DIR="$REPO_DIR/.smoke/$name" STEPS=2 \
      PROMPTS_PER_STEP=2 EVAL_EVERY=2 EVAL_SIZE=4 "$REPO_DIR/$PY" train_rl.py) \
      > "$SESSION_DIR/logs/smoke_$name.log" 2>&1 || { log "!! smoke $name failed"; return 1; }
    rm -rf ".smoke/$name"
  fi
  log "rl $name from $from"; local t0=$SECONDS
  (cd src && INIT_CHECKPOINT="$REPO_DIR/checkpoints/$from" OUTPUT_DIR="$REPO_DIR/checkpoints/$name.tmp" \
    STEPS="$RL_STEPS" "$REPO_DIR/$PY" train_rl.py) > "$SESSION_DIR/logs/rl_$name.log" 2>&1 \
    && mv "checkpoints/$name.tmp" "checkpoints/$name" && log "done $name in $(( (SECONDS - t0) / 60 )) min" \
    || { log "!! rl $name failed"; return 1; }
}

# bench <python> <name> [bench.run args]: one benchmark run into $SESSION_DIR/bench (skipped if it exists)
bench() {
  local py=$1 name=$2; shift 2
  [ -f "$SESSION_DIR/bench/$name.summary.json" ] && return 0
  "$py" -m bench.run --device cuda --n "$BENCH_N" --name "$name" --out "$SESSION_DIR/bench" "$@" \
    >> "$SESSION_DIR/logs/bench.log" 2>&1 && log "bench $name" || log "!! bench $name failed"
}
exec_of() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['exec'])" "$1" 2>/dev/null || echo 0; }

# --------------------------------------------------------------------------------------------------- setup
if [ ! -x "$PY" ] || [ ! -x "$PYV" ]; then
  log "setup (scripts/run_gpu_session.sh JOBS=setup)"
  JOBS=setup SESSION_DIR="$SESSION_DIR" bash scripts/run_gpu_session.sh >> "$SESSION_DIR/logs/setup.log" 2>&1
fi
if has spec && [ ! -x "$PYS" ]; then
  log "setup .venv-spec (speculators + vLLM)"
  python3 -m venv .venv-spec && .venv-spec/bin/pip install -q --upgrade pip
  .venv-spec/bin/pip install -q speculators vllm "datasets>=4" >> "$SESSION_DIR/logs/setup.log" 2>&1 \
    || log "!! speculators install failed: spec track will be skipped"
  [ -d .speculators ] || git clone -q https://github.com/vllm-project/speculators.git .speculators  # launch_vllm.py
fi
for c in ettin400-diffusion-iso-10ep ettin400-diffusion-iso-10ep-distilled sql-ar-qwen25-coder-0.5b-ar; do
  done_ckpt "$c" || log "!! checkpoints/$c missing (upload it, docs/gpu-session-2.md)"
done

# --------------------------------------------------------------------------------------------------- tracks
track_epochs() {
  local base=ettin400-diffusion-iso-$((10 + EPOCHS_MORE))ep
  train "$base" "$PY" train.py INIT_CHECKPOINT="$REPO_DIR/checkpoints/ettin400-diffusion-iso-10ep" \
    PROMPT_ISOLATION=1 NUM_EPOCHS="$EPOCHS_MORE" LEARNING_RATE=3e-5 BATCH_SIZE=32 GRAD_ACCUM=4 || return 1
  train "$base-distilled" "$PY" train.py INIT_CHECKPOINT="$REPO_DIR/checkpoints/$base" PROMPT_ISOLATION=1 \
    TRAJECTORY_MASKING=1 CERTAINTY_WEIGHT=0.5 NUM_EPOCHS=1 LEARNING_RATE=3e-5 BATCH_SIZE=32 GRAD_ACCUM=4 || return 1
  if [ "$RL_AFTER_EPOCHS" = 1 ]; then rl "$base-distilled" "$base-distilled-rl"; fi
}
track_rl() { rl ettin400-diffusion-iso-10ep-distilled ettin400-diffusion-iso-10ep-distilled-rl; }

# Speculators on one GPU: dump the verifier's hidden states (offline mode), then train with vLLM stopped.
drafter() {  # drafter <verifier> <eagle3|dflash>
  local v=$1 kind=$2 dir="checkpoints/spec-$1"
  local out="$dir/$kind"
  [ -f "$out/done" ] && return 0
  local args=(--speculator-type "$kind")
  [ "$kind" = dflash ] && args+=(--num-layers "$DFLASH_LAYERS" --block-size "$DFLASH_BLOCK" --lr 3e-4)
  log "drafter $kind for $v"
  .venv-spec/bin/torchrun --standalone --nproc_per_node 1 -m speculators.train \
    --verifier-name-or-path "checkpoints/$v" --data-path "$dir/prepared" --hidden-states-path "$dir/hidden_states" \
    --on-missing raise --save-path "$out" --epochs 5 --total-seq-len 2048 --draft-vocab-size 16384 "${args[@]}" \
    > "$SESSION_DIR/logs/drafter_${v}_$kind.log" 2>&1 && touch "$out/done" && log "done drafter $kind" \
    || { log "!! drafter $kind failed (logs/drafter_${v}_$kind.log)"; return 1; }
}

track_spec() {
  train qwen3-0.6b-ar "$PY" train_ar.py MODEL_NAME=Qwen/Qwen3-0.6B-Base NUM_EPOCHS=3 \
    || log "!! qwen3 SFT failed: verifier stays Qwen2.5"
  # verifier = the better Qwen SFT on 1000 test queries (quality only; latency comes later, alone)
  local v=sql-ar-qwen25-coder-0.5b-ar
  if done_ckpt qwen3-0.6b-ar; then
    VLLM_GPU_UTIL=0.2 bench "$PYV" q-qwen25 --arm ar --engine vllm --model checkpoints/sql-ar-qwen25-coder-0.5b-ar
    VLLM_GPU_UTIL=0.2 bench "$PYV" q-qwen3 --arm ar --engine vllm --model checkpoints/qwen3-0.6b-ar
    local a b; a=$(exec_of "$SESSION_DIR/bench/q-qwen25.summary.json"); b=$(exec_of "$SESSION_DIR/bench/q-qwen3.summary.json")
    "$PY" -c "import sys;sys.exit(0 if float('$b')>float('$a') else 1)" && v=qwen3-0.6b-ar
    log "verifier: $v (exec qwen2.5 $a, qwen3 $b)"
  fi
  echo "$v" > "$SESSION_DIR/verifier"
  [ -x "$PYS" ] || { log "!! no .venv-spec, skipping drafters"; return 1; }
  local dir="checkpoints/spec-$v"; mkdir -p "$dir"
  [ -s "$dir/data.jsonl" ] || VLLM_GPU_UTIL=0.3 "$PYS" scripts/spec_data.py --model "checkpoints/$v" --n "$SPEC_N" \
    --out "$dir/data.jsonl" > "$SESSION_DIR/logs/spec_data.log" 2>&1 || { log "!! spec data failed"; return 1; }
  [ -d "$dir/prepared" ] || .venv-spec/bin/speculators prepare-data --model "checkpoints/$v" --data "$dir/data.jsonl" \
    --output "$dir/prepared" --seq-length 2048 >> "$SESSION_DIR/logs/spec_data.log" 2>&1 \
    || { log "!! prepare-data failed"; return 1; }
  if [ ! -f "$dir/hidden_states.done" ]; then
    log "hidden states for $v"
    # own process group, so stopping it also stops vLLM's engine processes (and frees their GPU memory)
    setsid "$PYS" .speculators/scripts/launch_vllm.py "checkpoints/$v" -- --port 8011 --gpu-memory-utilization 0.25 \
      > "$SESSION_DIR/logs/spec_vllm.log" 2>&1 &
    local vpid=$!
    until curl -sf http://localhost:8011/health > /dev/null 2>&1; do
      kill -0 $vpid 2>/dev/null || { log "!! hidden-state server died (logs/spec_vllm.log)"; return 1; }; sleep 5
    done
    .venv-spec/bin/speculators generate-offline-data --preprocessed-data "$dir/prepared" \
      --endpoint http://localhost:8011/v1 --output "$dir/hidden_states" --concurrency 64 \
      >> "$SESSION_DIR/logs/spec_data.log" 2>&1 && touch "$dir/hidden_states.done"
    kill -- -"$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
    [ -f "$dir/hidden_states.done" ] || { log "!! hidden states failed"; return 1; }
  fi
  drafter "$v" eagle3
  drafter "$v" dflash
}

pids=()
has epochs && { track_epochs & pids+=($!); }
has rl && { track_rl & pids+=($!); }
has spec && { track_spec & pids+=($!); }
for p in "${pids[@]}"; do wait "$p"; done
log "tracks finished"

# --------------------------------------------------------------------------------------------------- bench (alone)
for c in ettin400-diffusion-iso-10ep-distilled ettin400-diffusion-iso-10ep-distilled-rl \
         "ettin400-diffusion-iso-$((10 + EPOCHS_MORE))ep-distilled" \
         "ettin400-diffusion-iso-$((10 + EPOCHS_MORE))ep-distilled-rl"; do
  done_ckpt "$c" || continue
  bench "$PY" "$c-bf16-cudagraphs" --arm diffusion --window 64 --dtype bfloat16 --cuda-graphs --model "checkpoints/$c"
  bench "$PY" "$c-bf16-cudagraphs-verify1" --arm diffusion --window 64 --dtype bfloat16 --cuda-graphs --verify 1 \
    --model "checkpoints/$c"
done
v=$(cat "$SESSION_DIR/verifier" 2>/dev/null || echo sql-ar-qwen25-coder-0.5b-ar)
for q in sql-ar-qwen25-coder-0.5b-ar qwen3-0.6b-ar; do
  done_ckpt "$q" && bench "$PYV" "$q-vllm" --arm ar --engine vllm --model "checkpoints/$q"
done
VLLM_SPEC='{"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_max": 4, "prompt_lookup_min": 2}' \
  bench "$PYV" "$v-vllm-ngram" --arm ar --engine vllm --model "checkpoints/$v"
SPY="$PYS"; [ -x "$SPY" ] || SPY="$PYV"
for k in eagle3:4 dflash:$((DFLASH_BLOCK - 1)); do
  kind=${k%%:*}; n=${k##*:}; d="checkpoints/spec-$v/$kind"
  [ -f "$d/done" ] || continue
  # Speculators saves <save-path>/<epoch>/ and a checkpoint_best symlink
  ck="$d/checkpoint_best"; [ -e "$ck" ] || ck=$(ls -d "$d"/[0-9]* 2>/dev/null | sort -V | tail -1)
  VLLM_SPEC="{\"method\": \"$kind\", \"model\": \"$(realpath "$ck")\", \"num_speculative_tokens\": $n}" \
    bench "$SPY" "$v-vllm-$kind" --arm ar --engine vllm --model "checkpoints/$v"
done
"$PY" - "$SESSION_DIR/bench" <<'PY' | tee "$SESSION_DIR/summary.md"
import glob, json, os, sys
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.summary.json")))]
print("| run | exec | exact | valid | fwd | p50 ms | mean ms |\n|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['exec']:.3f} | {r['exact']:.3f} | {r['valid']:.3f} | {r['avg_forward']} "
          f"| {r['lat_p50_ms']:.1f} | {r['lat_mean_ms']:.1f} |")
PY
log "session 2: ALL DONE"
