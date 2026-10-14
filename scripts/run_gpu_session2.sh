#!/usr/bin/env bash
# GPU session 2 (one H100 80 GB, ~10-12 h): close the gap to Qwen, and test "diffusion accelerates AR".
#
#   bash scripts/run_gpu_session2.sh                          # everything
#   TRACKS="rl spec" bash scripts/run_gpu_session2.sh         # a subset of the training tracks
#   SKIP_SMOKE=1 / BENCH_N=1000 / RL_STEPS=1000 / EPOCHS_MORE=10 ...   (see below)
#
# Order (every step skips work whose output exists, so re-running resumes):
#   setup     venvs (uv, in parallel) + model/data prefetch, while you upload the checkpoints
#   vllmpr    vllm-project/vllm#61276 on the Ettin decoders (pytest + scripts/vllm_ettin_check.py), during the upload
#   baseline  batch-1 speed of the current 400M diffusion (+verify 1) and Qwen2.5 on vLLM (plain, ngram), GPU alone
#   tracks    in parallel (small models, one GPU):
#     epochs  Ettin-400M diffusion +EPOCHS_MORE epochs from ettin400-diffusion-iso-10ep, then 1 epoch distillation
#     rl      execution-reward RL (src/train_rl.py) on ettin400-diffusion-iso-10ep-distilled; again on the
#             longer-trained model when 'epochs' finishes (RL_AFTER_EPOCHS=1)
#     spec    Qwen3-0.6B-Base SFT (same recipe as Qwen2.5: 3 epochs); the verifier is whichever Qwen SFT has the
#             higher exec; on-policy data from it; EAGLE-3 and DFlash drafters (Speculators, offline hidden states)
#   bench     GPU alone again: every new checkpoint, the Qwen SFTs on vLLM plain / ngram / EAGLE-3 / DFlash
#
# Needs in checkpoints/: ettin400-diffusion-iso-10ep, ettin400-diffusion-iso-10ep-distilled,
# sql-ar-qwen25-coder-0.5b-ar (docs/gpu-session-2.md); waits for them before 'baseline'.
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
PRE="${PRE:-vllmpr baseline}"      # before the tracks: vLLM PR checks (during the upload), baseline speed (GPU alone)
VLLM_PR="${VLLM_PR:-61276}"
ETTIN_DECODERS="${ETTIN_DECODERS:-17m 150m 400m 1b}"
NGRAM='{"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_max": 4, "prompt_lookup_min": 2}'
mkdir -p "$SESSION_DIR/logs" "$SESSION_DIR/bench" checkpoints
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
[ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ] && export WANDB_MODE=offline
PY=.venv-gpu/bin/python; PYV=.venv-vllm/bin/python; PYS=.venv-spec/bin/python
L="$SESSION_DIR/session.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
has() { [[ " $TRACKS " == *" $1 "* ]]; }
has_pre() { [[ " $PRE " == *" $1 "* ]]; }
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
summary() {  # summary.md from every bench run so far
  "$PY" - "$SESSION_DIR/bench" <<'PY' | tee "$SESSION_DIR/summary.md"
import glob, json, os, sys
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.summary.json")))]
print("| run | exec | exact | valid | fwd | p50 ms | mean ms |\n|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['exec']:.3f} | {r['exact']:.3f} | {r['valid']:.3f} | {r['avg_forward']} "
          f"| {r['lat_p50_ms']:.1f} | {r['lat_mean_ms']:.1f} |")
PY
}
exec_of() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['exec'])" "$1" 2>/dev/null || echo 0; }

# --------------------------------------------------------------------------------------------------- setup
# Everything that needs no GPU runs at once: venvs (uv, shared download cache), model/data prefetch, and the
# checkpoint upload from your machine. The GPU starts on the vLLM PR checks as soon as their venv exists.
command -v uv > /dev/null || { python3 -m pip install -q uv 2> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
log "=== $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) ($(uname -m))"
mkvenv() {  # mkvenv <dir> <install command run with $V as the venv's python>
  local d=$1; shift
  [ -f "$d/.ok" ] && return 0
  local t0=$SECONDS
  { [ -x "$d/bin/python" ] || uv venv -q --seed -p 3.12 "$d"; } && V="$d/bin/python" bash -c "$*" \
    >> "$SESSION_DIR/logs/setup${d}.log" 2>&1 && touch "$d/.ok" && log "venv $d ready ($(( SECONDS - t0 ))s)" \
    || log "!! $d failed (logs/setup${d}.log)"
}
mkvenv .venv-gpu 'uv pip install -q -p $V --torch-backend auto -r requirements.txt -r bench/requirements.txt' &
mkvenv .venv-vllm 'uv pip install -q -p $V vllm "datasets>=4"' &
has spec && { mkvenv .venv-spec 'uv pip install -q -p $V speculators vllm "datasets>=4"' &
  [ -d .speculators ] || git clone -q https://github.com/vllm-project/speculators.git .speculators; }  # launch_vllm.py
if has_pre vllmpr; then  # vllm-project/vllm#61276 (python only -> precompiled kernels, no build)
  { { [ -d .vllm-pr ] || git clone -q --filter=blob:none https://github.com/vllm-project/vllm.git .vllm-pr; } &&
    git -C .vllm-pr fetch -q origin "pull/$VLLM_PR/head:pr$VLLM_PR" && git -C .vllm-pr checkout -q "pr$VLLM_PR" &&
    mkvenv .venv-vllmpr 'cd .vllm-pr && VLLM_USE_PRECOMPILED=1 uv pip install -q -p ../$V -e . &&
      uv pip install -q -p ../$V pytest pytest-asyncio tblib pillow accelerate'; } &
fi
prefetch() {  # the downloads training would otherwise wait for
  local hf="uvx -q --from huggingface_hub[hf_xet] hf download"
  $hf --repo-type dataset gretelai/synthetic_text_to_sql
  has spec && $hf Qwen/Qwen3-0.6B-Base
  has_pre vllmpr && for m in $ETTIN_DECODERS; do $hf "jhu-clsp/ettin-decoder-$m"; done
} > "$SESSION_DIR/logs/prefetch.log" 2>&1
prefetch &
wait
for d in .venv-gpu .venv-vllm; do [ -f "$d/.ok" ] || { log "!! $d missing, stopping"; exit 1; }; done
# FlashInfer (vLLM's sampler) JIT-compiles kernels on first use and needs ninja on PATH, not just in the venv
mkdir -p "$HOME/.local/bin" && ln -sf "$REPO_DIR/.venv-vllm/bin/ninja" "$HOME/.local/bin/ninja"
"$PY" -c "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$L"

# --------------------------------------------------------------------------------------------------- vLLM PR (GPU, ~15 min)
# While the checkpoints upload: does #61276 load the Ettin decoders and match Transformers beyond the 17M test model?
# Their tests (pytest), then scripts/vllm_ettin_check.py per size and dtype, both model_impl=transformers and auto.
if has_pre vllmpr && [ -f .venv-vllmpr/.ok ] && [ ! -f "$SESSION_DIR/vllm-pr/done" ]; then
  mkdir -p "$SESSION_DIR/vllm-pr"; log "vLLM #$VLLM_PR checks"
  (cd .vllm-pr && timeout 1800 ../.venv-vllmpr/bin/python -m pytest -q tests/models/transformers/test_backend.py \
    -k "output_head or remote_code") > "$SESSION_DIR/vllm-pr/pytest.log" 2>&1
  log "vLLM #$VLLM_PR pytest: $(tail -1 "$SESSION_DIR/vllm-pr/pytest.log")"
  for m in $ETTIN_DECODERS; do for dt in float32 bfloat16; do for impl in transformers auto; do
    [ "$impl" = auto ] && [ "$dt" = float32 ] && continue
    timeout 900 .venv-vllmpr/bin/python scripts/vllm_ettin_check.py --model "jhu-clsp/ettin-decoder-$m" --dtype "$dt" \
      --impl "$impl" --out "$SESSION_DIR/vllm-pr/ettin.jsonl" >> "$SESSION_DIR/vllm-pr/check.log" 2>&1 \
      || log "!! ettin-decoder-$m $dt $impl failed (vllm-pr/check.log)"
  done; done; done
  touch "$SESSION_DIR/vllm-pr/done"; log "vLLM #$VLLM_PR checks done"
fi

# --------------------------------------------------------------------------------------------------- baseline speed (GPU alone)
# The numbers every later result is compared against, measured on this box before anything trains.
if ! { [ -f checkpoints/.uploaded ] || [ -f checkpoints/sql-ar-qwen25-coder-0.5b-ar/vocab.json ]; }; then
  log "waiting for the checkpoint upload (touch checkpoints/.uploaded when rsync is done)"
  until [ -f checkpoints/.uploaded ]; do sleep 15; done
fi
for c in ettin400-diffusion-iso-10ep ettin400-diffusion-iso-10ep-distilled sql-ar-qwen25-coder-0.5b-ar; do
  done_ckpt "$c" || { log "!! checkpoints/$c missing (docs/gpu-session-2.md), stopping"; exit 1; }
done
if has_pre baseline; then
  log "baseline speed"
  c=ettin400-diffusion-iso-10ep-distilled
  bench "$PY" "$c-bf16-cudagraphs" --arm diffusion --window 64 --dtype bfloat16 --cuda-graphs --model "checkpoints/$c"
  bench "$PY" "$c-bf16-cudagraphs-verify1" --arm diffusion --window 64 --dtype bfloat16 --cuda-graphs --verify 1 \
    --model "checkpoints/$c"
  bench "$PYV" sql-ar-qwen25-coder-0.5b-ar-vllm --arm ar --engine vllm --model checkpoints/sql-ar-qwen25-coder-0.5b-ar
  VLLM_SPEC="$NGRAM" bench "$PYV" sql-ar-qwen25-coder-0.5b-ar-vllm-ngram --arm ar --engine vllm \
    --model checkpoints/sql-ar-qwen25-coder-0.5b-ar
  summary > /dev/null; log "baseline done: $SESSION_DIR/summary.md"
fi

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
VLLM_SPEC="$NGRAM" bench "$PYV" "$v-vllm-ngram" --arm ar --engine vllm --model "checkpoints/$v"
SPY="$PYS"; [ -x "$SPY" ] || SPY="$PYV"
for k in eagle3:4 dflash:$((DFLASH_BLOCK - 1)); do
  kind=${k%%:*}; n=${k##*:}; d="checkpoints/spec-$v/$kind"
  [ -f "$d/done" ] || continue
  # Speculators saves <save-path>/<epoch>/ and a checkpoint_best symlink
  ck="$d/checkpoint_best"; [ -e "$ck" ] || ck=$(ls -d "$d"/[0-9]* 2>/dev/null | sort -V | tail -1)
  VLLM_SPEC="{\"method\": \"$kind\", \"model\": \"$(realpath "$ck")\", \"num_speculative_tokens\": $n}" \
    bench "$SPY" "$v-vllm-$kind" --arm ar --engine vllm --model "checkpoints/$v"
done
summary
log "session 2: ALL DONE"; touch "$SESSION_DIR/ALL_DONE"
