#!/usr/bin/env bash
# GPU session 3 (one H100 80 GB, ~3-3.5 h): get more out of the diffusion drafter, and a 1.5B verifier.
#
#   bash scripts/run_gpu_session3.sh                  # everything
#   STEPS="sweep" bash scripts/run_gpu_session3.sh    # a subset (setup always runs; every step resumes)
#
# Steps, in order (GPU alone unless noted):
#   sweep   acceptance + draft length for the session-2 drafters on Qwen2.5-0.5B: DFlash k in DFLASH_KS,
#           EAGLE-3 k in EAGLE_KS, SWEEP_N queries each (latency only picks k; quality is identical by design)
#   train   in parallel (GPU shared):
#             big   VAGUE_N vague rewrites of train questions (scripts/vague_prompts.py, VAGUE_MODEL), then drafters
#                   for Qwen2.5-0.5B on all train prompts (SPEC_N_BIG) + the vague ones, EAGLE-3 + DFlash
#             q15   Qwen2.5-Coder-1.5B SFT (QWEN15_EPOCHS), then its EAGLE-3 + DFlash drafters (+ the vague prompts)
#   bench   n=BENCH_N, every model at its best k: Qwen2.5-0.5B (session-2 and big drafters), Qwen2.5-1.5B plain /
#           EAGLE-3 / DFlash (DFlash k re-swept for 1.5B)
#
# Needs in checkpoints/: sql-ar-qwen25-coder-0.5b-ar and spec-qwen25/{eagle3,dflash} (session-2 drafters), see
# docs/gpu-session-3.md. Waits for checkpoints/.uploaded. W&B: export WANDB_API_KEY=... in ~/.wandb_env (chmod 600).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"

STEPS="${STEPS:-sweep train bench}"
SESSION_DIR="${SESSION_DIR:-results/gpu-session-3}"
BENCH_N="${BENCH_N:-1000}"
SWEEP_N="${SWEEP_N:-300}"
DFLASH_KS="${DFLASH_KS:-3 5 7 11 15}"
EAGLE_KS="${EAGLE_KS:-2 4 6 8}"
SPEC_N_BIG="${SPEC_N_BIG:-99800}"   # all train prompts except the last 200 (RL's eval rows)
SPEC_N_15="${SPEC_N_15:-50000}"  # the 1.5B drafters start late (after the SFT), so less data keeps them short
DRAFT_EPOCHS="${DRAFT_EPOCHS:-3}"
VAGUE_N="${VAGUE_N:-40000}"  # 0 = drafters on train prompts only
VAGUE_MODEL="${VAGUE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VAGUE=checkpoints/vague_prompts.jsonl
QWEN15_EPOCHS="${QWEN15_EPOCHS:-1}" # 0.5B had 3; session 1: AR barely gains from epochs, so 1 keeps this ~1.5 h
DFLASH_LAYERS="${DFLASH_LAYERS:-3}"
DFLASH_BLOCK="${DFLASH_BLOCK:-16}"
Q05=sql-ar-qwen25-coder-0.5b-ar
Q15=sql-ar-qwen25-coder-1.5b-ar
mkdir -p "$SESSION_DIR/logs" "$SESSION_DIR/bench" checkpoints
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}" TOKENIZERS_PARALLELISM=false DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$HOME/.wandb_env" ] && . "$HOME/.wandb_env"
export WANDB_PROJECT="${WANDB_PROJECT:-sql-diffusion}"
[ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ] && export WANDB_MODE=offline
PY=.venv-gpu/bin/python; PYV=.venv-vllm/bin/python; PYS=.venv-spec/bin/python
L="$SESSION_DIR/session.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
done_ckpt() { [ -f "checkpoints/$1/model.safetensors" ]; }

# bench <python> <name> [bench.run args]  (VLLM_SPEC / BENCH_N from the environment; skipped if it exists)
bench() {
  local py=$1 name=$2; shift 2
  [ -f "$SESSION_DIR/bench/$name.summary.json" ] && return 0
  "$py" -m bench.run --device cuda --n "${N:-$BENCH_N}" --name "$name" --out "$SESSION_DIR/bench" "$@" \
    >> "$SESSION_DIR/logs/bench.log" 2>&1 && log "bench $name" || log "!! bench $name failed"
}
field() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))$2)" "$1" 2>/dev/null; }
spec_json() {  # spec_json <eagle3|dflash> <drafter dir> <k>
  echo "{\"method\": \"$1\", \"model\": \"$(realpath "$2")\", \"num_speculative_tokens\": $3}"
}
drafter_dir() {  # Speculators saves <save>/<epoch>/ plus checkpoint_best; the session-2 copies are flat
  local d=$1
  if [ -e "$d/checkpoint_best" ]; then realpath "$d/checkpoint_best"
  elif [ -f "$d/config.json" ]; then realpath "$d"
  else ls -d "$d"/[0-9]* 2>/dev/null | sort -V | tail -1; fi
}
# sweep_k <verifier> <kind> <drafter dir> <tag> <ks...>: SWEEP_N-query latency per k; echoes the fastest k
sweep_k() {
  local v=$1 kind=$2 d=$3 tag=$4; shift 4
  local best="" best_ms=1e9 k ms
  for k in "$@"; do
    N=$SWEEP_N VLLM_SPEC="$(spec_json "$kind" "$d" "$k")" bench "$PYV" "sweep-$v-$tag-k$k" --arm ar --engine vllm \
      --model "checkpoints/$v" > /dev/null
    ms=$(field "$SESSION_DIR/bench/sweep-$v-$tag-k$k.summary.json" "['lat_mean_ms']")
    [ -n "$ms" ] && "$PY" -c "import sys;sys.exit(0 if $ms < $best_ms else 1)" && { best=$k; best_ms=$ms; }
  done
  echo "$best"
}

# --------------------------------------------------------------------------------------------------- setup
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
mkvenv .venv-spec 'uv pip install -q -p $V speculators vllm wandb "datasets>=4"' &
[ -d .speculators ] || git clone -q https://github.com/vllm-project/speculators.git .speculators &  # launch_vllm.py
{ uvx -q --from 'huggingface_hub[hf_xet]' hf download --repo-type dataset gretelai/synthetic_text_to_sql
  uvx -q --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen2.5-Coder-1.5B
  [ "$VAGUE_N" -gt 0 ] && uvx -q --from 'huggingface_hub[hf_xet]' hf download "$VAGUE_MODEL"; } > "$SESSION_DIR/logs/prefetch.log" 2>&1 &
wait
for d in .venv-gpu .venv-vllm .venv-spec; do [ -f "$d/.ok" ] || { log "!! $d missing, stopping"; exit 1; }; done
# FlashInfer (vLLM's sampler) JIT-compiles kernels on first use and needs ninja on PATH, not just in the venv
mkdir -p "$HOME/.local/bin" && ln -sf "$REPO_DIR/.venv-vllm/bin/ninja" "$HOME/.local/bin/ninja"
# Speculators 0.8.1: DFlash passes head_dim=None for verifiers without it in their config (Qwen2/2.5)
sed -i 's/^        head_dim=head_dim,$/        head_dim=resolved_head_dim,/' \
  .venv-spec/lib/python3.12/site-packages/speculators/train/cli.py
if [ ! -f checkpoints/.uploaded ]; then
  log "waiting for the checkpoint upload (touch checkpoints/.uploaded when rsync is done)"
  until [ -f checkpoints/.uploaded ]; do sleep 15; done
fi
for c in "$Q05" spec-qwen25/eagle3 spec-qwen25/dflash; do
  done_ckpt "$c" || { log "!! checkpoints/$c missing (docs/gpu-session-3.md), stopping"; exit 1; }
done

# --------------------------------------------------------------------------------------------------- sweep
K="$SESSION_DIR/best_k"; touch "$K"
best_k() { grep "^$1 " "$K" | tail -1 | cut -d' ' -f2; }
if has sweep; then
  log "sweep (n=$SWEEP_N): draft length for the session-2 drafters"
  N=$SWEEP_N bench "$PYV" "sweep-$Q05-plain" --arm ar --engine vllm --model "checkpoints/$Q05"
  [ -n "$(best_k s2-dflash)" ] || echo "s2-dflash $(sweep_k "$Q05" dflash checkpoints/spec-qwen25/dflash s2-dflash $DFLASH_KS)" >> "$K"
  [ -n "$(best_k s2-eagle3)" ] || echo "s2-eagle3 $(sweep_k "$Q05" eagle3 checkpoints/spec-qwen25/eagle3 s2-eagle3 $EAGLE_KS)" >> "$K"
  log "best k: $(tr '\n' ' ' < "$K")"
fi

# --------------------------------------------------------------------------------------------------- train
train() {  # train <name> <python> <script> [ENV=VAL ...]: 20-step smoke run, then the real run
  local name=$1 py=$2 script=$3; shift 3
  done_ckpt "$name" && { log "skip $name (exists)"; return 0; }
  (cd src && env "$@" MAX_TRAIN_STEPS=20 TRAIN_SIZE=2000 VAL_SIZE=32 GEN_EVAL_SIZE=4 EVAL_STEPS=20 \
    WANDB_MODE=disabled OUTPUT_DIR="$REPO_DIR/.smoke/$name" "$REPO_DIR/$py" "$script") \
    > "$SESSION_DIR/logs/smoke_$name.log" 2>&1 || { log "!! smoke $name failed"; return 1; }
  rm -rf ".smoke/$name"
  log "train $name"; local t0=$SECONDS
  (cd src && env "$@" RUN_NAME="$name" OUTPUT_DIR="$REPO_DIR/checkpoints/$name" "$REPO_DIR/$py" "$script") \
    > "$SESSION_DIR/logs/train_$name.log" 2>&1 || { log "!! train $name failed"; return 1; }
  log "done $name in $(( (SECONDS - t0) / 60 )) min"
}

# drafters <verifier> <n prompts> <tag> <port>: on-policy data, hidden states (offline), EAGLE-3 + DFlash
drafters() {
  local v=$1 n=$2 tag=$3 port=$4 dir="checkpoints/spec-$1-$3"
  mkdir -p "$dir"
  local extra=(); [ -s "$VAGUE" ] && extra=(--extra "$VAGUE")
  [ -s "$dir/data.jsonl" ] || VLLM_GPU_UTIL=0.25 "$PYS" scripts/spec_data.py --model "checkpoints/$v" --n "$n" \
    "${extra[@]}" --out "$dir/data.jsonl" > "$SESSION_DIR/logs/spec_data_$tag.log" 2>&1 || { log "!! spec data $tag failed"; return 1; }
  [ -d "$dir/prepared" ] || .venv-spec/bin/speculators prepare-data --model "checkpoints/$v" --data "$dir/data.jsonl" \
    --output "$dir/prepared" --seq-length 2048 >> "$SESSION_DIR/logs/spec_data_$tag.log" 2>&1 \
    || { log "!! prepare-data $tag failed"; return 1; }
  if [ ! -f "$dir/hidden_states.done" ]; then
    log "hidden states $tag"
    # own process group, so stopping it stops vLLM's engine too; vLLM's hidden-states KV connector refuses
    # expandable_segments
    env -u PYTORCH_CUDA_ALLOC_CONF setsid "$PYS" .speculators/scripts/launch_vllm.py "checkpoints/$v" -- \
      --port "$port" --gpu-memory-utilization 0.2 > "$SESSION_DIR/logs/spec_vllm_$tag.log" 2>&1 &
    local vpid=$!
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
      kill -0 $vpid 2>/dev/null || { log "!! hidden-state server $tag died"; return 1; }; sleep 5
    done
    .venv-spec/bin/speculators generate-offline-data --preprocessed-data "$dir/prepared" \
      --endpoint "http://localhost:$port/v1" --output "$dir/hidden_states" --concurrency 64 \
      >> "$SESSION_DIR/logs/spec_data_$tag.log" 2>&1 && touch "$dir/hidden_states.done"
    kill -- -"$vpid" 2>/dev/null; wait "$vpid" 2>/dev/null
    [ -f "$dir/hidden_states.done" ] || { log "!! hidden states $tag failed"; return 1; }
  fi
  local kind
  for kind in eagle3 dflash; do
    [ -f "$dir/$kind/done" ] && continue
    local args=(--speculator-type "$kind")
    [ "$kind" = dflash ] && args+=(--num-layers "$DFLASH_LAYERS" --block-size "$DFLASH_BLOCK" --lr 3e-4)
    [ -n "${WANDB_API_KEY:-}" ] && args+=(--logger wandb --run-name "spec-$kind-$v-$tag")
    log "drafter $kind $tag"
    .venv-spec/bin/torchrun --standalone --nproc_per_node 1 --master-port $((port + 100)) -m speculators.train \
      --verifier-name-or-path "checkpoints/$v" --data-path "$dir/prepared" --hidden-states-path "$dir/hidden_states" \
      --on-missing raise --save-path "$dir/$kind" --epochs "$DRAFT_EPOCHS" --total-seq-len 2048 \
      --draft-vocab-size 16384 "${args[@]}" > "$SESSION_DIR/logs/drafter_${tag}_$kind.log" 2>&1 \
      && touch "$dir/$kind/done" && log "done drafter $kind $tag" || log "!! drafter $kind $tag failed"
  done
  rm -rf "$dir/hidden_states"  # ~100+ GB; the drafters are what we keep
}

if has train; then
  pids=()
  { if [ "$VAGUE_N" -gt 0 ] && [ ! -s "$VAGUE" ]; then
      log "vague prompts ($VAGUE_N, $VAGUE_MODEL)"
      VLLM_GPU_UTIL=0.3 "$PYS" scripts/vague_prompts.py --model "$VAGUE_MODEL" --n "$VAGUE_N" --out "$VAGUE.tmp" \
        > "$SESSION_DIR/logs/vague_prompts.log" 2>&1 && mv "$VAGUE.tmp" "$VAGUE" && log "$(wc -l < "$VAGUE") vague prompts" \
        || log "!! vague prompts failed, drafters use train prompts only"
    fi
    drafters "$Q05" "$SPEC_N_BIG" big 8011; } & pids+=($!)
  { train "$Q15" "$PY" train_ar.py MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B NUM_EPOCHS="$QWEN15_EPOCHS" \
      && drafters "$Q15" "$SPEC_N_15" s3 8012; } & pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
  log "training done"
fi

# --------------------------------------------------------------------------------------------------- bench (alone)
if has bench; then
  bench "$PYV" "$Q05-vllm" --arm ar --engine vllm --model "checkpoints/$Q05"
  for kind in eagle3 dflash; do  # session-2 drafters at their best k, then the big ones (k re-swept)
    k=$(best_k "s2-$kind"); [ -n "$k" ] && VLLM_SPEC="$(spec_json $kind checkpoints/spec-qwen25/$kind "$k")" \
      bench "$PYV" "$Q05-vllm-$kind-s2-k$k" --arm ar --engine vllm --model "checkpoints/$Q05"
    d="checkpoints/spec-$Q05-big/$kind"; [ -f "$d/done" ] || continue
    ks="$EAGLE_KS"; [ $kind = dflash ] && ks="$DFLASH_KS"
    [ -n "$(best_k "big-$kind")" ] || echo "big-$kind $(sweep_k "$Q05" $kind "$(drafter_dir "$d")" "big-$kind" $ks)" >> "$K"
    k=$(best_k "big-$kind"); VLLM_SPEC="$(spec_json $kind "$(drafter_dir "$d")" "$k")" \
      bench "$PYV" "$Q05-vllm-$kind-big-k$k" --arm ar --engine vllm --model "checkpoints/$Q05"
  done
  if done_ckpt "$Q15"; then
    bench "$PYV" "$Q15-vllm" --arm ar --engine vllm --model "checkpoints/$Q15"
    for kind in eagle3 dflash; do
      d="checkpoints/spec-$Q15-s3/$kind"; [ -f "$d/done" ] || continue
      ks="$EAGLE_KS"; [ $kind = dflash ] && ks="$DFLASH_KS"
      [ -n "$(best_k "q15-$kind")" ] || echo "q15-$kind $(sweep_k "$Q15" $kind "$(drafter_dir "$d")" "q15-$kind" $ks)" >> "$K"
      k=$(best_k "q15-$kind"); VLLM_SPEC="$(spec_json $kind "$(drafter_dir "$d")" "$k")" \
        bench "$PYV" "$Q15-vllm-$kind-k$k" --arm ar --engine vllm --model "checkpoints/$Q15"
    done
  fi
  "$PY" - "$SESSION_DIR/bench" <<'PY' | tee "$SESSION_DIR/summary.md"
import glob, json, os, sys
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.summary.json")))]
print("| run | n | exec | exact | valid | p50 ms | mean ms | accepted/draft | accept rate |\n|---|---|---|---|---|---|---|---|---|")
for r in rows:
    s = r.get("spec") or {}
    print(f"| {r['name']} | {r['n']} | {r['exec']:.3f} | {r['exact']:.3f} | {r['valid']:.3f} | {r['lat_p50_ms']:.1f} "
          f"| {r['lat_mean_ms']:.1f} | {s.get('mean_accepted', '')} | {s.get('acceptance_rate', '')} |")
PY
fi
log "session 3: ALL DONE"; touch "$SESSION_DIR/ALL_DONE"
