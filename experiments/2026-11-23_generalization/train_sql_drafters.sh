#!/usr/bin/env bash
# Text-to-SQL drafters for Qwen3-8B, then the benchmark (README: "Text-to-SQL on Qwen3-8B"). One H100, ~2.5 h.
#   ONLY_KEYS="qwen3-8b" ./setup.sh && uv pip install --python .venv-vllm/bin/python -e ./copyflash_plugin
#   python sqlbench.py prepare   (or ship data/sql1000.json)
#   ./train_sql_drafters.sh
# Needs a second venv with Speculators (made below). Resumable: every stage is skipped once its output exists.
set -euxo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
D=spec-sql8b; S=.venv-spec/bin; mkdir -p $D logs
[ -x $S/python ] || { uv venv -q .venv-spec --python 3.12; uv pip install -q --python $S/python speculators vllm "datasets>=4" --torch-backend=auto; }
[ -d .speculators ] || git clone -q https://github.com/vllm-project/speculators.git .speculators   # scripts/launch_vllm.py

# 1. on-policy data: Qwen3-8B answers 12k TRAIN rows in the served prompt format (loss on its own output only)
[ -s $D/data.jsonl ] || VLLM_GPU_UTIL=0.85 $S/python spec_sql_data.py --model Qwen/Qwen3-8B --n 12000 --out $D/data.jsonl
[ -d $D/prepared ] || $S/speculators prepare-data --model Qwen/Qwen3-8B --data $D/data.jsonl --output $D/prepared --seq-length 2048

# 2. z-lab drafter in Speculators format. NOTE: the converter writes "sliding_window_non_causal": false, and vLLM then
# serves the drafter with a CAUSAL mask inside the draft block (vllm/transformers_utils/configs/speculators/algos.py),
# although DFlash is bidirectional and Speculators also trains it bidirectionally. That costs ~1 token per call.
[ -d $D/zlab ] || $S/speculators convert z-lab/Qwen3-8B-DFlash-b16 --algorithm dflash --verifier Qwen/Qwen3-8B --output-path $D/zlab

# 3. target hidden states (ids are hidden_states indices, 0 = embeddings; the last layer is appended automatically).
# DFlash: [1 9 17 25 33], z-lab's own numbers taken literally, i.e. one layer shallower than the converted config's
# [2 10 18 26 34]; the drafter is served on the same ids (step 5), so training and serving match. EAGLE-3: defaults.
dump() {  # dump <out dir> [layer ids]
  [ -f $1.done ] && return 0
  env -u PYTORCH_CUDA_ALLOC_CONF setsid $S/python .speculators/scripts/launch_vllm.py train Qwen/Qwen3-8B ${2:+--target-layer-ids $2} \
    -- --port 8011 --gpu-memory-utilization 0.8 --max-model-len 2048 > logs/hidden_$(basename $1).log 2>&1 &
  local vp=$!
  until curl -sf http://localhost:8011/health > /dev/null 2>&1; do kill -0 $vp; sleep 5; done
  $S/speculators generate-offline-data --preprocessed-data $D/prepared --endpoint http://localhost:8011/v1 --output $1 --concurrency 64
  touch $1.done; kill -- -$vp; wait $vp || true; sleep 10
}
dump $D/hidden_states "1 9 17 25 33"
dump $D/hidden_states_eagle

# 4. fine-tune both released drafters with the same recipe: 3 epochs, lr 1e-4, best checkpoint by validation
train() {  # train <kind> <from> <hidden states> <out>
  [ -f $4/done ] && return 0
  $S/torchrun --standalone --nproc_per_node 1 -m speculators.train --speculator-type $1 --verifier-name-or-path Qwen/Qwen3-8B \
    --from-pretrained $2 --data-path $D/prepared --hidden-states-path $3 --on-missing raise --save-path $4 \
    --epochs 3 --total-seq-len 2048 --lr 1e-4 $([ $1 = dflash ] && echo --block-size 16) 2>&1 | tee logs/train_$1.log
  touch $4/done
}
train dflash $D/zlab $D/hidden_states $D/drafter
train eagle3 RedHatAI/Qwen3-8B-speculator.eagle3 $D/hidden_states_eagle $D/eagle

# 5. served variants (same weights, config edits only): layer ids [1 9 ..] and/or the bidirectional block
variant() {  # variant <src> <dst> <set ids 0|1> <non-causal 0|1>
  rm -rf $D/$2; cp -rL $D/$1 $D/$2
  python3 - $D/$2/config.json $3 $4 <<'EOF'
import json, sys
p, ids, nc = sys.argv[1], sys.argv[2] == '1', sys.argv[3] == '1'
c = json.load(open(p))
if ids: c['aux_hidden_state_layer_ids'] = [1, 9, 17, 25, 33]
if nc: c['sliding_window_non_causal'] = True
json.dump(c, open(p, 'w'), indent=2)
EOF
}
variant zlab zlab_l1 1 0
variant zlab zlab_nc 0 1
variant drafter/checkpoint_best drafter_l1 1 0
variant drafter/checkpoint_best drafter_nc 1 1

# 6. benchmark: baselines and results on all 1,000 queries; the diagnosis variants on the first 300
./run_sql_suite.sh qwen3-8b ar dflash eagle7 copyflash63 dflash_sql dflash_sql_nc eagle_sql
LIMIT=300 ./run_sql_suite.sh qwen3-8b dflash_conv dflash_conv_l1 dflash_conv_nc dflash_sql_l1
