#!/usr/bin/env bash
# One-time setup on the GPU box (1x H100 80GB or A100 80GB; Ubuntu; >= 300 GB disk). ~45-60 min, mostly downloads (~115 GB).
# Llama targets are gated: export HF_TOKEN=... (account with Llama 3.1 and 3.2 licences accepted) before running,
# or SKIP_LLAMA=1 to leave them out.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv-vllm --python 3.12
# same vLLM build as the A100 run if the nightly index still has it, else the newest nightly (recorded in env.txt)
uv pip install --python .venv-vllm/bin/python "vllm==${VLLM_PIN:-0.34.0rc1.dev1134+ge72fc9ae5}" --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly \
  || uv pip install --python .venv-vllm/bin/python -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install --python .venv-vllm/bin/python requests
uv venv .venv-torch --python 3.12
uv pip install --python .venv-torch/bin/python "dflash[local]" numpy requests accelerate matplotlib

.venv-torch/bin/python - <<'EOF'
import json, os, sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
T = json.load(open('targets.json'))
skip_llama = os.environ.get('SKIP_LLAMA') == '1'
ids = []
for key, t in T.items():
    if t['gated'] and skip_llama:
        print('skipping gated target', key); continue
    ids += [t['model'], t['dflash'], t['draft']] + ([t['eagle']] if t.get('eagle') else [])
failed = []
for m in dict.fromkeys(ids):
    try:
        print(m, snapshot_download(m, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'tokenizer*', 'merges.txt', 'vocab.json']), flush=True)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print('!!! cannot download', m, type(e).__name__, flush=True); failed.append(m)
if failed:
    print('!!! missing (gated or not found):', failed, '-> set HF_TOKEN with access, or SKIP_LLAMA=1'); sys.exit(1)
EOF

mkdir -p results
{
  date -u
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
  .venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  .venv-torch/bin/python -c "import torch, transformers; print('harness torch', torch.__version__, '| transformers', transformers.__version__)"
  sha256sum predictions.json
} | tee results/env.txt
echo "setup done"
