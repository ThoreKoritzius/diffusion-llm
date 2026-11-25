#!/usr/bin/env bash
# One-time setup on the rented H100 box (Ubuntu, NVIDIA driver installed). About 20-30 minutes, mostly downloads.
#   .venv-vllm   vLLM nightly (DFlash support, per the z-lab model cards)
#   .venv-torch  dflash reference stack for the harness (dflash[local] pins torch 2.13 + transformers 5.15 on Linux)
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install --python .venv-vllm/bin/python requests

uv venv .venv-torch --python 3.12
uv pip install --python .venv-torch/bin/python "dflash[local]" numpy requests accelerate

MODELS=(Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16 RedHatAI/Qwen3-8B-speculator.eagle3 Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B)
[ "${WITH_4B:-0}" = 1 ] && MODELS+=(Qwen/Qwen3-4B z-lab/Qwen3-4B-DFlash-b16)
.venv-torch/bin/python - "${MODELS[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    print(m, snapshot_download(m), flush=True)
EOF

mkdir -p results/Qwen3-8B
{
  date -u
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
  nvidia-smi topo -m | head -3
  .venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  .venv-torch/bin/python -c "import torch, transformers; print('harness torch', torch.__version__, '| transformers', transformers.__version__)"
} | tee results/Qwen3-8B/env.txt
echo "setup done"
