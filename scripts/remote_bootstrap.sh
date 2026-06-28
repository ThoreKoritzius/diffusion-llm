#!/usr/bin/env bash
# Build an isolated venv on the GH200 box so torch/numpy/datasets are a
# self-consistent set (the system site-packages mix numpy 1.x and 2.x ABIs).
set -euo pipefail
cd /home/ubuntu/diffusion-llm

VENV=/home/ubuntu/dvenv
if [ ! -d "$VENV" ]; then
  echo "=== Creating venv $VENV ==="
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "=== Installing CUDA torch (aarch64, cu124) ==="
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing project deps ==="
python -m pip install -r requirements.txt

echo "=== Sanity: torch + numpy + datasets + transformers ==="
python - <<'PY'
import numpy, torch, pandas
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "bf16", torch.cuda.is_bf16_supported(), "dev", torch.cuda.get_device_name(0))
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
print("transformers", transformers.__version__, "imports OK")
PY
echo "=== Bootstrap OK ==="
