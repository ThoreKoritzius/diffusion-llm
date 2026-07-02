#!/usr/bin/env bash
# PAPL A/B on a rented 1x H100 80GB (also fine on GH200): finetune the base
# 10-epoch checkpoint twice -- tau=0 (plain-LLaDA control) and tau=0.3 (PAPL) --
# under identical data/steps, then print a side-by-side of exact-match and
# avg-steps-to-converge.
#
# The control isolates PAPL's effect: base->tau0 is the gain from extra
# training, tau0->tau0.3 is the gain from the PAPL reweighting itself.
#
# Usage:
#   export WANDB_API_KEY=...                 # optional; offline without it
#   bash run_papl_ab.sh                      # deps check, smoke test, then A/B
#   SKIP_SMOKE=1 bash run_papl_ab.sh         # skip the smoke test
#   TAUS="0 0.3 0.5" bash run_papl_ab.sh     # sweep more than two taus
#
# Knobs (env, with full-run defaults):
#   CKPT_DIR      base checkpoint to finetune     (diffusion-sql-modernbert)
#   TAUS          space-separated taus to run     ("0 0.3")
#   BATCH_SIZE    per-device batch (80GB headroom)(96; try 128 if mem allows)
#   FT_EPOCHS     epochs per run                  (2)
#   FT_LR         learning rate                   (2e-5)
#   TRAIN_SIZE    train examples                  (100000)
#   GEN_EVAL_SIZE eval examples (lower = noisier) (256)
#   CONF_STOP     eval early-stop threshold       (0.9)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"  # inductor breaks ModernBERT+flash-attn+bf16
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-12}"
mkdir -p "$HF_HOME"

CKPT_DIR="${CKPT_DIR:-diffusion-sql-modernbert}"
TAUS="${TAUS:-0 0.3}"
# 80GB has less room than the GH200's 96GB, so default to 96 (the base trained
# at 128 on 96GB). Bump to 128 if `nvidia-smi` shows headroom; drop to 64 on OOM.
export BATCH_SIZE="${BATCH_SIZE:-96}"
export FT_EPOCHS="${FT_EPOCHS:-2}"
export FT_LR="${FT_LR:-2e-5}"
export TRAIN_SIZE="${TRAIN_SIZE:-100000}"
export GEN_EVAL_SIZE="${GEN_EVAL_SIZE:-256}"   # bigger than the Mac smoke -> ~1pt resolution
export CONF_STOP="${CONF_STOP:-0.9}"
export CKPT_DIR

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_DIR/papl_ab_$STAMP"
mkdir -p "$RESULTS_DIR"

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi failed — is this a GPU box?"; exit 1; }

# ---------------------------------------------------------------------------
# Dependencies (x86_64 H100 or aarch64 GH200). Skip entirely inside the NGC
# PyTorch container, which already ships torch + flash-attn.
#   SKIP_DEPS=1   don't touch pip at all (deps already installed)
#   SKIP_FLASH=1  don't attempt the flash-attn source build (use SDPA fallback)
# ---------------------------------------------------------------------------
if [ "${SKIP_DEPS:-0}" != "1" ]; then
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "=== Installing CUDA PyTorch (cu124) ==="
    python3 -m pip install --upgrade pip
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  fi
  echo "=== Installing training deps ==="
  # Training only needs these; the prod-server deps in requirements.txt
  # (flask/gradio/onnx/redis/...) are not needed for the finetune.
  python3 -m pip install "transformers>=4.48.0" datasets accelerate wandb sentencepiece
  if [ "${SKIP_FLASH:-0}" != "1" ]; then
    if python3 -c "import flash_attn" 2>/dev/null; then
      echo "=== flash-attn already present ==="
    else
      echo "=== Attempting flash-attn (best effort; SDPA fallback is fine) ==="
      python3 -m pip install flash-attn --no-build-isolation || \
        echo "!! flash-attn unavailable — ModernBERT will use SDPA (slower but fine)"
    fi
  else
    echo "=== SKIP_FLASH=1 -> ModernBERT will use SDPA (fine for this run) ==="
  fi
fi
python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| bf16={torch.cuda.is_bf16_supported()} | dev={torch.cuda.get_device_name(0)}")
PY

if [ ! -d "$CKPT_DIR" ]; then
  echo "!! base checkpoint '$CKPT_DIR' not found — scp it to the box first."; exit 1;
fi

# ---------------------------------------------------------------------------
# Smoke test: 20 steps, tiny eval, no wandb — catches config/OOM in ~1 min.
# ---------------------------------------------------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "=== Smoke test (tau=0.3, 20 steps) ==="
  OUTPUT_DIR="$RESULTS_DIR/smoke" TAU=0.3 MAX_TRAIN_STEPS=20 TRAIN_SIZE=2000 \
    GEN_EVAL_SIZE=32 WANDB_MODE=disabled python3 src/finetune_papl.py
  rm -rf "$RESULTS_DIR/smoke"
  echo "=== Smoke test passed ==="
fi

# ---------------------------------------------------------------------------
# A/B: one full finetune per tau.
# ---------------------------------------------------------------------------
for TAU in $TAUS; do
  OUT="$RESULTS_DIR/tau_${TAU}"
  echo "=== Finetune tau=$TAU -> $OUT ==="
  OUTPUT_DIR="$OUT" TAU="$TAU" python3 src/finetune_papl.py \
    2>&1 | tee "$RESULTS_DIR/tau_${TAU}.log"
done

# ---------------------------------------------------------------------------
# Side-by-side comparison from the per-run papl_results.json files.
# ---------------------------------------------------------------------------
echo ""
echo "=== PAPL A/B summary ($RESULTS_DIR) ==="
python3 - "$RESULTS_DIR" <<'PY'
import glob, json, os, sys

results_dir = sys.argv[1]
runs = []
for p in sorted(glob.glob(os.path.join(results_dir, "tau_*", "papl_results.json"))):
    with open(p) as f:
        runs.append(json.load(f))

if not runs:
    print("no papl_results.json found"); raise SystemExit(1)

base_em = runs[0]["baseline_exact_match"]
base_st = runs[0]["baseline_avg_steps"]
print(f"base checkpoint:   exact_match={base_em:.3f}   avg_steps={base_st:.2f}\n")

hdr = f"{'tau':>5} | {'exact_match':>11} {'Δvs base':>9} | {'em(aug)':>8} | {'avg_steps':>9} {'Δvs base':>9}"
print(hdr); print("-" * len(hdr))
for r in runs:
    em, st = r["final_exact_match"], r["final_avg_steps"]
    print(f"{r['tau']:>5} | {em:>11.3f} {em - base_em:>+9.3f} | "
          f"{r['final_exact_match_aug']:>8.3f} | {st:>9.2f} {st - base_st:>+9.2f}")

# Isolate PAPL: compare each tau>0 against the tau==0 control if present.
ctrl = next((r for r in runs if float(r["tau"]) == 0.0), None)
if ctrl:
    print(f"\nvs tau=0 control (isolates the PAPL term, not just extra training):")
    for r in runs:
        if float(r["tau"]) == 0.0:
            continue
        d_em = r["final_exact_match"] - ctrl["final_exact_match"]
        d_st = r["final_avg_steps"] - ctrl["final_avg_steps"]
        print(f"  tau={r['tau']}: exact_match {d_em:+.3f}   avg_steps {d_st:+.2f}")
    print("\nGreen light for PAPL if a tau>0 shows avg_steps DOWN at "
          "exact_match equal-or-UP vs the control.")
PY

echo ""
echo "=== Done. Per-run models + papl_results.json under $RESULTS_DIR ==="
echo "If logging offline, sync with: wandb sync wandb/offline-run-*"
