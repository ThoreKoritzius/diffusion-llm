# GPU session runbook

One unattended session on a single H100 (80 GB) or GH200 (96 GB) runs every experiment below via
[`scripts/run_gpu_session.sh`](../scripts/run_gpu_session.sh). Expected wall time: **~5–6 h** (estimates from
earlier runs: ModernBERT-base trained at 515 samples/s on GH200 with flash-attn, Qwen 0.5B AR at 79 samples/s on
H100 PCIe; masked runs use sdpa, assumed ~2× slower).

| Job | What | Est. time | Question it answers |
|---|---|---:|---|
| `isolation` | ModernBERT, +2 epochs from the current checkpoint, **with** the one-way prompt mask; and a control with the same +2 epochs **without** it | ~0.5–1 h | Does prompt caching cost quality once the model is trained for it? (Go if exec drops ≤2 points vs. control.) |
| `ettin150`, `ettin400` | Ettin encoder → masked diffusion vs. Ettin decoder → AR, same data, 3 epochs each | ~2–3 h | Diffusion vs. AR at matched size and pretraining data |
| `qwen35` | Qwen3.5-0.8B-Base AR fine-tune (transformers 5) | ~1.5 h | The strongest small AR model to beat |
| `bench` | Exec accuracy + batch-1 GPU latency for every checkpoint: diffusion fp32/bf16; AR fp32, bf16 + `torch.compile` static cache, and vLLM | ~0.5 h | The GPU half of the fair comparison |

The Ettin diffusion arms use the prompt mask only if it held up in `isolation` (`ETTIN_ISOLATION=auto`; force with
`0`/`1`).

## H100 or GH200

Both are plenty for these model sizes. **Prefer an x86 H100** if prices are similar: vLLM ships x86 wheels, while on
GH200 (ARM) the vLLM install may fail. The script then still benchmarks AR with `torch.compile`, just without vLLM.

## 1. Before renting

Everything was dry-run on a Mac (CPU, 17M Ettin models, 2-step trainings): all training scripts, the quality gate,
resume, benchmarks and the summary table. Untested here: CUDA-only paths (vLLM, `--ar-compile`, flash-attn installs).

## 2. On the box

```bash
git clone https://github.com/ThoreKoritzius/diffusion-llm.git && cd diffusion-llm
git checkout main            # after the GPU-session PR is merged
mkdir -p checkpoints
```

From your machine, upload the current model (~575 MB; skip the intermediate `checkpoint-*` folders):

```bash
rsync -avP --exclude 'checkpoint-*' \
  ~/Documents/projects/diffusion-prod/diffusion-llm/diffusion-sql-modernbert/ \
  <user>@<gpu-host>:diffusion-llm/checkpoints/diffusion-sql-modernbert/
```

Run inside tmux so a dropped SSH connection doesn't kill it:

```bash
tmux new -s session
export WANDB_API_KEY=...        # optional; otherwise logs are kept offline
bash scripts/run_gpu_session.sh
# detach: Ctrl-b d   · reattach: tmux attach -t session
tail -f results/gpu-session-*/session.log
```

Each job runs a 20-step smoke test first, so a config or out-of-memory error shows up within a minute. A failed job
is logged and the session continues. Re-running the script skips finished checkpoints, so it resumes after an
interruption. Useful knobs: `JOBS="isolation bench"`, `SKIP_SMOKE=1`, `EPOCHS=3`, `ETTIN400_BATCH=64`,
`INSTALL_FLASH_ATTN=1`, `TORCH_INDEX=...` (default cu126 wheels).

## 3. Bring results back

```bash
rsync -avP <user>@<gpu-host>:diffusion-llm/results/ results/
rsync -avP --exclude 'checkpoint-*' <user>@<gpu-host>:diffusion-llm/checkpoints/ checkpoints/
```

`results/gpu-session-*/summary.md` has the table; per-example predictions are in `bench/*.jsonl`, training logs in
`logs/`. Then run the CPU track locally on the new checkpoints (`python -m bench.run ... --engine onnx`; the
prompt-isolated models currently run on the torch engine).

## Local dry run of the script

```bash
mkdir -p checkpoints && ln -s ../../diffusion-llm/diffusion-sql-modernbert checkpoints/diffusion-sql-modernbert
JOBS="isolation ettin17 bench" ETTIN_SIZES=17 PY=.venv/bin/python BENCH_DEVICE=cpu BENCH_N=4 SKIP_SMOKE=1 \
  WANDB_MODE=disabled EXTRA_TRAIN_ENV="MAX_TRAIN_STEPS=2 TRAIN_SIZE=16 BATCH_SIZE=2 GRAD_ACCUM=1 VAL_SIZE=4 \
  GEN_EVAL_SIZE=2 EVAL_STEPS=2 DATALOADER_WORKERS=0" bash scripts/run_gpu_session.sh
```
