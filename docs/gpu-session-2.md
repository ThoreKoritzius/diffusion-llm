# GPU session 2 runbook

One H100 80 GB (x86 for the vLLM wheels, a recent NVIDIA driver; the log prints its version), ~10–12 h, via [`scripts/run_gpu_session2.sh`](../scripts/run_gpu_session2.sh).

| Phase | What | Est. time | GPU |
|---|---|---:|---|
| setup | venvs with uv in parallel, prefetch Qwen3 / Ettin decoders / data; checkpoint upload from your machine | ~5–10 min | idle |
| `vllmpr` | [vllm#61276](https://github.com/vllm-project/vllm/pull/61276) on the Ettin decoders 17M–1B: its tests, then HF-vs-vLLM greedy agreement and speed ([`scripts/vllm_ettin_check.py`](../scripts/vllm_ettin_check.py)) | ~15 min | alone, during the upload |
| `baseline` | batch-1 speed, n=1000: 400M diffusion (±verify 1), Qwen2.5 on vLLM plain and n-gram | ~15 min | alone |
| `epochs` | Ettin-400M diffusion +10 epochs (20 total), distill, then RL | ~6–8 h | shared |
| `rl` | Execution-reward RL ([`src/train_rl.py`](../src/train_rl.py)) on the current 400M | ~1–1.5 h | shared |
| `spec` | Qwen3-0.6B-Base SFT; better Qwen = verifier; EAGLE-3 + DFlash drafters with Speculators | ~3–4 h | shared |
| bench | every new checkpoint; Qwen on vLLM plain / ngram / EAGLE-3 / DFlash | ~1 h | alone |

`epochs` is the long pole: `EPOCHS_MORE=5` saves ~3 h. All quality numbers use 1,000 test queries; RL and drafter
data use only train rows. `results/gpu-session-2/summary.md` is rewritten after `baseline` and at the end;
`results/gpu-session-2/ALL_DONE` marks the end (terminate the box then).

## Before renting

Tested on a Mac (CPU): `src/train_rl.py` end to end (sampling, SQLite reward, coupled-mask gradient, KL, eval, save;
reward rises on a 3-prompt overfit check), `bash -n` on the script. Not testable here: everything vLLM / Speculators
(spec data, hidden states, drafter training, speculative serving, #61276 on GPU). DFlash in Speculators is validated end to end only
for Qwen3-8B; a 0.5B target is untested upstream.

## On the box

```bash
git clone -b feat/gpu-session-2 https://github.com/ThoreKoritzius/diffusion-llm.git && cd diffusion-llm && mkdir -p checkpoints
```

From your machine (~4 GB):

```bash
cd ~/Documents/projects/diffusion-prod/diffusion-llm/checkpoints
rsync -avP --exclude 'checkpoint-*' ettin400-diffusion-iso-10ep ettin400-diffusion-iso-10ep-distilled \
  sql-ar-qwen25-coder-0.5b-ar <user>@<host>:diffusion-llm/checkpoints/ && ssh <user>@<host> touch diffusion-llm/checkpoints/.uploaded
```

Start the script right after cloning (it sets up and runs the vLLM checks during the upload), in tmux:

```bash
tmux new -s s2 'bash scripts/run_gpu_session2.sh; bash'
tail -f results/gpu-session-2/session.log
```

Results: `results/gpu-session-2/summary.md` and `bench/*.summary.json`; copy them back before terminating.
