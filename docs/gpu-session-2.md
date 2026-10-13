# GPU session 2 runbook

One H100 80 GB (x86, for vLLM wheels), ~10–12 h, via [`scripts/run_gpu_session2.sh`](../scripts/run_gpu_session2.sh).

| Track | What | Est. time | Question |
|---|---|---:|---|
| `epochs` | Ettin-400M diffusion +10 epochs (20 total), distill, then RL | ~6–8 h | Does diffusion keep gaining with epochs at 400M? |
| `rl` | Execution-reward RL ([`src/train_rl.py`](../src/train_rl.py)) on the current 400M | ~1–1.5 h | Does RL fix invalid SQL (13% at 400M vs Qwen's 4%)? |
| `spec` | Qwen3-0.6B-Base SFT; better Qwen = verifier; EAGLE-3 + DFlash drafters with Speculators | ~3–4 h | Does a diffusion drafter make AR faster than EAGLE / n-gram at identical output? |
| bench | after all tracks, alone: 400M (+verify-1) for every checkpoint; Qwen on vLLM plain / ngram / EAGLE-3 / DFlash | ~1 h | Latency without training running next to it |

All quality numbers use 1,000 test queries; RL and drafter data use only train rows.

## Before renting

Tested on a Mac (CPU): `src/train_rl.py` end to end (sampling, SQLite reward, coupled-mask gradient, KL, eval, save;
reward rises on a 3-prompt overfit check), `bash -n` on the script. Not testable here: everything vLLM / Speculators
(spec data, hidden states, drafter training, speculative serving). DFlash in Speculators is validated end to end only
for Qwen3-8B; a 0.5B target is untested upstream.

## On the box

```bash
git clone https://github.com/ThoreKoritzius/diffusion-llm.git && cd diffusion-llm && mkdir -p checkpoints
```

From your machine (~4 GB):

```bash
cd ~/Documents/projects/diffusion-prod/diffusion-llm/checkpoints
rsync -avP --exclude 'checkpoint-*' ettin400-diffusion-iso-10ep ettin400-diffusion-iso-10ep-distilled \
  sql-ar-qwen25-coder-0.5b-ar <user>@<host>:diffusion-llm/checkpoints/
```

Then, in tmux:

```bash
tmux new -s s2 'bash scripts/run_gpu_session2.sh; bash'
tail -f results/gpu-session-2/session.log
```

Results: `results/gpu-session-2/summary.md` and `bench/*.summary.json`; copy them back before terminating.
