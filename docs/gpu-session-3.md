# GPU session 3 runbook

One H100 80 GB (x86), ~4 h, via [`scripts/run_gpu_session3.sh`](../scripts/run_gpu_session3.sh). Goal: a faster
diffusion drafter, and a 1.5B verifier that is more accurate than Qwen2.5-0.5B at no extra latency.

| Step | What | Est. time | GPU |
|---|---|---:|---|
| setup | venvs (uv, parallel), prefetch Qwen2.5-Coder-1.5B + data, checkpoint upload | ~10 min | idle |
| `sweep` | acceptance + draft length of the session-2 drafters on Qwen2.5-0.5B: DFlash k ∈ {3, 5, 7, 11, 15}, EAGLE-3 k ∈ {2, 4, 6, 8}, 300 queries each | ~25 min | alone |
| `train` | in parallel: (a) EAGLE-3 + DFlash for Qwen2.5-0.5B on all ~100k train prompts, 3 epochs; (b) Qwen2.5-Coder-1.5B SFT (1 epoch), then its EAGLE-3 + DFlash (50k prompts) | ~2.5–3 h | shared |
| `bench` | n=1000 at the best k: Qwen2.5-0.5B plain / session-2 / big drafters; Qwen2.5-1.5B plain / EAGLE-3 / DFlash | ~40 min | alone |

Every speculative run records acceptance (`spec` in its summary: mean accepted tokens per draft, rate per position).
`results/gpu-session-3/summary.md` has the table; `ALL_DONE` marks the end. Cut time with `QWEN15_EPOCHS`,
`SPEC_N_BIG`, `SPEC_N_15`, `DRAFT_EPOCHS`, or `STEPS="sweep"` for a ~35 min session.

## On the box

```bash
git clone https://github.com/ThoreKoritzius/diffusion-llm.git && cd diffusion-llm && mkdir -p checkpoints
umask 077; echo 'export WANDB_API_KEY=...' > ~/.wandb_env    # optional
tmux new -s s3 'bash scripts/run_gpu_session3.sh; bash'
```

From your machine (~1.6 GB), while setup runs:

```bash
cd ~/Documents/projects/diffusion-prod/diffusion-llm/checkpoints
rsync -avP sql-ar-qwen25-coder-0.5b-ar spec-qwen25 <user>@<host>:diffusion-llm/checkpoints/ \
  && ssh <user>@<host> touch diffusion-llm/checkpoints/.uploaded
```

Copy back `results/gpu-session-3/`, `checkpoints/sql-ar-qwen25-coder-1.5b-ar` and the `checkpoints/spec-*/{eagle3,dflash}`
drafters before terminating (hidden states are deleted after each drafter run).

## Known issues handled by the script

- FlashInfer JIT needs `ninja` on PATH (linked into `~/.local/bin`).
- Speculators 0.8.1 DFlash passes `head_dim=None` for Qwen2-family verifiers; patched in place to `resolved_head_dim`.
- vLLM's hidden-states connector refuses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`; unset for that server.
- n-gram speculation is 2× slower on vLLM 0.31 (V1-runner fallback), so it is not swept.
