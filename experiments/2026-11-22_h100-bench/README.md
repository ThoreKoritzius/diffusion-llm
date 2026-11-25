# GPU benchmark (A100 or H100): lossless speculative code editing (prepared 2026-11-22)

**Question.** On real code edits, is copy + a one-pass diffusion drafter (DFlash) faster than every single-drafter
baseline and than EfficientEdit-style copy + small AR drafters, *at measured H100 cost*? The Mac work
([copy audit](../2026-11-21_copy-audit/README.md), [DFlash on MLX](../2026-11-21_dflash-mlx/README.md)) projects
8.6–9.8× for copy + DFlash, but with assumed cycle costs. This run replaces the assumptions with measurements.

No training. All decoding is greedy and lossless (output = the target's own greedy output, up to bf16 near-ties).

## What to rent

- **1× A100 40GB SXM4** (recommended) or 1× H100. An A6000 48GB also works at roughly half the speed. Not suitable:
  24 GB cards (8B + drafter + cache barely fit) and Turing cards such as the Quadro RTX 6000 (no bf16, no FlashAttention).
- ≥ 100 GB disk, Ubuntu 22.04/24.04, NVIDIA driver recent enough for CUDA 12.8+ wheels.
- Budget: **about 3 hours on an A100 (4 h cap), about $6**; A6000 about 4–5 h. The optional Qwen3-4B cross-check adds about 40 min.
- Method-vs-method comparisons (gates G2, G3) are valid on any of these GPUs because every method runs on the same card.
  Absolute cycle costs differ between GPUs: an H100 has more compute per byte of bandwidth, so wide verification is
  usually slightly cheaper there. Use the A100 run as the go/no-go; take final paper numbers on the hardware you report.

## Runbook

```sh
# from the repo root on the Mac
rsync -av --exclude '.venv-*' --exclude 'smoke_mac' experiments/2026-11-22_h100-bench/ USER@HOST:~/h100-bench/   # USER is often ubuntu

# on the box
cd ~/h100-bench
./setup.sh                       # ~25 min: two venvs + ~25 GB of models, writes results/Qwen3-8B/env.txt
tmux new -s bench                # everything below survives an ssh drop
./run_all.sh smoke               # ~15 min: 5 edits per key vLLM config, 3 per harness mode, test summary
./run_all.sh core                # ~2 h: 9 vLLM configs + harness + analysis -> results/Qwen3-8B/summary.md
# optional: WITH_4B=1 ./setup.sh && ./run_all.sh extra4b

# back on the Mac, then terminate the instance
rsync -av USER@HOST:~/h100-bench/results/ experiments/2026-11-22_h100-bench/results/
rsync -av USER@HOST:~/h100-bench/logs/ experiments/2026-11-22_h100-bench/logs/
```

Stop after `smoke` if any config fails to start (see `logs/server_*.log`) and fix it before spending the core
budget. Every script resumes: rerunning skips edits already written.

## What runs

**Stage 1, vLLM (`run_vllm_suite.sh` → `bench_vllm.py`)**: one server per method, target Qwen3-8B, batch one,
greedy, prefix caching off, 261 edits. Per edit: time to first token, end-to-end latency, and exact draft-step and
accepted-token counts from Prometheus counter deltas.

| Config | Method |
|---|---|
| `ar` | plain greedy decoding (denominator) |
| `ngram7/15/31` | prompt lookup (copy), 7/15/31 drafted tokens; also calibrates verification cost vs width |
| `draft06b7`, `draft17b7` | Qwen3-0.6B / 1.7B as AR draft model, γ = 7 (EfficientEdit's draft length, untuned) |
| `eagle3_3`, `eagle3_7` | EAGLE-3 (`RedHatAI/Qwen3-8B-speculator.eagle3`), 3 and 7 drafted tokens |
| `dflash15` | DFlash (`z-lab/Qwen3-8B-DFlash-b16`), block 16 |

**Stage 2, PyTorch harness (`harness.py`)**: for what no engine implements. Acceptance counts only; costs come from
Stage 1.
- `dflash`: reproduces `dflash.model.dflash_generate` token for token (verified on the Mac).
- `hybrid_oracle`: copy + DFlash with two-branch tree semantics, both branches verified by the target itself.
- `hybrid_router --router-min 4/8/16`: deployable single-branch policy: copy when the suffix match is long enough, else DFlash.
- `ardraft`: copy + Qwen3-0.6B / 1.7B replays (γ 7 and 15), the EfficientEdit-style lossless configuration.

**Stage 3, `analyze_h100.py`** writes `results/Qwen3-8B/summary.md`:
1. measured speedup, τ, cycle cost and output agreement for every vLLM config, overall and per group;
2. calibrated costs: per-cycle cost in AR steps, and the implied AR-drafter step cost r;
3. projected hybrids = harness acceptance × measured costs (tree overhead = ngram31 − ngram15 cost);
4. decision gates.

## Decision gates

| Gate | Pass condition | If it fails |
|---|---|---|
| G1 cost model | DFlash cycle ≤ 1.8 AR steps (paper's B200 value: 1.55) | the Mac projections were optimistic on this GPU; re-derive the decision curve from section 2 |
| G2 headroom | deployable copy + DFlash ≥ 1.15× the best measured baseline | option A has no margin: pivot to the scientific question (option C) |
| G3 one pass vs AR | copy + DFlash beats the best copy + AR-drafter hybrid at measured r | the "one-pass drafter for edits" claim is false on H100 |

If all pass, the next allocation is training an edit-specialised one-pass drafter, with target: raise accepted
tokens per draft on new code from the measured value toward the 1.7B AR drafter's (6.1–7.8 on the Mac).

## Validated on the Mac before renting

- `harness.py`: all four modes on Qwen3-4B / MPS. `dflash` mode is token-identical to `dflash_generate`;
  outputs in `smoke_mac/`. Hybrids match DFlash output up to bf16 near-ties (both drift at the same position).
- `analyze_h100.py`: end to end on synthetic vLLM records. `bench_vllm.scrape`: on sample Prometheus text.
- Shell scripts: syntax only. **vLLM itself could not be tested locally (no CUDA).** The smoke stage exists to catch
  that on the box.

## Known risks and fallbacks

- **vLLM DFlash support is nightly-only** (model cards: `--extra-index-url https://wheels.vllm.ai/nightly`; README
  mentions vLLM PR 52816). If `dflash15` fails to start: try SGLang from the card
  (`pip install "git+https://github.com/sgl-project/sglang.git@refs/pull/20547/head#subdirectory=python"`,
  `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --attention-backend fa3`;
  on an A100/A6000 replace `fa3`, which is Hopper-only, with `flashinfer`),
  and time it with the same client against its OpenAI endpoint (port 30000; counters differ, so τ then comes from the harness).
- **Metric names**: if Prometheus counters are missing, `drafts` stays 0 and τ/cycle cost are blank; acceptance then
  comes from the harness and cost from latency alone. Check `curl localhost:8000/metrics | grep spec_decode` in the smoke stage.
- **Flags**: if a `vllm serve` flag was renamed in the nightly, the server log says so; edit `run_vllm_suite.sh`.
- **Thinking mode**: prompts are built client-side with `enable_thinking=False` and a prefilled ```` ```python ```` fence, as on the Mac.

## Optional: EfficientEdit's own code (not in the core budget)

Released code ([github](https://github.com/zhu-zhu-ding/EfficientEdit)) needs patching before it runs:
`inference_example.py` hardcodes `CUDA_VISIBLE_DEVICES="1,2,3,4.5"`, processes only `data[:1]`, and has every
method call commented out (NameError on the first item). It pins torch 2.1.2 / transformers 4.39.1, runs eager fp16
HF, and its paper targets are 32–33B models with 7B drafters (LoRA weights on figshare). On one H100 that is about
78 GB of weights, so it is a tight fit. The core run already measures its lossless configuration (copy + AR draft model,
greedy) in vLLM and the harness on a matched target. Reproduce theirs only if a reviewer-facing number in their exact
setup is needed.

## Results: A100-SXM4-40GB, Qwen3-8B, 2026-11-22

Full tables: [results/Qwen3-8B/summary.md](results/Qwen3-8B/summary.md), [postprocess.md](results/Qwen3-8B/postprocess.md),
[diverge.json](results/Qwen3-8B/diverge.json). Environment: vLLM 0.34.0rc1 nightly, torch 2.14 (vLLM) / 2.13 + transformers
5.15 (harness). About 3.3 GPU-hours.

**Measured in vLLM, 261 edits, batch one, greedy (speedup over plain AR, end to end):** DFlash 6.97× (τ 9.29, 1.21 AR
steps per cycle), prompt lookup 4.53× (31 tokens), EAGLE-3 3.18× (7 tokens), Qwen3-0.6B draft model 2.84×, Qwen3-1.7B
2.13×. Implied AR-drafter step cost: 0.17 (0.6B) and 0.30 (1.7B) of a target step.

**Copy + DFlash:**
- Measured in the PyTorch harness on identical requests, deployable router vs DFlash alone: 1.27–1.31× (all 261 edits,
  95% CIs within 1.23–1.36; 1.22–1.28× single functions, 1.30–1.34× full files). Paired repeat, 54 requests × 3
  alternating repeats: **1.33× (95% CI 1.23–1.45)**.
- Projected into vLLM with measured costs: 9.0–9.1× over AR (router), 9.6× (two-branch tree). The projection
  method reproduces measured DFlash (6.96× vs 6.97×), but the hybrid itself has not been run in a serving engine.
- Copy + Qwen3-0.6B (EfficientEdit-style, lossless): 6.07× projected (γ = 7). Gate G3 passes: 9.03× vs 6.07×.

**Gates:** G1 PASS (DFlash cycle 1.21 AR steps), G2 PASS (1.30× over best baseline), G3 PASS.

**Losslessness:** 30/261 outputs differ from plain AR in mid-sequence for every speculative method (none are stop
artifacts). Of 20 diagnosed cases, the top-2 logit margin at the first mismatch is 0–0.5 (median 0.25, i.e. 0–2 bf16
steps), and in 6 the argmax flips between one-token, 16-token-block and full-forward replays of the same prefix. Plain
AR rerun in vLLM reproduced itself in 20/20. So the differences are numerical near-ties in multi-token verification,
not acceptance bugs.

**Drafter resume cost after copy runs:** none measurable (drafted cycles 50.9–51.3 ms across 1 to >128 queued states,
vs 49.9 ms for plain DFlash).

The Qwen3-1.7B copy replay was moved to the Mac (MLX, same recorded outputs; acceptance is hardware independent).

## Performance law (perf_law.py, CPU, 2026-11-22)

[results/Qwen3-8B/perf_law.md](results/Qwen3-8B/perf_law.md). Law: cycles = C/τ_copy + N/τ_new + δ·R from the edit's
copy/new structure; latency = prefill(prompt) + AR step × measured cycle cost × cycles.

- **Held-out edits (20 random splits):** per-edit latency error 6–15% for all 8 measured vLLM methods; aggregate speedup
  within 1–7% (e.g. DFlash 7.11× predicted vs 6.95× measured). Fitted constants are physical: prompt lookup τ_copy ≈ draft
  length (7.2 / 14.0 / 31.8) and τ_new ≈ 1.5; DFlash τ_new = 5.3 (= 4.3 accepted + 1, matching the harness).
- **Group transfer:** model-based drafters transfer between functions and full files within 3–8% on speedup; prompt
  lookup does not (functions → files overpredicts 5.55× vs 4.53×): copy behaviour depends on edit type.
- **Composition:** components bracket the hybrid (8.8 ≤ 10.7 ≤ 14.1 tokens per call on functions; 10.9 ≤ 12.8 ≤ 14.3 on
  files) rather than predict it; the router realises 36–56% of the headroom.
- **Before decoding:** using the benchmark's reference diff instead of the model output predicts aggregate speedup
  within 0–6% (per-edit error 13–41%; new-code share correlates 0.67 between output and reference).

## Hero figure

`hero_figure.py` → [figures/hero_Qwen3-8B.png](figures/hero_Qwen3-8B.png) (and .pdf). Panel (a): end-to-end speedup over
plain AR, measured bars solid, projected bars hatched and labelled. Panel (b): measured router gain over DFlash with 95%
bootstrap intervals. All values are read from `results/Qwen3-8B/`.
