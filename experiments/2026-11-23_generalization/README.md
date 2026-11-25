# Generalization run: targets × benchmarks × output format (prepared 2026-11-22)

**Question.** Does the A100 result generalize? On Qwen3-8B, copy + one-pass DFlash beat every single-drafter baseline
(DFlash 6.97×, copy + DFlash router 1.33× over DFlash, measured), and a simple performance law predicted held-out
latency within 6–15% ([A100 results](../2026-11-22_h100-bench/README.md)). This run tests the same claims on:

| Axis | Levels |
|---|---|
| Target (all with released DFlash drafters and documented vLLM support) | Qwen3-4B, Qwen3-8B, Llama-3.1-8B-Instruct (family), Qwen3-Coder-30B-A3B-Instruct (code-specialised MoE) |
| Benchmark | edit261 (A100 cohort), **ceb150**: CodeEditorBench bug fixes, 50 each Python / C++ / Java |
| Output format | full rewrite, **diff104**: the 104 CanItEdit files answered as search/replace blocks |
| Hardware | the new GPU vs the A100 (Qwen3-8B rerun) |
| Load | optional: 1 / 4 / 16 concurrent clients on Qwen3-8B |

Methods per target: plain AR, prompt lookup (7/15/31 tokens), small AR draft model (γ = 7: Qwen3-0.6B or Llama-3.2-1B),
EAGLE-3 where a vLLM checkpoint exists (Qwen3-8B, Llama-3.1-8B), DFlash; plus copy + DFlash router in the harness.
Not included: Qwen3.5 (DFlash vLLM support was an unmerged PR on its cards; hybrid attention), Gemma 4 (no model card),
gpt-oss (reasoning model; different output regime). CodeIF-Bench was considered but is from-scratch generation, not editing.

## Pre-registered predictions

`predictions.json` (frozen copy `predictions_frozen_*.json`, sha256 recorded in `results/env.txt` at setup) was written
**before any decoding of the new cohorts**. It applies the law fitted on the A100 Qwen3-8B run to the *reference* answers
only. Key falsifiable predictions for Qwen3-8B (tokens per target call):

| Cohort | New-code share | Prompt lookup (31) | Draft model | EAGLE-3 | DFlash |
|---|---:|---:|---:|---:|---:|
| edit261 | 11% | 5.1 | 6.9 | 4.5 | 9.5 |
| ceb150 (bug fixes, 3 languages) | 9% | 5.6 | 6.9 | 4.5 | 9.7 |
| diff104 (search/replace output) | 33% | **2.5** | 6.3 | 4.0 | **7.2** |

In words: in diff format copying loses most of its value, while one-pass drafting degrades much less. Rows for the
other targets reuse the Qwen3-8B constants unchanged and are labelled exploratory. The copy + DFlash composition is not
pre-registered (from reference structure alone its bound degenerates); it stays a measured result.

## What to rent

- **1× H100 80GB** (preferred) or 1× A100 80GB. 80 GB is required for Qwen3-Coder-30B-A3B (~61 GB of bf16 weights).
  Without it, run with the 30B target removed (edit `KEYS` in `run_all.sh`).
- ≥ 300 GB disk (~115 GB of models).
- **Llama is gated:** `export HF_TOKEN=...` for an account that has accepted the Llama 3.1 and 3.2 licences,
  or `export SKIP_LLAMA=1`.

## Runbook

```sh
rsync -av --exclude '.venv-*' experiments/2026-11-23_generalization/ USER@HOST:~/gen/
ssh USER@HOST; cd ~/gen
export HF_TOKEN=...            # or: export SKIP_LLAMA=1
./setup.sh                     # ~45-60 min; fails early if a gated model is not accessible
tmux new -s gen
./run_all.sh smoke             # ~30 min: every target × cohort; stop and fix if any server fails
./run_all.sh core              # ~6-7 h on H100: all targets, then results/summary.md
./run_all.sh concurrency       # optional, ~30 min
rsync -av USER@HOST:~/gen/results/ experiments/2026-11-23_generalization/results/   # then terminate
```

Order inside `core`: qwen3-8b first (hardware transfer + the strongest prediction tests), then qwen3-4b, llama31-8b,
qwen3coder-30b. `summarize.py` runs after each target, so partial results are usable if the run is cut short.

## Budget (H100 80GB; an A100 80GB is roughly 1.5× slower)

| Stage | Time |
|---|---|
| setup (downloads) | 0.75–1 h |
| smoke | 0.5 h |
| per 8B-class target: 7 vLLM methods × 515 requests + harness (DFlash, router) × 515 | ~1.4 h each (×3) |
| Qwen3-Coder-30B-A3B (MoE; eager harness is slower) | ~2–2.5 h |
| concurrency (optional) | 0.5 h |
| **Total** | **~8–9 h, 11 h cap** |

## Outputs and how they answer the question

`results/summary.md`:
1. Measured speedup per target × cohort for every method, copy + DFlash projected from measured costs, the measured
   router gain over DFlash with a 95% CI, and DFlash's accepted tokens per guess on new code (the size/family axis).
2. Cycle costs per target and output agreement with plain AR.
3. Every pre-registered prediction against its measurement, with the median error for same-target and cross-target rows.

Decision rules: the generality claim holds if copy + DFlash beats the best single method on most target × cohort cells
(measured router gain CI above 1.0), and the law holds if same-target tokens-per-call predictions land within ~15%.
The diff104 cohort is where the claim is most likely to weaken: report it either way.

## Validated on the Mac before renting

- Cohorts built from pinned sources (`prepare_data.py`); C++/Java fences and the search/replace format generate
  correctly with Qwen3-4B (harness, MPS).
- `harness.py`: DFlash and router modes on all three cohorts; stops exactly at end-of-sequence (an earlier version
  counted tokens accepted after EOS; wall-clock unaffected, tokens per call slightly inflated in the A100 harness logs).
- `summarize.py`: end to end on real harness output + synthetic vLLM records, including the prediction table.
- Not testable locally: vLLM itself, Llama access, the 30B MoE. The smoke stage covers all three.

Early signal from the Mac smoke test: on one short Python bug fix the router was slower than DFlash alone (11.8 vs 15.1
tokens per call). Short bug fixes may not benefit from copying; ceb150 will show whether that is systematic.

## Result (2026-11-22): copy + DFlash router in vLLM, measured end to end

Plugin `copyflash_plugin/` registers speculative method `copyflash` through vLLM's custom-speculator API (no vLLM
patch): every step DFlash drafts as usual (this also keeps the drafter's context in sync), vLLM's GPU n-gram kernels
look for the current suffix (≥ 8 tokens) earlier in prompt + output, and per request the copy (up to K tokens) or the
DFlash draft padded to K is verified by the stock greedy rejection sampler. No host sync; CUDA graphs unchanged.
Install: `uv pip install --python .venv-vllm/bin/python -e ./copyflash_plugin`; configs `copyflash`, `copyflash47`,
`copyflash63`, `dflash_pad31/63`, `copyflash63_rm5/12` in `common.vllm_configs(key, extra=True)`.

Qwen3-8B, H100, batch 1, greedy, all methods measured in vLLM (`cf_check.py`, `figures_h100.py`):

| | Code edits (261) | Bug fixes (150) | Diff output (104) |
|---|---:|---:|---:|
| Prompt lookup (31) | 4.42× | 2.52× | 3.65× |
| EAGLE-3 (7) | 3.13× | 3.28× | 2.77× |
| DFlash | 6.76× | 7.48× | 5.48× |
| **Copy + DFlash, 63-token window** | **9.04× (95% CI 8.38–9.75)** | 7.63× | 6.03× |
| gain over DFlash, paired | **1.34× (1.29–1.39)** | 1.02× (0.98–1.06) | 1.10× (1.06–1.15) |

Ablations on code edits: window 31 / 47 / 63 → 8.25× / 8.67× / 9.04×; DFlash padded to 32 / 64 tokens costs 2% / 3%;
router threshold 5 / 8 / 12 → 9.08× / 9.04× / 8.95×. Output byte-identical to AR: 88% (DFlash 87%; the rest are bf16
near-ties). Caveat: the window (63) was chosen on edit261; other targets run with it fixed (held-out check).
Next session: `queue_next.txt` (Coder-30B and Llama with ar/dflash/copyflash63 first, then concurrency incl. the router).

Held-out check (window fixed at 63): **Qwen3-4B edit261 8.81× (CI 8.11–9.55), 1.33× over DFlash (1.28–1.39)**; bug fixes
1.07× (1.03–1.12), diffs 1.09× (1.06–1.13). The 4B router output is byte-identical to AR on 91% vs 99% for DFlash;
`margin.py` (Mac MPS) shows every disagreement is between the target's top-2 tokens (45/46), median gap 0.375 logits
(1–3 bf16 steps), and an independent forward sides with the router in 20/46 -> numerical near-ties from the 64-wide
verify kernels, not a bias. Definitive check queued: `VLLM_BATCH_INVARIANT=1` run (first line of `queue_next.txt`).

## Speedup as a curve (`curves.py`) and the large-file "fast apply" cohort (`prepare_apply.py`)

`figures/curves_<key>.png` (paper) and `figures/curve_social_<key>.png`: speedup over AR binned by copy share (share
of the AR output in lines copied verbatim from the input) and by output length, pooled over all cohorts, 95% CI bands.
Qwen3-8B, 515 requests: ours 6.3× (<50% copied) → 8.3× (50–80%) → 14.1× (80–90%) → 16.3× (90–95%) → 19.5× (≥98%);
DFlash flat at 6–8×; copy-only (prompt lookup 31) 2.5× → 17.0×. By output length ours 5.3× (<100 tokens) → 14.4×
(800–1.6k). Ours is fastest in every bin, but at ≥98% copied, plain copying is close (17.0× vs 19.5×, CIs overlap).

`data/apply100.json`: 100 files of 840–2984 tokens (median 1808) from 35 repositories, gold patches of SWE-bench
Verified (50 Python) and SWE-bench Multilingual (50: Go, Java, JS, PHP, Ruby, Rust, TS, C); median 1.9% of lines
changed. Prompt = original file + edit snippet (changed regions, "... existing code ..." markers) -> full updated file
(the apply step of coding agents). Validated on the Mac with Qwen3-4B: outputs match the reference fix (1 exact, 1 at
0.99 line similarity), 98–99% copied. Queued in `queue_next.txt` for Qwen3-8B with ar / ngram31 / ngram63 / dflash /
copyflash63 (~30 min, AR dominates) and for Qwen3-4B. ngram63 (= copy-only at the same window) is the baseline to
beat here: this is the regime where "speculative edits" style copying is strongest.
