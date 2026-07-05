# PAPL A/B — text-to-SQL masked diffusion (2026-06-30)

**TL;DR (two independent findings):**

1. **PAPL is a narrow latency optimization, not a quality upgrade.** τ=0.3 buys a
   real but modest **~7% reduction in denoising steps at zero accuracy gain**,
   and *only* under adaptive `confidence_stop`. At fixed step budgets it is no
   better than base. Confirmed across four semantic evaluations (quality bar,
   failure modes, early-stop frontier, fixed-NFE) plus the exact-match A/B.
2. **The model's real quality is ~47%, not ~29% — exact-match was misleading.**
   A GPT-5.4-mini semantic judge shows string exact-match undercounts correct SQL
   by ~60% relative. The biggest real weakness is **syntax-invalid output (~14%)**,
   then schema-grounding (wrong column/filter/aggregate ~30% combined) — none of
   which PAPL touches.

**Verdict:** don't ship PAPL as a model swap (base is as good, marginally better
at fixed budget); keep it only if a ~5% early-stop latency cut matters in prod.
The high-impact next work is **syntax-valid decoding + schema grounding**, not
PAPL variants. See the [Verdict](#verdict) below.

## What was run

Finetune the converged 10-epoch LLaDA-style checkpoint with the PAPL
confidence-reweighted loss, against a τ=0 control (plain LLaDA loss), identical
data/steps. Then a frontier benchmark across all three models.

| | base | τ=0 control | τ=0.3 PAPL |
|---|---|---|---|
| recipe | 10-ep LLaDA | +2ep, lr 2e-5 | +2ep, lr 2e-5, **τ=0.3** |
| checkpoint dir | `diffusion-sql-modernbert/` | `diffusion-sql-modernbert-papl-tau0/` | `diffusion-sql-modernbert-papl/` |

- Backbone: ModernBERT-base · dataset: `gretelai/synthetic_text_to_sql`
- Hardware: 1× H100 PCIe 80GB (Lambda), torch 2.7, transformers 4.49, SDPA (no flash-attn)
- Eval: generation exact-match + avg steps-to-converge (batch-1 confidence-planner decoding)

## Results

All four panels use **semantic accuracy** (GPT-5.4-mini judge, n=256) as the
primary metric; exact string-match appears only in panel 1 as the contrast.

![summary](plots/summary.png)

### 1 · Real quality ≈ 47%, not ≈ 29%

![quality](plots/quality.png)

Exact string-match undercounts correct SQL by ~18 points (**~60% relative**) —
aliases, column/JOIN order, equivalent predicates, and cases where the prediction
answers the question *better* than the gold. Judge validation: on rows where the
strings matched exactly the judge agreed 98–100%, so the lift is real, not
inflation. **The model is moderate (~47%), not the near-failure the 29% implied.**

### 2 · Failure modes — syntax errors dominate

![failures](plots/failures.png)

`syntax_error` ~14% is the single biggest failure (invalid SQL), then
`wrong_or_missing_column` ~11%, `missing_or_wrong_filter` ~10%,
`wrong_aggregate_or_groupby` ~9% (schema grounding). **Nearly identical across all
three models** — PAPL changes neither the rate nor the distribution of failures.
The clear, actionable levers are syntax validity and schema grounding.

### 3 · Speed/quality frontier (early-stop sweep)

![frontier](plots/frontier.png)

Semantic accuracy vs decoding steps across the `confidence_stop` sweep (n=256).
Base (grey) and τ=0.3 (blue) as emphasized curves, τ=0 control dashed for context;
light ties join equal-threshold points, labeled once. The read: **τ=0.3 traces the
same hump as base shifted left** (same accuracy, ≈7% fewer steps — PAPL's entire
effect), the hump peaks at 0.9 (~11 steps), and past it more decoding loses
accuracy. Model accuracy gaps are within ±1 s.e. (reference whisker in plot).
The purple dash-dot line adds the **base + verify-repair** config from the
follow-up below: base shifted slightly right, not up — its only (sub-s.e.)
gains are on the over-decoding tail (0.95/0.99), where more outputs break and
repair has more to fix:

| conf_stop | base | τ=0 | τ=0.3 |
|---|---|---|---|
| 0.8 | 0.477 @ 9.0 | 0.480 @ 8.9 | 0.461 @ 8.3 |
| 0.9 | **0.500 @ 11.8** | 0.484 @ 11.4 | 0.484 @ 10.9 |
| 0.95 | 0.469 @ 14.4 | 0.484 @ 14.1 | 0.465 @ 13.4 |
| 0.99 | 0.406 @ 19.5 | 0.430 @ 18.9 | 0.434 @ 18.1 |

Two findings: (a) **the frontier is non-monotonic** — quality *peaks* around
conf_stop=0.9 (~11 steps) and *drops* ~10 points by 0.99 (~19 steps). More
decoding hurts ("locally confident, globally stuck"); the sweet spot is ~11 steps.
(b) The three models overlap in accuracy at every threshold (ordering flips), but
**τ=0.3 sits consistently left** — ~7% fewer steps at matched accuracy. That step
saving is PAPL's one real effect, now confirmed on the semantic metric.

### 4 · Quality per compute (fixed-NFE)

![nfe](plots/nfe.png)

Forced fixed step budget K, no early stop:

| K | base | τ=0 | τ=0.3 |
|---|---|---|---|
| 4 | 0.121 | 0.113 | 0.117 |
| 8 | 0.164 | 0.156 | 0.148 |
| 16 | 0.336 | 0.344 | 0.332 |
| 24 | 0.438 | 0.422 | 0.430 |

All three **tied at every budget** (gaps ~1–4 examples, within ~3% SE; ordering
flips). The exact-match version *looked* like the finetune eroded fixed-budget
quality (base 0.293 vs τ=0.3 0.281 at K=24) — but that was a ~3-example artifact
that vanishes under semantic grading. Note quality climbs 0.12 → 0.44 across
K=4→24: more steps genuinely help in the forced-budget sense.

## Verdict

1. **PAPL = speed-only.** Zero quality change across all four evaluations (quality
   bar, failure modes, frontier, fixed-NFE — semantic *and* exact). Its sole effect
   is **~7% fewer denoising steps under adaptive `confidence_stop`**, via
   certainty-forcing (crosses the threshold sooner). Not a better model.
2. **Don't ship PAPL as a model swap** — base is as good at every operating point.
   Keep τ=0.3 only if a ~7% early-stop latency cut matters in prod; then run it at
   conf_stop≈0.9 (the frontier sweet spot).
3. **Real quality is moderate (~47%), not the ~29% exact-match implied** — but 47%
   with ~14% invalid SQL is still weak, and we have **no AR baseline or latency
   measurement**, so "diffusion wins on quality or speed" remains unproven.
4. **Highest-impact next work: syntax-valid decoding + schema grounding**, not any
   PAPL/loss-reweighting variant. Those attack the failure modes that actually
   dominate; PAPL touches none of them.

## Follow-up: training-free inference improvements (2026-07-02)

Tested the two training-free levers suggested by the failure analysis, on
base @ conf_stop 0.9, n=256 (scripts: `src/sql_repair.py`, `refine_frac` in
`src/denoising.py`; results: `data/judge_inference_summary.json`):

| config | steps | parse-valid | exact | semantic |
|---|---|---|---|---|
| baseline | 11.79 | 85.9% | 0.320 | 0.500 |
| + sqlglot verify-repair | 12.46 | **93.8%** | 0.320 | 0.504 |
| + targeted-remask refinement | 13.45 | 85.9% | 0.328 | 0.508 |
| + both | 14.11 | **93.8%** | 0.328 | 0.508 |

**Verdict: validity improved for real, semantics didn't.** Repair fixed 20/36
invalid queries (syntax), but only **1/20 became semantically correct** — the
rest turned valid-but-wrong (e.g. `INSERT ... SELECT (1, 1, '` → parseable
garbage). Refinement re-predicted shaky commits without changing outcomes
(all deltas ≤ 2/256, within ±1 s.e.).

![repair_effect](plots/repair_effect.png)

The histogram shows the mechanism: the `syntax_error` bucket shrinks, but the
freed mass migrates *into the wrong-content buckets* (wrong column, hallucinated
identifier, wrong join) — not out of the failure distribution. Correct answers:
50.0% → 50.4%.

**The lesson:** invalid SQL from this model is a *symptom of confusion, not a
typo* — the syntax-error bucket overlaps almost entirely with "model doesn't
know the answer", so decoding-level fixes can make outputs parseable but not
correct. The earlier projection (+5–10 pts from syntax recovery) was wrong.
Repair is still worth keeping for production robustness (7.9 pts fewer hard
failures downstream, +0.7 steps), but **accuracy gains require training-level
levers** (RL with verifiable reward, better data) — not sampling tricks.

## Reproduce

```bash
# rebuild all figures from the archived judge summaries
python3 make_report_plots.py

# frontier benchmark (exact-match, needs the three checkpoints + a GPU box)
python3 ../../src/bench_papl.py <papl_ab_run_dir>

# regenerate predictions + semantic grading (needs OPENAI_API_KEY)
python3 ../../src/dump_predictions.py . DUMP_N=200                              # early-stop
FIXED_STEPS=4,8,16,24 DUMP_OUT=predictions_nfe python3 ../../src/dump_predictions.py .
CONF_SWEEP=0.8,0.9,0.95,0.99 DUMP_OUT=predictions_frontier python3 ../../src/dump_predictions.py .
OPENAI_API_KEY=... python3 judge_sql.py .                        # early-stop headline + failures
OPENAI_API_KEY=... python3 judge_nfe.py predictions_nfe          # fixed-NFE curve
OPENAI_API_KEY=... python3 judge_frontier.py predictions_frontier  # frontier
```

## Contents

```
data/
  judge_summary.json           semantic accuracy + failure histograms (early-stop headline)
  judge_nfe_summary.json       semantic fixed-NFE curve (per model × K)
  judge_frontier_summary.json  semantic frontier (per model × confidence_stop)
  bench_results.json           original exact-match frontier metrics
  papl_results_tau0{,.3}.json  per-arm A/B metrics
  provenance.json              full recipe + lineage of the τ=0.3 checkpoint
predictions/                   early-stop predictions + judge verdicts (pred/graded_<model>.jsonl)
predictions_nfe/               fixed-budget preds + verdicts   (…_<model>_k<K>.jsonl)
predictions_frontier/          conf_stop-sweep preds + verdicts (…_<model>_c<cs>.jsonl)
plots/
  summary.png  quality.png  failures.png  frontier.png  nfe.png
make_report_plots.py           builds the five figures from the judge summaries
judge_sql.py / judge_nfe.py / judge_frontier.py   GPT-5.4-mini semantic judges
```

Scripts also in `src/`: `finetune_papl.py` (training), `bench_papl.py` (exact-match
benchmark), `dump_predictions.py` (prediction dump; early-stop / FIXED_STEPS / CONF_SWEEP modes).
Checkpoints live in their `diffusion-sql-modernbert*` dirs (not copied here — too large).
