# Qwen AR vs Diffusion Text-to-SQL

Date: 2026-07-05

This experiment compares an autoregressive SQL SFT baseline against a direct masked-diffusion adaptation using the same base model, dataset, tags, and evaluation slice.

## Setup

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B` |
| Dataset | `gretelai/synthetic_text_to_sql` |
| Train split | first 100,000 filtered train rows |
| Eval split | first 256 filtered test rows for this report |
| Training epochs | 3 |
| Max sequence length | 512 |
| SQL window | 128 tokens (AR + original diffusion); 64 tokens for diffusion v2 |
| Tags | `<PROMPT>`, `<CONTEXT>`, `<SQL>` |
| Judge | `gpt-5.4-mini` |

The original target was `Qwen/Qwen3.5-0.8B-Base`, but the available Transformers 4.x stack did not recognize `model_type=qwen3_5`, while Transformers 5.x broke the current Trainer API. The run uses the validated Qwen2.5-Coder fallback.

## Current Results

| Model | Params | Exact | SQLite exec | SQL valid | Semantic judge | Avg steps/tokens | H100 latency | Mac latency² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **AR-Qwen** (Qwen2.5-Coder-0.5B, causal SFT) | 500M | 0.355 | 0.562 | 0.754 | **0.645** | 29.86 tokens | ~400 ms | **898 ms** (p50 722) |
| **Diffusion-Qwen v2** (same base, masked-diffusion adaptation, fixed recipe, ckpt-4689) | 500M | 0.121 | 0.180 | 0.219 | **0.191** | 11.8 denoise steps | **~150–195 ms** | 1467 ms³ |
| **Diffusion-Qwen v1** (confounded pad=EOS recipe, ckpt-3126) | 500M | 0.102 | 0.164 | 0.254 | 0.148 | 24.00 denoise steps | ~360 ms | 2927 ms |
| **Diffusion-ModernBERT** (ModernBERT-base, native MLM, 10 ep, [PAPL A/B](../2026-06-30_papl-ab/README.md)) | 150M | 0.320 | — | 0.859¹ | **0.500** | 11.79 denoise steps | — | 1209 ms (9.2 steps) |

¹ sqlglot parse-valid, not SQLite exec-valid (different harness; the PAPL runs
did not record SQLite execution). All rows share the same dataset, the same
first-256 filtered gretelai test slice, and the same judge protocol
(`judge_sql.py`, gpt-5.4-mini; judge validation 1.0 on every graded run).
² Uniform local benchmark (`bench_local_latency.py`): same Mac, MPS, fp16,
eager attention, batch 1, same 32 eval examples, 3 warmups, each model at its
documented decode operating point (`data/local_latency.json`). Supersedes the
earlier mixed-config local numbers quoted in the details sections below.
³ Under fp16 the v2 model's calibrated confidences shift, so `confidence_stop`
never fired and it ran the full 12-step cap (vs 11.8 avg under bf16) — its Mac
number is a ≤2% overestimate. Quality columns are from the bf16/fp32 evals.

**The latency ranking is hardware-dependent — and inverts.** On the H100,
Diffusion-Qwen v2 is ~2.4× faster than AR-Qwen (11.8 wide parallel passes beat
~30 sequential ones when the hardware crushes a 512-token forward). On the Mac
(overhead/bandwidth-bound, batch 1), **AR is fastest**: ~30 tiny KV-cached
steps cost less than 12 full-sequence passes. Diffusion's speed edge exists on
server-class parallel hardware and disappears on local/edge inference — the
deploy target decides which column matters.

AR exact-match undercounts correctness materially: semantic accuracy is +0.289 over exact match on 256 judged samples. The Qwen diffusion arms get almost no such lift (v2 final: +0.070) — their wrong outputs are mostly *invalid SQL* (65.6% syntax_error for v2), not cosmetically-different-but-correct SQL.

## AR Details

Training:

- Script: `src/train_ar.py`
- Objective: causal LM SFT
- Loss mask: SQL completion only
- Final train loss: 0.2508
- Final eval loss: 0.2126
- W&B run: `https://wandb.ai/tkoritzius/sql-diffusion/runs/sfayjedq`

256-sample benchmark:

- Exact match: 0.3555
- SQLite execution match: 0.5625
- SQLite executable/valid rate: 0.7539
- LLM semantic accuracy: 0.6445
- Average generated tokens: 29.8555
- Average local latency: 588.66 ms

Failure modes from the semantic judge:

| Category | Count | Share |
|---|---:|---:|
| wrong_or_missing_column | 29 | 0.113 |
| wrong_aggregate_or_groupby | 28 | 0.109 |
| missing_or_wrong_filter | 18 | 0.070 |
| wrong_table | 5 | 0.020 |
| wrong_join | 4 | 0.016 |
| hallucinated_identifier | 3 | 0.012 |
| wrong_order_or_limit | 3 | 0.012 |
| other | 1 | 0.004 |

## Diffusion Details

The first diffusion run was pulled locally from remote checkpoint `sql-diffusion-qwen0.8b/checkpoint-3126` and evaluated on the same first 256 filtered test rows as AR. This checkpoint comes from the original direct Qwen diffusion adaptation recipe.

256-sample benchmark for the pulled checkpoint:

- Exact match: 0.1016
- SQLite execution match: 0.1641
- SQLite executable/valid rate: 0.2539
- LLM semantic accuracy: 0.1484
- Average denoising steps: 24.0000
- Average local latency: 3567.99 ms

This pulled-checkpoint number is far below AR on exact, execution, and validity. The local latency is not a useful speed comparison because it ran on local MPS/CPU-class hardware while the training eval latencies below are H100 numbers.

Failure modes from the semantic judge:

| Category | Count | Share |
|---|---:|---:|
| syntax_error | 137 | 0.535 |
| missing_or_wrong_filter | 29 | 0.113 |
| truncated_or_empty | 20 | 0.078 |
| wrong_or_missing_column | 15 | 0.059 |
| hallucinated_identifier | 7 | 0.027 |
| wrong_aggregate_or_groupby | 5 | 0.020 |
| wrong_table | 4 | 0.016 |
| wrong_order_or_limit | 1 | 0.004 |

The Qwen2/Transformers 4.57 mask path required patching because Qwen2 now calls module-level `create_causal_mask` directly instead of exposing `_update_causal_mask`.

The patched diffusion smoke test passed:

- `[bidir] verified: attention is bidirectional.`
- 1-step train/eval completed

Original diffusion run:

- Script: `src/train_diffusion_qwen.py`
- Objective: masked denoising over the fixed SQL window
- Mask token: `<|mask|>`
- Attention: patched to padding-only bidirectional mask
- Loss: continuous-t Bernoulli masking with `1/t` weighting
- W&B run: `https://wandb.ai/tkoritzius/sql-diffusion/runs/jodhkup5`

Important caveat: this original diffusion recipe is confounded. Qwen has no native pad token, so the script set `PAD_ID == EOS_ID`; the fixed 128-token SQL window is mostly padding for typical SQL examples, and the diffusion collator samples masked/loss positions uniformly across the full SQL window. This means the loss and generation behavior are partly dominated by padding/EOS behavior, not only SQL content.

## Diffusion v2 (fixed recipe) — completed run

The v2 run (`sql-diffusion-qwen2.5-coder-0.5b`, W&B `2l0tp3yi`) fixes the v1
confounds and completed 3 epochs on the H100 (4689 steps, 1.66 h) with no
instability. The v1 → v2 changes:

1. **Distinct `<|pad|>` token** (v1 aliased pad to EOS; 76% of the 128-token
   window taught "emit EOS" → output-length collapse at step ~2000).
2. **SQL window 128 → 64** (median gold SQL is 27 tokens, p90 = 54; halves the
   pad fraction and the decode cost).
3. **Pad CE down-weighted 0.1×** (loss/eval_loss tracks SQL tokens, not padding).
4. **Causal→bidirectional annealing** (additive −20·α bias on future positions,
   linear over the first 30% of steps, instead of a hard mask switch).
5. **Dream-style shifted prediction** (masked token *i* is read from the logits
   at *i−1*, in training and decoding, preserving the AR-pretrained head).
6. **SDPA attention** (4D float mask; ~2–3× faster than v1's eager), lr 1e-5,
   warmup 10%.
7. Eval decoding: `confidence_stop=0.9`, 12-step cap (the PAPL frontier sweet
   spot) instead of 24 fixed steps.

Training-time gen-evals, n=32 (α = causal-anneal coefficient; the run is only
fully bidirectional once α = 0; the scheduled step-2000 eval never fired —
trainer quirk, not a crash):

| Step | Exact | SQLite exec | SQL valid | Avg steps | H100 latency | α |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.031 | 0.031 | 0.031 | 12.0 | 193 ms | 0.65 |
| 1000 | 0.156 | 0.219 | 0.250 | 11.8 | 173 ms | 0.29 |
| 1500 | 0.219 | 0.250 | 0.344 | 11.7 | 186 ms | 0.00 |
| 2500 | 0.188 | 0.156 | 0.281 | 11.7 | 147 ms | 0.00 |
| 3000 | 0.188 | 0.219 | 0.375 | 11.8 | 153 ms | 0.00 |
| 3500 | 0.125 | 0.125 | 0.250 | 11.8 | 195 ms | 0.00 |
| 4000 | 0.188 | 0.188 | 0.312 | 11.7 | 176 ms | 0.00 |
| 4500 | 0.188 | 0.219 | 0.344 | 11.8 | 157 ms | 0.00 |
| 4689 | 0.188 | 0.188 | 0.312 | 11.8 | 212 ms | 0.00 |

Two findings: (a) the v1 collapse is gone — no instability anywhere, and
eval_loss fell smoothly 1.00 → 0.56; (b) generation quality **plateaued in a
noisy band from step ~1000 onward** while the denoising loss kept improving —
the same loss/generation decoupling the PAPL work found on ModernBERT.

Final checkpoint-4689, n=256 (deterministic + judge; predictions regenerated
locally with the mask patch + shifted-logits adapter re-applied — the raw
checkpoint loads as a causal model without them, see `src/dump_diffusion_qwen.py`):

- Exact match: 0.1211
- SQLite execution match: 0.1797
- SQLite executable/valid rate: 0.2188
- LLM semantic accuracy: **0.1914** (+0.070 over exact; judge validation 1.0)
- Average denoising steps: 11.8 (vs ~30 AR forward passes)

Failure modes from the semantic judge (share of 256):

| Category | Share |
|---|---:|
| syntax_error | 0.656 |
| missing_or_wrong_filter | 0.059 |
| hallucinated_identifier | 0.027 |
| wrong_aggregate_or_groupby | 0.027 |
| wrong_or_missing_column | 0.027 |
| truncated_or_empty | 0.004 |
| wrong_table | 0.004 |
| other | 0.004 |

v2 beats the confounded v1 checkpoint (semantic 0.191 vs 0.148) at **half the
denoise steps**, so the fixes were real — but the recipe-level ceiling did not
move much: the model still mostly fails by emitting invalid SQL.

## Comparison to the ModernBERT diffusion finetune (PAPL, 2026-06-30)

The [PAPL A/B](../2026-06-30_papl-ab/README.md) provides the third reference
point: ModernBERT-base (150M, natively bidirectional MLM), 10 epochs on the
same data, graded with the same judge on the same n=256 protocol. At its
conf_stop=0.9 operating point it averaged **11.79 denoise steps — essentially
decode-matched** to diffusion v2's 11.8:

| | Diffusion-ModernBERT | Diffusion-Qwen v2 | AR-Qwen |
|---|---:|---:|---:|
| Params | 150M | 500M | 500M |
| Epochs | 10 | 3 | 3 |
| Pretraining attention | bidirectional (MLM) | causal, adapted | causal (native) |
| Semantic accuracy | **0.500** | 0.191 | **0.645** |
| Exact match | 0.320 | 0.121 | 0.355 |
| syntax_error share | ~0.14 | 0.656 | (0.754 exec-valid) |
| Avg denoise steps | 11.79 | 11.8 | ~30 tokens |

Three conclusions follow:

1. **Native bidirectionality + convergence beat 3.3× parameters.** The 150M
   MLM-pretrained BERT more than doubles the adapted 500M Qwen's semantic
   accuracy at the same decode budget. The dominant v2 failure (65.6% invalid
   SQL vs BERT's ~14%) is exactly the *fluency-under-masking* competence that
   MLM pretraining provides natively and that 3 epochs of direct SQL
   adaptation could not instill in a causal model. A general-corpus diffusion
   adaptation phase (Dream/DiffuLLaMA-style, ~$1.5k+) is the known fix; it was
   deliberately skipped here, so v2 is a lower bound, not diffusion's ceiling.
2. **The quality ordering is AR (0.645) > BERT diffusion (0.500) > adapted
   diffusion (0.191)** — and even the best diffusion model trails AR by ~15
   points semantic at this scale.
3. **The loss/generation decoupling replicated across architectures.** Both
   diffusion models improve steadily on per-token denoising while end-to-end
   generation plateaus (BERT: PAPL "the ceiling is the model, not the
   decoding"; Qwen v2: plateau from step ~1000). This looks like a property of
   masked-diffusion SQL generation at these scales, not a bug in either setup.

Caveats: BERT trained 10 epochs vs 3 (but the PAPL pass@k probe showed those
extra epochs bought memorization/distribution collapse, not robustness);
BERT used the 128-token window vs v2's 64; BERT's validity metric is
parse-valid rather than SQLite exec-valid.

## Artifacts

| Path | Description |
|---|---|
| `data/ar_summary.json` | deterministic AR benchmark summary |
| `data/judge_summary.json` | semantic judge summary |
| `data/diffusion_checkpoint3126_summary.json` | pulled diffusion checkpoint deterministic benchmark summary |
| `data/diffusion_checkpoint3126_judge_summary.json` | pulled diffusion checkpoint semantic judge summary |
| `predictions/pred_ar_qwen25.jsonl` | AR predictions on 256 eval examples |
| `predictions/graded_ar_qwen25.jsonl` | LLM-judge graded AR predictions |
| `predictions/pred_diffusion_qwen25_checkpoint3126.jsonl` | pulled diffusion checkpoint predictions on 256 eval examples |
| `predictions/graded_diffusion_qwen25_checkpoint3126.jsonl` | LLM-judge graded pulled diffusion checkpoint predictions |
| `predictions/pred_diffusion_qwen25_final.jsonl` | diffusion v2 final checkpoint-4689 predictions (n=256, local regen) |
| `predictions/graded_diffusion_qwen25_final.jsonl` | LLM-judge graded v2 final predictions |
| `data/diffusion_final_judge_summary.json` | v2 final checkpoint judge summary + failure histogram |
| `logs/train_ar_*.log`, `logs/train_diffusion_*.log`, `logs/v2_launch.log` | full H100 training logs (AR, diffusion v1, diffusion v2) |
| `bench_local_latency.py` / `data/local_latency.json` | uniform Mac (MPS fp16 eager) latency benchmark of all four models |

Checkpoints (repo root): `sql-diffusion-qwen2.5-coder-0.5b/` (v2 final; must be
loaded with `make_bidirectional` + the shifted-logits adapter — see
`src/dump_diffusion_qwen.py` for the reference load path). W&B project:
`https://wandb.ai/tkoritzius/sql-diffusion` (AR `sfayjedq`, v1 `jodhkup5`, v2 `2l0tp3yi`).

## Interpretation

The AR baseline is strong enough that diffusion needs to be judged primarily on semantic accuracy at matched examples, not exact match. The pulled checkpoint is clearly not competitive: semantic accuracy is 0.148 versus AR's 0.645, execution accuracy is 0.164 versus AR's 0.562, and SQL validity is 0.254 versus AR's 0.754. Unlike AR, exact match is not heavily undercounting the pulled diffusion checkpoint; the semantic lift is only +0.047 because most wrong outputs are syntax errors or truncated/empty.

This should not be written as a clean proof that diffusion cannot work for text-to-SQL. The original pulled checkpoint has a known recipe confound around pad/EOS handling and fixed-window loss, and even the fixed v2 recipe is a *direct* SQL adaptation with no general-corpus diffusion phase — the step the Dream/DiffuLLaMA recipes consider essential. The defensible conclusions, with the completed v2 run and the judge on its final checkpoint:

1. **AR wins decisively at this scale**: semantic 0.645 vs 0.191 (3.4×), exec 0.562 vs 0.180 (3.1×). Diffusion's ~2.4× latency edge exists **only on server-class hardware** (H100: ~150–195 ms vs ~400 ms) and *inverts on local inference* (Mac MPS: AR 898 ms vs diffusion 1467 ms — see the uniform benchmark above). A hardware-conditional speed edge cannot compensate a 3× quality gap for short (~30-token) outputs on any target.
2. **The v2 fixes worked as engineering** (stable training, +0.043 semantic over v1 at half the denoise steps) **but did not move the recipe's ceiling**: 65.6% of failures are still invalid SQL, i.e. the adapted causal model never acquired BERT-grade fluency under masking.
3. **The cross-architecture comparison localizes the problem**: a 150M natively-bidirectional MLM at the same decode budget reaches 0.500 semantic — so the bottleneck is bidirectional/infilling pretraining (and convergence), not parameter count. Closing it would cost a general-corpus adaptation phase (~$1.5k+) with parity-at-best as the realistic outcome.
4. **Next quality levers are AR-side**: data realism (Spider/BIRD, de-templated synthetic) and execution-reward RL, per the PAPL failure analysis. Diffusion remains interesting only for latency-critical or long-output regimes where parallel decode actually compounds.
