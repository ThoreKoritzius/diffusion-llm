# DFlash on real code edits (Qwen3-4B, MLX, M4 Pro, 2026-11-21)

Question: how many tokens per guess does a released one-pass diffusion drafter (DFlash) get accepted on the *new*
code inside edits? The copy-audit decision curve says 10× on H100 for single functions needs about 5–6 accepted
tokens per guess on new code ([../2026-11-21_copy-audit/README.md](../2026-11-21_copy-audit/README.md)).

Setup: target `Qwen/Qwen3-4B` (bf16, thinking disabled), drafter `z-lab/Qwen3-4B-DFlash-b16` (5 layers, block 16),
greedy, same 261-edit cohort and prompt as the copy audit. `run.py` reimplements the DFlash MLX loop with logging;
it reproduces the stock `stream_generate` output token for token. Results table: [results_Qwen3-4B.md](results_Qwen3-4B.md).

```sh
python3 -m venv .venv && .venv/bin/pip install "dflash[local]" numpy transformers
.venv/bin/python run.py dflash && .venv/bin/python run.py hybrid && .venv/bin/python run.py ar
.venv/bin/python analyze.py
```

## Findings

| Group (actual edits) | DFlash accepted/guess on new code | Copy only | DFlash alone | Copy + DFlash | H100 projection copy / DFlash / hybrid |
|---|---:|---:|---:|---:|---|
| Full file (n=96) | 4.67 | 5.8 | 9.9 | 14.7 | 4.5× / 6.3× / 9.7× |
| Function, CanItEdit (n=37) | 4.29 | 4.5 | 8.6 | 11.7 | 3.6× / 5.4× / 7.6× |
| Function, EditPackFT (n=96) | 2.98 | 6.2 | 8.4 | 12.1 | 4.9× / 5.3× / 8.0× |

(Columns 3–5: tokens per target call.)

1. **DFlash accepts 3.0–4.7 tokens per guess on new code, below the 5–6 needed.** The prior from DFlash's
   from-scratch HumanEval result (about 5.5) was too optimistic for edits. EditPackFT, with commit subjects as
   instructions, is hardest.
2. **The obvious combination gets close but not there.** Copy + DFlash projects to 7.6–9.7× on H100, and copy
   adds 35–55% over DFlash alone. Single functions stay under 10×.
3. **The gap to 10× is concrete and trainable.** New-code acceptance needs to rise by about 1.5–2.5 tokens per guess
   (+40–60%). The released drafter was trained on Nemotron and CodeAlpaca responses, not on edits.
4. **A bigger target edits more.** Qwen3-4B reproduced the old code unchanged in 27/256 cases (Qwen2.5-Coder-1.5B:
   54/258), and new code is 13–20% of actual edits (1.5B: 8–14%). More work lands on the drafter.
5. **The Mac is a poor platform for this.** Measured wall clock: DFlash 2.5×, hybrid 3.5× over plain AR. A 16-token
   verify costs 104 ms versus 37 ms for one AR step, so verification is far from free here. Acceptance counts
   transfer to the GPU; these timings do not.

## Limits

- H100 numbers are projections from a cost model calibrated on the DFlash paper's B200/SGLang result, not measurements.
- Hybrid acceptance is evaluated by choosing the branch the target accepts furthest, as two-branch tree verification
  does; tree cost is charged in the projection, not in the Mac timing. 26/261 hybrid runs diverged from the DFlash
  reference through bf16 numerics and are excluded from hybrid figures.
- bf16 batched verification differs from one-token decoding at near-ties: DFlash and plain AR outputs agree on the
  common prefix in 29/32 timed edits. Same in any bf16 speculative decoder.
- One target size (4B) and greedy decoding only.

## Matched comparison: one-pass DFlash vs small AR drafters (EfficientEdit-style), 2026-11-22

`ar_drafter.py` replays the same Qwen3-4B greedy outputs with Qwen3-0.6B / 1.7B as AR drafters (γ = 7 as in
EfficientEdit, and γ = 15 to match DFlash's block), alone and combined with copy. `compare.py` puts all
methods on the same 235 edits and the same H100 cost model. Full table: [compare_Qwen3-4B.md](compare_Qwen3-4B.md).

| Functions (n=147) / full files (n=88) | Accepted per draft on new code | H100 projection |
|---|---:|---|
| Copy only | – | 5.0× / 4.9× |
| Copy + DFlash (0.5B, one pass) | 3.4 / 4.8 | **8.6× / 9.8×** |
| Copy + Qwen3-0.6B, γ=7 | 3.3 / 4.1 | 3.9–6.6× / 4.2–7.1× |
| Copy + Qwen3-1.7B, γ=15 | 6.1 / 7.8 | 2.4–3.1× / 2.8–3.6× |
| Best AR hybrid with oracle early stop, cheapest cost | – | 7.4× / 8.1× |

1. **At equal cost the one-pass drafter wins in every scenario.** Even granting the AR drafter an oracle stop at its
   first wrong token and a drafter step at 15% of a target step, copy + DFlash stays 16–21% ahead. At mid cost it is
   about 45–60% ahead.
2. **AR drafters guess new code better, but pay for it sequentially.** At γ = 15, Qwen3-1.7B accepts 6.1–7.8 tokens per
   draft on new code versus DFlash's 3.4–4.8. The one-pass drafter loses about 40% of acceptance and wins on cost.
3. **That gap is the opportunity.** A one-pass drafter that reached the 1.7B AR drafter's acceptance on new code at
   DFlash's cost would land around 11–14× on the copy-audit decision curve.

Limits: AR drafter step cost on H100 (r) is an assumed range, not measured; EfficientEdit itself uses a 7B drafter
LoRA-tuned on edits with a 32B target, which this does not reproduce (its tuning added 5–8% in its own ablation).
EAGLE-style feature drafters are not tested. Memory stayed at ≥55% free during the run (watchdog.log).
