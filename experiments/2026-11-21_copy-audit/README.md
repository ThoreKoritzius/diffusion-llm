# Copy-acceptance audit for lossless speculative code editing (M4 Pro, 2026-11-21)

Question: in code editing, where do target calls go under copy speculation, and how much room is left for a
(diffusion) drafter on the edited spans? Everything here is lossless greedy speculative decoding replayed exactly
against recorded target outputs, so all policies emit identical text and differ only in target calls and cost.

Setup: target Qwen2.5-Coder-1.5B-Instruct, FP16, MPS, batch one, prompt prefilled with a Python code fence.
Cohort: 258 edits (36 CanItEdit single-function, 119 EditPackFT single-function with the commit subject as
instruction, 103 CanItEdit full-file rewrites); 3 outputs that hit the length cap were dropped. Generated numbers
are in [results.md](results.md). Acceptance counts are hardware-independent; the latency columns use the measured
per-width verification cost on this Mac (width 1: 33.8 ms, 32: 62.0 ms, 64: 75.9 ms).

```sh
python3 prepare.py && python3 generate.py && python3 cost.py && python3 sim.py
python3 draft.py A && python3 draft.py B && python3 analyze.py
```

## Findings

1. **The calls are spent on the edited spans.** Tokens not copied from the source are 8–14% of the output
   (median per group) but take 59–78% of target calls under copy speculation.
2. **Resync is not the bottleneck.** Knowing exactly where to resume copying adds only 9–19% over plain
   suffix matching (for example 6.9 → 7.5 tokens per call on EditPackFT).
3. **Edited content arrives in blocks.** 68–80% of edited tokens are in spans longer than 16 tokens, so the filler
   must generate real code, and multi-token block drafting is where the remaining gain is.
4. **Copy alone is far from 10× on actual edits.** Restricted to outputs that differ from the old code, suffix
   copy gives 3.7 / 5.5 / 8.2 tokens per call (CanItEdit function / EditPackFT / full file), i.e. 2.3–4.5×
   modeled Mac latency. A perfect 16-token filler on edited spans would give 16.8 / 21.1 / 24.8 tokens per call.
5. **A small same-family drafter recovers most of that, but sequential AR drafting is too expensive.**
   Copy + Qwen2.5-Coder-0.5B-Instruct (two-branch tree, 16-token drafts) reaches 16.8 tokens per call on functions and
   20.9 on full files, versus 6.1 and 8.1 for copy alone. The drafter accepts about 7 tokens per draft on
   edited spans. On this Mac a drafter step costs 57% of a target step, so AR drafting nets only 2.0–3.0×. The
   one-pass columns *assume* a parallel drafter keeps the AR drafter's acceptance; that is the open question.
6. **Right context does not help an untrained infill drafter.** On 2,292 edited-span positions, Qwen2.5-Coder-0.5B
   base in FIM mode accepted 5.39 tokens with no suffix, 4.61 with the remaining old source as suffix and 4.33 with
   the true next anchor (middle + resumed suffix scored). Right context helps where left-only drafting fails
   (0 → 1.1 and 1.8 → 2.6 tokens) and hurts where it already succeeds (16 → 11). Choosing the better of the two
   per position would give 6.58 (+22%).

## Limits

- One small target. 54/258 of its outputs reproduce the old code unchanged, which inflates copy speedups; see the
  actual-edits table. A 7–8B target will make more real edits and shift more cost onto edited spans.
- Mac verification cost grows with width; an H100 with an 8B target is closer to flat up to tens of tokens. The
  flat-verify columns are projections, not measurements, and exclude prefill.
- The FIM ablation uses an off-the-shelf model on an unusual prompt (old code as comments). It shows that right
  context is not free; it does not show that a drafter trained for anchored infilling cannot use it.
- Drafting acceptance stops at the first mismatch; drafting cost is charged for the full 16 tokens.

## Projection to H100: the decision curve (synth.py)

Synthetic drafter: each of 15 drafted tokens is accepted with probability p, combined with suffix copy as a two-branch
tree, replayed exactly on the recorded outputs. Validation: at the measured 0.5B drafter's mean acceptance on edited
spans (6.75), the synthetic hybrid gives 16.9 tokens per call versus 16.8 measured on functions. Cost per cycle in AR
steps is calibrated on DFlash Table 3 (Qwen3-8B, B200, SGLang: one draft+verify cycle = 1.55 AR steps). Copy-only cycles
are assumed to cost 1.25, and prefill is assumed to cost 4 AR steps per 1,000 prompt tokens.

| Drafter tokens accepted per guess on new code | Functions: tok/call → speedup (range) | Full files: tok/call → speedup (range) |
|---:|---|---|
| copy only | 6.1 → 4.8× (4.4–5.2) | 9.7 → 7.5× (7.0–8.1) |
| 2.3 | 10.9 → 7.1× (6.3–7.8) | 15.4 → 10.2× (9.1–11.2) |
| 3.9 | 13.2 → 8.6× (7.6–9.5) | 17.8 → 11.9× (10.6–13.0) |
| 5.2 | 14.8 → 9.7× (8.6–10.6) | 19.4 → 13.0× (11.7–14.2) |
| 6.3 | 15.8 → 10.3× (9.2–11.3) | 20.5 → 13.8× (12.4–15.0) |
| 7.1 | 16.9 → 11.0× (9.8–12.1) | 21.4 → 14.4× (12.9–15.6) |

Reading: 10× on single functions needs a one-pass drafter that gets about 5–6 tokens accepted per guess on new code.
DFlash's published code result (Qwen3-8B, HumanEval, from-scratch generation, tau 6.50) corresponds to about 5.5
accepted tokens per cycle. If new code in edits behaves like that, DFlash + copy lands right at the 10× line.
Measuring DFlash's acceptance on edited spans therefore decides the plan.
