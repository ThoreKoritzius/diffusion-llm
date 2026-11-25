# DFlash on code edits, target Qwen3-4B (MLX, bf16, greedy, M4 Pro)

256 completed edits; 27 outputs reproduce the old code unchanged.

## Tokens per target call and DFlash acceptance

| Group | n | New-code share | Copy only | DFlash alone | DFlash accepted/guess on new code | Copy + DFlash (measured) | Drafter accepted/guess when needed | H100 projection: copy / DFlash / hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full file (all) | 99 | 19% | 6.0 | 10.1 | 4.67 (4.45–4.91) | 15.1 | 5.44 | 4.7× / 6.3× / 10.0× |
| function (CanItEdit) (all) | 37 | 20% | 4.5 | 8.6 | 4.29 (3.80–4.80) | 11.7 | 5.76 | 3.6× / 5.4× / 7.6× |
| function (EditPackFT) (all) | 120 | 9% | 7.3 | 8.8 | 2.98 (2.68–3.31) | 13.6 | 4.07 | 5.7× / 5.5× / 9.0× |
| full file (actual edits) | 96 | 20% | 5.8 | 9.9 | 4.67 (4.46–4.90) | 14.7 | 5.44 | 4.5× / 6.3× / 9.7× |
| function (CanItEdit) (actual edits) | 37 | 20% | 4.5 | 8.6 | 4.29 (3.80–4.80) | 11.7 | 5.76 | 3.6× / 5.4× / 7.6× |
| function (EditPackFT) (actual edits) | 96 | 13% | 6.2 | 8.4 | 2.98 (2.68–3.30) | 12.1 | 4.03 | 4.9× / 5.3× / 8.0× |

## Mac wall clock (M4 Pro, batch one, includes prefill)

32 edits timed with plain AR. DFlash speedup 2.52×.
Hybrid on 32 of them: 3.48× (verifies only the chosen branch, so slightly optimistic versus a real two-branch tree).
AR and DFlash outputs agree on the common prefix in 29/32 edits (bf16 numerics).
Median per cycle: drafter 20.4 ms, 16-token verify 103.7 ms; AR step 36.6 ms.

H100 projection uses 1.55 AR steps per drafted cycle (DFlash paper, B200 SGLang), 1.25 per copy-only cycle, and 4 AR steps of prefill per 1,000 prompt tokens. It is a model, not a measurement.
