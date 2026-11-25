# Copy-acceptance audit results

Target Qwen2.5-Coder-1.5B-Instruct, greedy FP16, MPS, batch one. 258 edits kept; 3 dropped because the target hit the generation cap (repetition).

## Verification cost per target call (median ms, cached context)

| Width | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 33.8 | 38.0 | 39.0 | 39.7 | 53.4 | 59.3 | 62.0 | 71.5 | 75.9 | 93.2 |

## function (CanItEdit): n=36, median output 106 tokens, median edited-token share 14%

| Policy | Tokens per target call | 95% CI | Call speedup (median item) | Share of calls on edited spans | Modeled Mac latency speedup |
|---|---:|---:|---:|---:|---:|
| Greedy AR | 1.00 | 1.00–1.00 | 1.00× | 23% | 1.00× |
| Prompt lookup (3-gram, K=32) | 4.46 | 3.22–6.90 | 5.38× | 66% | 2.59× |
| Suffix-match copy (K=32) | 4.50 | 3.21–7.21 | 5.42× | 66% | 2.72× |
| Suffix-match copy (K=64) | 4.67 | 3.26–7.81 | 5.84× | 68% | 2.56× |
| Copy ceiling: perfect resync (K=32) | 5.36 | 3.77–8.41 | 6.22× | 78% | 3.21× |
| Copy + perfect 4-token filler on edited spans | 11.68 | 9.10–15.51 | 13.00× | 53% | 6.29× |
| Copy + perfect 8-token filler | 14.66 | 12.08–18.33 | 15.71× | 42% | 7.18× |
| Copy + perfect 16-token filler | 18.56 | 15.72–22.08 | 19.06× | 35% | 7.94× |
| Copy + perfect 32-token filler | 23.73 | 20.99–26.88 | 24.62× | 30% | 9.13× |
| Perfect drafter, K=32 everywhere | 30.06 | 28.92–30.86 | 30.58× | 27% | 10.80× |

Measured AR wall time / modeled AR time: 0.99. Latency includes prefill; speedups are ratios of total time over the group.

## function (EditPackFT): n=119, median output 116 tokens, median edited-token share 8%

| Policy | Tokens per target call | 95% CI | Call speedup (median item) | Share of calls on edited spans | Modeled Mac latency speedup |
|---|---:|---:|---:|---:|---:|
| Greedy AR | 1.00 | 1.00–1.00 | 1.00× | 15% | 1.00× |
| Prompt lookup (3-gram, K=32) | 6.72 | 5.65–8.10 | 11.33× | 69% | 3.88× |
| Suffix-match copy (K=32) | 6.93 | 5.84–8.39 | 11.50× | 68% | 4.14× |
| Suffix-match copy (K=64) | 7.55 | 6.27–9.34 | 12.55× | 74% | 4.15× |
| Copy ceiling: perfect resync (K=32) | 7.54 | 6.32–9.24 | 12.40× | 74% | 4.47× |
| Copy + perfect 4-token filler on edited spans | 15.35 | 13.81–17.08 | 17.38× | 48% | 7.80× |
| Copy + perfect 8-token filler | 18.82 | 17.52–20.26 | 19.71× | 39% | 8.73× |
| Copy + perfect 16-token filler | 22.77 | 21.68–23.85 | 22.67× | 30% | 9.37× |
| Copy + perfect 32-token filler | 26.77 | 25.98–27.50 | 26.33× | 26% | 10.23× |
| Perfect drafter, K=32 everywhere | 29.21 | 28.76–29.64 | 29.17× | 17% | 11.01× |

Measured AR wall time / modeled AR time: 0.94. Latency includes prefill; speedups are ratios of total time over the group.

## full file (CanItEdit): n=103, median output 274 tokens, median edited-token share 11%

| Policy | Tokens per target call | 95% CI | Call speedup (median item) | Share of calls on edited spans | Modeled Mac latency speedup |
|---|---:|---:|---:|---:|---:|
| Greedy AR | 1.00 | 1.00–1.00 | 1.00× | 12% | 1.00× |
| Prompt lookup (3-gram, K=32) | 9.00 | 7.85–10.34 | 9.07× | 59% | 4.74× |
| Suffix-match copy (K=32) | 9.73 | 8.40–11.33 | 10.17× | 61% | 5.25× |
| Suffix-match copy (K=64) | 11.06 | 9.17–13.21 | 11.67× | 69% | 5.33× |
| Copy ceiling: perfect resync (K=32) | 10.61 | 9.15–12.37 | 10.62× | 67% | 5.65× |
| Copy + perfect 4-token filler on edited spans | 18.85 | 17.32–20.40 | 18.67× | 41% | 9.15× |
| Copy + perfect 8-token filler | 22.31 | 20.93–23.64 | 22.13× | 32% | 10.05× |
| Copy + perfect 16-token filler | 25.99 | 24.88–26.96 | 25.19× | 24% | 10.66× |
| Copy + perfect 32-token filler | 29.21 | 28.51–29.80 | 28.67× | 18% | 11.33× |
| Perfect drafter, K=32 everywhere | 31.48 | 31.20–31.71 | 31.14× | 12% | 12.00× |

Measured AR wall time / modeled AR time: 1.22. Latency includes prefill; speedups are ratios of total time over the group.

## Actual edits only (target output differs from the old code)

| Group | n | Suffix-match copy | Perfect resync | + perfect 8-token filler | + perfect 16-token filler | Perfect drafter |
|---|---:|---:|---:|---:|---:|---:|
| function (CanItEdit) | 30 | 3.7 tok/call, 2.3× Mac | 4.4 tok/call, 2.7× Mac | 12.9 tok/call, 6.6× Mac | 16.8 tok/call, 7.4× Mac | 30.1 tok/call, 10.8× Mac |
| function (EditPackFT) | 90 | 5.5 tok/call, 3.5× Mac | 6.1 tok/call, 3.8× Mac | 16.8 tok/call, 8.2× Mac | 21.1 tok/call, 9.0× Mac | 29.0 tok/call, 11.1× Mac |
| full file (CanItEdit) | 84 | 8.2 tok/call, 4.5× Mac | 9.0 tok/call, 4.9× Mac | 20.6 tok/call, 9.6× Mac | 24.8 tok/call, 10.3× Mac | 31.4 tok/call, 12.0× Mac |

54 of 258 target outputs reproduced the old code unchanged.

## Real drafter: Qwen2.5-Coder-0.5B-Instruct (16-token drafts)

Drafter forward cost: 1 token 19.2 ms, 16 tokens 29.0 ms. AR drafting charges 16 sequential steps; the one-pass projection charges one 16-token forward at the same acceptance (an assumption about a future diffusion drafter, not a measurement).

| Group | Policy | Tokens per target call | Accepted on edited-span cycles | Mac model, AR drafting | Mac model, one-pass drafting | Flat verify, one pass = 5% of target step | = 20% |
|---|---|---:|---:|---:|---:|---:|---:|
| function (n=155) | ar_draft | 12.54 | 6.66 | 1.08× | 3.46× | 11.9× | 10.4× |
| function (n=155) | hybrid | 16.79 | 6.75 | 1.97× | 4.92× | 16.3× | 14.9× |
| function | suffix32 (reference) | 6.09 | – | 3.65× | – | 6.1× | 6.1× |
| full file (n=49) | ar_draft | 13.63 | 7.52 | 1.17× | 3.81× | 13.0× | 11.4× |
| full file (n=49) | hybrid | 20.94 | 8.17 | 2.95× | 6.56× | 20.4× | 19.1× |
| full file | suffix32 (reference) | 8.05 | – | 4.56× | – | 8.1× | 8.1× |

## Right-context ablation on edited spans (Qwen2.5-Coder-0.5B base, FIM, function regime)

2292 edited-span cycle starts in 111 edits. Accepted tokens per draft (max 16).

| Suffix given to the drafter | Mean accepted | 95% CI (by edit) | ≥4 accepted | Better than no suffix | Worse than no suffix |
|---|---:|---:|---:|---:|---:|
| None | 5.39 | 4.65–6.08 | 44% | 0% | 0% |
| Remaining old source (deployable) | 4.61 | 3.88–5.31 | 39% | 16% | 21% |
| Old source from the true next anchor (oracle) | 4.33 | 3.55–5.02 | 38% | 14% | 22% |
