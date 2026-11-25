# Performance law: Qwen3-8B, 261 edits

cycles = C/τ_copy + N/τ_new + δ·R (non-negative fit); latency = prefill(P) + s·c_m·cycles. Errors are on held-out edits only; speedups are aggregate (sum of AR latency / sum of method latency).

## 1a. Random 50/50 splits (20 seeds; medians over seeds)

| Method | τ_copy | τ_new | δ per copy run | Per-edit cycle error | Per-edit latency error | Speedup predicted / measured |
|---|---:|---:|---:|---:|---:|---|
| ngram7 | 7.2 | 1.5 | 1.57 | 11% | 10% | 3.53× / 3.45× |
| ngram15 | 14.0 | 1.5 | 1.80 | 14% | 13% | 4.24× / 4.15× |
| ngram31 | 31.8 | 1.6 | 2.25 | 16% | 15% | 4.60× / 4.53× |
| eagle3_3 | 3.3 | 2.7 | 0.21 | 6% | 6% | 2.71× / 2.69× |
| eagle3_7 | 5.0 | 3.2 | 0.26 | 10% | 9% | 3.22× / 3.18× |
| draft06b7 | 7.8 | 5.0 | 0.22 | 6% | 7% | 3.02× / 2.84× |
| draft17b7 | 7.9 | 5.7 | 0.16 | 4% | 6% | 2.28× / 2.13× |
| dflash15 | 13.1 | 5.3 | 0.36 | 15% | 13% | 7.11× / 6.95× |

## 1b. Group transfer (fit on one edit type, predict the other)

| Method | functions → files: speedup pred / true (latency err) | files → functions: speedup pred / true (latency err) |
|---|---|---|
| ngram7 | 4.02× / 3.43× (14%) | 3.33× / 3.53× (7%) |
| ngram15 | 5.04× / 4.14× (19%) | 3.93× / 4.23× (11%) |
| ngram31 | 5.55× / 4.53× (20%) | 4.22× / 4.53× (13%) |
| eagle3_3 | 2.73× / 2.72× (6%) | 2.69× / 2.64× (6%) |
| eagle3_7 | 3.33× / 3.24× (10%) | 3.18× / 3.09× (9%) |
| draft06b7 | 3.05× / 2.88× (5%) | 2.91× / 2.78× (6%) |
| draft17b7 | 2.32× / 2.15× (7%) | 2.18× / 2.10× (4%) |
| dflash15 | 7.17× / 7.36× (14%) | 6.81× / 6.35× (12%) |

## 2. Composition: predict copy + DFlash (router-8) from its components

τ_copy and δ from prompt lookup (ngram31), τ_new from DFlash (dflash15), both fitted in vLLM on all edits; compared with the hybrid's actual cycles in the harness (no hybrid data used in the fit).

| Scope | n | Tokens per call predicted | Actual | Per-edit cycle error | DFlash-alone actual tokens per call |
|---|---:|---:|---:|---:|---:|
| all | 261 | 10.1 | 11.9 | 27% | 9.3 |
| functions | 157 | 8.8 | 10.7 | 27% | 8.6 |
| full files | 104 | 10.9 | 12.8 | 27% | 9.8 |

## 3. Before decoding: structure from the reference diff instead of the model output (random splits)

| Method | Per-edit latency error (output structure) | Per-edit latency error (reference structure) | Speedup pred / true (reference structure) |
|---|---:|---:|---|
| ngram7 | 10% | 30% | 3.46× / 3.46× |
| ngram15 | 13% | 37% | 4.16× / 4.16× |
| ngram31 | 15% | 41% | 4.55× / 4.56× |
| eagle3_3 | 6% | 13% | 2.71× / 2.69× |
| eagle3_7 | 9% | 17% | 3.22× / 3.19× |
| draft06b7 | 7% | 15% | 2.99× / 2.83× |
| draft17b7 | 6% | 13% | 2.27× / 2.12× |
| dflash15 | 13% | 22% | 7.03× / 6.95× |

Correlation between new-code share in the model output and in the reference diff: 0.67.

## 2b. Composition as bounds (post-hoc, to be validated on another target)

| Rule | Functions: tokens per call | Full files |
|---|---:|---:|
| Lower: copy C rate and copy boundary cost δ (ngram31) + DFlash N rate | 8.8 | 10.9 |
| **Measured router-8 hybrid (harness)** | **10.7** | **12.8** |
| Upper: best component per term (copy C rate, DFlash N rate, DFlash δ) | 14.1 | 14.3 |

The deployable router realises 36% (functions) and 56% (full files) of the gap between the two bounds. The upper
bound is the headroom available to a better router or a drafter that handles copy boundaries.
