# Generalization run: summary

## 1. Measured end-to-end speedup over plain AR (vLLM, batch one, greedy)

| Target | Cohort | ngram31 | eagle7 | draft7 | dflash | Copy + DFlash projected | Router vs DFlash, measured (95% CI) | DFlash accepted/guess on new code |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-8b | edit261 | 4.42× | 3.13× | 2.43× | 6.76× | 8.92× | 1.15× (1.08–1.22) | 3.45 |
| qwen3-8b | ceb150 | 2.52× | 3.28× | 2.56× | 7.48× | 7.96× | 0.98× (0.94–1.02) | 6.95 |
| qwen3-8b | diff104 | 3.65× | 2.77× | 2.34× | 5.48× | 6.34× | 1.03× (0.98–1.09) | 2.83 |
| qwen3-4b | edit261 | – | – | – | – | – | – | – |

## 2. Cycle costs (AR steps) and agreement with plain AR

| Target | Cohort | ngram31 | eagle7 | draft7 | dflash | Output identical to AR (dflash) |
|---|---|---:|---:|---:|---:|---:|
| qwen3-8b | edit261 | 1.05 | 1.35 | 2.62 | 1.25 | 87% |
| qwen3-8b | ceb150 | 1.05 | 1.35 | 2.63 | 1.25 | 88% |
| qwen3-8b | diff104 | 1.05 | 1.35 | 2.64 | 1.25 | 80% |
| qwen3-4b | edit261 | – | – | – | – | – |

## 3. Pre-registered predictions vs measured

Predictions created 2026-11-22T11:05:45Z (sha256 f29757a0d707…), before any decoding of these cohorts. Speedup predictions use A100 cost constants; the GPU here may differ, so tokens per call is the hardware-independent test.

| Target | Cohort | Kind | Method | Tokens/call predicted | measured | error | Speedup predicted (A100 costs) | measured |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3-8b | edit261 | same target | ngram31 | 5.1 | 4.9 | +5% | 4.67× | 4.42× |
| qwen3-8b | edit261 | same target | draft7 | 6.9 | 6.7 | +3% | 3.06× | 2.43× |
| qwen3-8b | edit261 | same target | eagle7 | 4.5 | 4.4 | +2% | 3.28× | 3.13× |
| qwen3-8b | edit261 | same target | dflash | 9.5 | 9.2 | +4% | 7.16× | 6.76× |
| qwen3-8b | ceb150 | same target | ngram31 | 5.6 | 2.7 | +107% | 5.11× | 2.52× |
| qwen3-8b | ceb150 | same target | draft7 | 6.9 | 7.1 | -2% | 3.08× | 2.56× |
| qwen3-8b | ceb150 | same target | eagle7 | 4.5 | 4.6 | -2% | 3.29× | 3.28× |
| qwen3-8b | ceb150 | same target | dflash | 9.7 | 10.2 | -4% | 7.31× | 7.48× |
| qwen3-8b | diff104 | same target | ngram31 | 2.5 | 4.1 | -38% | 2.37× | 3.65× |
| qwen3-8b | diff104 | same target | draft7 | 6.3 | 6.5 | -3% | 2.75× | 2.34× |
| qwen3-8b | diff104 | same target | eagle7 | 4.0 | 4.0 | +2% | 2.92× | 2.77× |
| qwen3-8b | diff104 | same target | dflash | 7.2 | 7.6 | -4% | 5.33× | 5.48× |

Median absolute tokens-per-call error, same target: 4% over 12 predictions.
