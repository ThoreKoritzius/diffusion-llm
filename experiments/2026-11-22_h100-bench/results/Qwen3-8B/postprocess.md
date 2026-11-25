# Post-processing: Qwen3-8B

## 1. Measured in the PyTorch harness: DFlash total time / hybrid total time (same requests, prefill included)

| Hybrid | Scope | n | Speedup over DFlash | 95% bootstrap CI |
|---|---|---:|---:|---|
| hybrid_router4 | all | 261 | 1.31× | 1.27–1.36 |
| hybrid_router4 | single function | 157 | 1.28× | 1.22–1.34 |
| hybrid_router4 | full file | 104 | 1.34× | 1.27–1.41 |
| hybrid_router8 | all | 261 | 1.30× | 1.26–1.34 |
| hybrid_router8 | single function | 157 | 1.25× | 1.20–1.31 |
| hybrid_router8 | full file | 104 | 1.33× | 1.27–1.39 |
| hybrid_router16 | all | 261 | 1.27× | 1.23–1.31 |
| hybrid_router16 | single function | 157 | 1.22× | 1.18–1.27 |
| hybrid_router16 | full file | 104 | 1.30× | 1.25–1.36 |
| hybrid_oracle (extra verify calls, not deployable timing) | all | 261 | 0.99× | 0.95–1.03 |
| hybrid_oracle (extra verify calls, not deployable timing) | single function | 157 | 0.98× | 0.93–1.04 |
| hybrid_oracle (extra verify calls, not deployable timing) | full file | 104 | 1.00× | 0.94–1.06 |

## 2. Drafter resume cost (drafted cycles: draft + 16-token verify)

| Source | Queued context at draft | n cycles | Median cycle ms |
|---|---|---:|---:|
| harness_dflash | 1–16 | 6204 | 49.9 |
| harness_dflash | linear fit | 6204 | -1.05 ms per 100 queued states |
| hybrid_router8 | 1–16 | 2791 | 50.9 |
| hybrid_router8 | 17–48 | 242 | 51.0 |
| hybrid_router8 | 49–128 | 127 | 51.3 |
| hybrid_router8 | 129–∞ | 62 | 50.9 |
| hybrid_router8 | linear fit | 3222 | -0.01 ms per 100 queued states |

## 3. vLLM output vs plain AR

| Config | n | Identical | Stop/length only | Mid-sequence mismatch | Median first-mismatch position |
|---|---:|---:|---:|---:|---:|
| ar_repeat | 20 | 20 | 0 | 0 | – |
| dflash15 | 261 | 231 | 0 | 30 | 191 |
| draft06b7 | 261 | 231 | 0 | 30 | 191 |
| draft17b7 | 261 | 231 | 0 | 30 | 191 |
| eagle3_3 | 261 | 228 | 0 | 33 | 172 |
| eagle3_7 | 261 | 231 | 0 | 30 | 191 |
| ngram15 | 261 | 231 | 0 | 30 | 191 |
| ngram31 | 261 | 227 | 0 | 34 | 195 |
| ngram7 | 261 | 231 | 0 | 30 | 191 |

20 mid-sequence cases written to diverge_ids.json for the GPU diagnosis.
