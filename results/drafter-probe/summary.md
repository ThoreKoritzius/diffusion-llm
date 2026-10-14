| set | config | tokens / pass | passes | mean s | speedup | same as Qwen |
|---|---|---:|---:|---:|---:|---:|
| vague | qwen | 1.00 | 32.4 | 0.95 | 1.00x | 50/50 |
| vague | dflash_k1 | 3.34 | 9.7 | 1.01 | 0.94x | 49/50 |
| vague | dflash_k2 | 3.67 | 8.8 | 0.88 | 1.07x | 49/50 |
| vague | dflash_k3 | 3.69 | 8.8 | 1.02 | 0.93x | 49/50 |
| test | qwen | 1.00 | 31.4 | 0.98 | 1.00x | 50/50 |
| test | dflash_k1 | 5.62 | 5.6 | 0.64 | 1.54x | 50/50 |
| test | dflash_k2 | 5.81 | 5.4 | 0.63 | 1.56x | 50/50 |
| test | dflash_k3 | 5.85 | 5.4 | 0.69 | 1.42x | 50/50 |
