# Experiments

Each dated folder is a self-contained write-up with the scripts, raw data, graded predictions and plots behind its conclusions.

| Folder | Question | Outcome |
|---|---|---|
| [`2026-10-09_fair-cpu-benchmark/`](2026-10-09_fair-cpu-benchmark/README.md) | Same-engine CPU comparison of diffusion vs. AR, plus cheap inference wins. | Cheap changes make diffusion 2.1× faster on CPU (863 → 415 ms). Schema verification adds +6.5 points exec. On CPU a 3.3× larger AR model still matches diffusion fp32 latency at +14 points accuracy. |
| [`2026-06-30_papl-ab/`](2026-06-30_papl-ab/README.md) | Does PAPL (planner-aware, confidence-reweighted) fine-tuning improve the ModernBERT diffusion model? | Only speed: ~7% fewer steps under adaptive early stop, same accuracy. Real quality is ~0.50 semantic (exact match undercounts it). Invalid SQL is the biggest failure mode. |
| [`2025-06_v0-from-scratch/`](2025-06_v0-from-scratch/) | First prototype: a small transformer trained from scratch with the GPT-2 tokenizer. | Superseded. Too small to generate useful text, which led to fine-tuning pretrained MLMs (RoBERTa, then ModernBERT). |

Decoding-strategy comparisons (confidence vs. dependency ordering, fixed vs. adaptive steps) are documented in the main [README](../README.md#decoding-efficiency-steps-vs-autoregressive).
