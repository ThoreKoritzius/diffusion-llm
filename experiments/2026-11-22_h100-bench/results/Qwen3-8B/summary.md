# GPU benchmark summary: Qwen3-8B (GPU recorded in env.txt below)

```
Sun Nov 22 07:59:32 UTC 2026
name, memory.total [MiB], driver_version, clocks.max.sm [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 40960 MiB, 580.105.08, 1410 MHz, 400.00 W
	[4mGPU0	NIC0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	0-29	0		N/A
NIC0	PHB	 X 				
vllm 0.34.0rc1.dev1134+ge72fc9ae5 | torch 2.14.0+cu132 | cuda 13.2
harness torch 2.13.0+cu130 | transformers 5.15.0
```

## 1. Measured in vLLM (batch one, greedy)

| Config | n | Speedup vs AR (end to end) | Decode-only speedup | Mean accepted length τ | Cycle cost (AR steps) | Output identical to AR |
|---|---:|---:|---:|---:|---:|---:|
| ar | 261 | 1.00× | 1.00× | 1.00 | 1.00 | 100% |
| ar_repeat | 20 | 1.00× | 1.00× | 1.00 | 1.00 | 100% |
| dflash15 | 261 | 6.97× | 7.51× | 9.29 | 1.21 | 89% |
| draft06b7 | 261 | 2.84× | 2.99× | 6.75 | 2.19 | 89% |
| draft17b7 | 261 | 2.13× | 2.24× | 7.10 | 3.06 | 89% |
| eagle3_3 | 261 | 2.69× | 2.75× | 3.16 | 1.13 | 87% |
| eagle3_7 | 261 | 3.18× | 3.28× | 4.40 | 1.31 | 89% |
| ngram15 | 261 | 4.17× | 4.33× | 4.37 | 0.99 | 89% |
| ngram31 | 261 | 4.53× | 4.73× | 4.94 | 1.03 | 87% |
| ngram7 | 261 | 3.46× | 3.57× | 3.58 | 0.98 | 89% |

Per group, end-to-end speedup vs AR:

| Config | function (CanItEdit) | function (EditPackFT) | full file |
|---|---:|---:|---:|
| ar | 1.00× | 1.00× | 1.00× |
| ar_repeat | 1.00× | – | 1.00× |
| dflash15 | 6.82× | 6.19× | 7.36× |
| draft06b7 | 2.76× | 2.79× | 2.88× |
| draft17b7 | 2.08× | 2.10× | 2.15× |
| eagle3_3 | 2.71× | 2.62× | 2.72× |
| eagle3_7 | 3.19× | 3.06× | 3.24× |
| ngram15 | 3.40× | 4.64× | 4.14× |
| ngram31 | 3.48× | 5.10× | 4.53× |
| ngram7 | 3.00× | 3.77× | 3.43× |

## 2. Calibrated costs (median per edit, in plain AR decode steps)

- ar: 1.00 per cycle
- ar_repeat: 1.00 per cycle
- dflash15: 1.21 per cycle
- draft06b7: 2.19 per cycle
- draft17b7: 3.06 per cycle
- eagle3_3: 1.13 per cycle
- eagle3_7: 1.31 per cycle
- ngram15: 0.99 per cycle
- ngram31: 1.03 per cycle
- ngram7: 0.98 per cycle
- implied AR-drafter step cost r (draft06b7): 0.171 of a target step
- implied AR-drafter step cost r (draft17b7): 0.296 of a target step

## 3. Projected hybrids (acceptance from the PyTorch harness, costs from section 2)

| Method | n | Tokens per target call | Accepted per draft on new code | Projected end-to-end speedup | Speedup vs best measured baseline |
|---|---:|---:|---:|---:|---:|
| Copy + DFlash, two-branch tree | 261 | 13.1 | 4.14 | 9.60× | 1.38× |
| Copy + DFlash, router (deployable) | 261 | 11.9 | 4.22 | 9.03× | 1.30× |
| Copy + DFlash, router min 4 | 261 | 11.9 | 4.04 | 9.12× | 1.31× |
| Copy + DFlash, router min 16 | 261 | 11.6 | 4.21 | 8.74× | 1.25× |
| DFlash alone (harness) | 261 | 9.3 | 4.28 | 6.96× | 1.00× |
| Copy + Qwen3-0.6B, γ=7 (EfficientEdit-style) | 261 | 12.1 | 3.53 | 6.07× | 0.87× |
| Copy + Qwen3-0.6B, γ=15 (EfficientEdit-style) | 261 | 14.3 | 5.25 | 4.72× | 0.68× |
| Copy + Qwen3-1.7B, γ=7 (EfficientEdit-style) | 261 | 12.9 | 4.04 | 5.06× | 0.73× |
| Copy + Qwen3-1.7B, γ=15 (EfficientEdit-style) | 261 | 15.8 | 6.26 | 3.77× | 0.54× |

Best measured baseline: dflash15. Tree verification adds 0.04 AR steps per drafted cycle (ngram31 minus ngram15 cost).

## 4. Decision gates

- G1 cost model holds (DFlash cycle ≤ 1.8 AR steps; DFlash paper on B200: 1.55): PASS (1.21)
- G2 deployable copy + DFlash ≥ 1.15× best measured baseline: PASS (1.30×)
- G3 one-pass beats best AR-drafter hybrid at measured cost: PASS (9.03× vs 6.07× for Copy + Qwen3-0.6B, γ=7 (EfficientEdit-style))
