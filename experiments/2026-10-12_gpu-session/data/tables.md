### GPU

| configuration | params | exec | valid | exact | forward passes | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AR** Qwen2.5-0.5B AR · vLLM | 494M | 0.721 | 0.805 | 0.363 | 30.1 | 50 | 146 | 63 |
| **AR** Qwen2.5-0.5B AR · HF static cache + compile, bf16 | 494M | 0.726 | 0.809 | 0.359 | 30.1 | 152 | 409 | 285 |
| **AR** Qwen2.5-0.5B AR · HF eager, fp32 | 494M | 0.716 | 0.805 | 0.355 | 29.9 | 264 | 717 | 320 |
| **AR** Ettin-150M AR, 10 ep · HF static cache + compile, bf16 | 150M | 0.542 | 0.742 | 0.254 | 33.3 | 197 | 517 | 261 |
| **AR** Ettin-150M AR, 3 ep · HF static cache + compile, bf16 | 150M | 0.517 | 0.730 | 0.246 | 32.6 | 192 | 501 | 343 |
| **Diffusion** ModernBERT diffusion, distilled · CUDA graphs, bf16 | 150M | 0.592 | 0.688 | 0.297 | 7.4 | 17 | 49 | 23 |
| **Diffusion** ModernBERT diffusion, distilled · PyTorch eager + prompt cache, bf16 | 150M | 0.592 | 0.691 | 0.301 | 7.4 | 98 | 245 | 122 |
| **Diffusion** Ettin-150M diffusion, 10 ep, distilled · CUDA graphs, bf16 | 150M | 0.557 | 0.648 | 0.305 | 7.8 | 17 | 49 | 25 |
| **Diffusion** Ettin-150M diffusion, 10 ep · CUDA graphs, bf16 | 150M | 0.547 | 0.629 | 0.320 | 9.6 | 28 | 49 | 30 |
| **Diffusion** ModernBERT diffusion, production model · CUDA graphs, bf16 | 150M | 0.592 | 0.688 | 0.312 | 9.4 | 29 | 49 | 29 |
| **Diffusion** ModernBERT diffusion, production model · PyTorch eager, bf16 | 150M | 0.577 | 0.668 | 0.305 | 9.4 | 160 | 266 | 154 |

### CPU

| configuration | params | exec | valid | exact | forward passes | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AR** Qwen2.5-0.5B AR · llama.cpp Q8_0 | 494M | 0.716 | 0.801 | 0.363 | 31.3 | 264 | 584 | 304 |
| **AR** Qwen2.5-0.5B AR · llama.cpp f16 | 494M | 0.716 | 0.805 | 0.355 | 30.9 | 400 | 906 | 462 |
| **AR** Qwen2.5-0.5B AR · ONNX Runtime fp32 (KV cache) | 494M | 0.716 | 0.805 | 0.355 | 29.9 | 543 | 1395 | 649 |
| **AR** Ettin-150M AR, 10 ep · PyTorch fp32 (KV cache) | 150M | 0.527 | 0.734 | 0.250 | 32.2 | 385 | 870 | 459 |
| **AR** Ettin-150M AR, 3 ep · PyTorch fp32 (KV cache) | 150M | 0.517 | 0.719 | 0.242 | 32.3 | 380 | 904 | 467 |
| **Diffusion** ModernBERT diffusion, distilled · ONNX + prompt cache, int8 | 150M | 0.582 | 0.688 | 0.297 | 7.5 | 168 | 444 | 216 |
| **Diffusion** ModernBERT diffusion, distilled · ONNX + prompt cache, fp32 | 150M | 0.602 | 0.695 | 0.309 | 7.4 | 253 | 757 | 347 |
| **Diffusion** Ettin-150M diffusion, 10 ep, distilled · ONNX + prompt cache, int8 | 150M | 0.532 | 0.625 | 0.285 | 8.1 | 175 | 429 | 221 |
| **Diffusion** Ettin-150M diffusion, 10 ep, distilled · ONNX + prompt cache, fp32 | 150M | 0.542 | 0.629 | 0.305 | 7.8 | 271 | 747 | 367 |
| **Diffusion** Ettin-150M diffusion, 10 ep · ONNX + prompt cache, fp32 | 150M | 0.562 | 0.637 | 0.324 | 9.6 | 447 | 771 | 448 |
| **Diffusion** Ettin-150M diffusion, 3 ep · PyTorch fp32 | 150M | 0.473 | 0.598 | 0.250 | 10.5 | 882 | 1840 | 872 |
| **Diffusion** ModernBERT diffusion, production config (window 128) · ONNX fp32 | 150M | 0.547 | 0.582 | 0.309 | 9.5 | 803 | 1734 | 863 |
