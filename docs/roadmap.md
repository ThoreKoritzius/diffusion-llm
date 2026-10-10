# Research roadmap

Goal of the repo: show where diffusion language models are *actually* viable versus autoregressive (AR) models,
with honest, reproducible comparisons. Benchmarks live in [`bench/`](../bench/README.md).

## Where diffusion can win, and where it can't

Per query, AR does one prefill plus one cheap KV-cached step per output token. Masked diffusion does one forward
pass over prompt + output window per denoising step. So diffusion only wins when

    steps × cost(full forward)  <  tokens × cost(cached decode step)

- **GPU, batch 1** (memory/launch bound): a full forward costs about the same as a cached step, so diffusion wins
  whenever steps < tokens. This is where published speedups come from.
- **CPU** (compute bound, the hosted playground): a full forward over ~250 tokens is much more work than a 1-token
  step, so diffusion needs very few steps, a cached prompt, and a small window to compete.
- **Short outputs** (text-to-SQL, ~30 tokens) leave little room: at 10 steps the forward-pass saving is ≤3×.
  **Long outputs** (hundreds of tokens) are where parallel decoding compounds.

## Track A: cheap wins on the current ModernBERT model (in progress)

- [x] Fair CPU benchmark, same engine for both arms (`bench/`)
- [x] Schema-compile verification + resampling (`--verify K`), applied to both arms
- [x] MLM head on the SQL window only, smaller window, int8 weights
- [x] Prompt isolation (prompt cannot attend to the SQL window) in training, benchmark and playground
      (`src/prompt_isolation.py`). Inference-only it costs −13 points exec; trained, unknown → first GPU job
- [ ] Prompt caching itself: cached prompt keys/values + window-only forward (and ONNX graphs), if the GPU job is a go
- [ ] Few-step distillation (dParallel-style certainty forcing) to reach ~4 steps
- [ ] Grammar-constrained decoding for diffusion (Mündler et al., ICLR 2026; LAVE, 2026)

GPU experiments for Tracks A and B are scripted in [`gpu-session.md`](gpu-session.md).

## Track B: diffusion vs. AR on the same modern base model

Fine-tune both arms from the *same* pretrained weights so the decoding paradigm is the only variable.

| Pair | AR arm | Diffusion arm | Notes |
|---|---|---|---|
| B1 | Qwen3-1.7B-Base, SFT | SDAR-1.7B (block diffusion continued-pretrained from Qwen3-1.7B), SFT | Cleanest pair available today: same base, open weights, block diffusion gives a real KV cache |
| B2 | Qwen3.5-0.8B-Base / 2B-Base, SFT | own block-diffusion conversion (Fast-dLLM v2 / Efficient-DLM recipe) | Newest small Qwen (hybrid linear attention); conversion is a research problem in itself (see dQwen3.5) |
| B3 | Ettin decoder 150M / 400M, SFT | Ettin encoder 150M / 400M (ModernBERT architecture), masked diffusion | Encoder and decoder trained on identical data with the same recipe: isolates the paradigm at matched size |

Then add execution-reward RL to both arms (GRPO for AR, as in Arctic-Text2SQL-R1; TraceRL for diffusion, which also
targets the loss/generation decoupling found in the PAPL experiment).

Serving for the speed comparison: AR in vLLM/SGLang (GPU) and llama.cpp (CPU); diffusion with CUDA graphs and
prompt caching. Report accuracy-vs-latency at batch 1 and throughput vs. batch size.

## Track C: a task where diffusion's strengths matter

Text-to-SQL stays as the demo, but its short outputs hide diffusion's advantages. Candidates that play to them:

- **Long structured outputs**: multi-statement SQL / dbt models, JSON extraction from long documents.
  Parallel decoding compounds with length.
- **Infilling and editing**: "fix this query", "add a WHERE clause", completing a partially written query.
  Bidirectional context is native to diffusion and awkward for AR.
- **Global constraints**: outputs that must satisfy a schema or grammar end to end; diffusion can revise any
  position, AR cannot revisit earlier tokens.

A SQL *repair/edit* benchmark keeps the domain (and the playground) while testing a natural diffusion strength.
