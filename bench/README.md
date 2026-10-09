# Benchmark: diffusion vs. autoregressive text-to-SQL

Measures quality and latency of both arms on the same examples, same hardware, same engine.

```bash
pip install -r bench/requirements.txt

# CPU track (matches the hosted playground: ONNX Runtime, 3 threads, batch 1)
python -m bench.run --arm diffusion --engine onnx --model checkpoints/diffusion-sql-modernbert
python -m bench.run --arm ar        --engine onnx --model checkpoints/<ar-checkpoint>
python -m bench.report bench/results/cpu          # table + accuracy-vs-latency plot

# GPU track (run on a CUDA box)
python -m bench.run --arm ar        --engine vllm  --device cuda --model ...
python -m bench.run --arm diffusion --engine torch --device cuda --model ...
```

## Protocol

| | |
|---|---|
| Eval set | first 256 rows of `gretelai/synthetic_text_to_sql` test with a non-empty prompt (same slice as `experiments/`) |
| Metrics | **exec**: result matches gold on the example's own SQLite DB (only over rows whose gold SQL runs in SQLite, `n_exec`) · **valid**: prediction executes · **exact**: normalized string match |
| Latency | wall clock per example at batch 1, including tokenization, after 3 warmup examples; p50 / p95 / mean |
| Cost | `avg_forward`: forward passes per example (AR: generated tokens incl. stop token; diffusion: denoising steps) |
| Fairness | both arms use the same engine per track (ONNX Runtime fp32 on CPU), same thread count, greedy decoding, the same prompt layout they were trained with, and the same optional verify step (`--verify K`) |

Engines: `diffusion/torch`, `diffusion/onnx` (production graph; `--window-head` applies the MLM head to the SQL window only, `--int8` dynamic weight quantization, `--window` canvas size), `ar/torch` (HF `generate` + KV cache), `ar/onnx` (optimum export with KV cache), `ar/vllm` (GPU, CUDA graphs).

`--verify K`: keep the greedy output if it compiles against the schema in the prompt (`EXPLAIN` on an in-memory SQLite DB built from the context); otherwise try up to K sampled candidates and take the first that compiles. Uses only inference-time inputs; extra forward passes are counted.

## Caveats

- One dataset (synthetic, simple single-statement SQL, ~30 output tokens). Short outputs are the hardest case for diffusion speed: AR needs only ~30 cheap KV-cached steps.
- SQLite execution undercounts correct predictions written in another dialect; `exec` is restricted to rows where gold runs, `valid` is not.
- Run one benchmark at a time: concurrent jobs distort latency.
