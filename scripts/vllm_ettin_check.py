"""Ettin (ModernBERT) decoders on vLLM's Transformers backend vs Transformers itself, on one GPU.

For vllm-project/vllm#61246 / #61276. One model and dtype per process (vLLM keeps GPU memory until exit); appends one
JSON line to --out:
  * greedy agreement: HF and vLLM greedy continuations of fixed prompts (exact matches; at the first divergence, how
    close the two tokens are under HF, so near-ties are told apart from real bugs)
  * vLLM throughput (batch of 256, ignore_eos) and batch-1 latency, plus HF batch-1 latency for scale

    python scripts/vllm_ettin_check.py --model jhu-clsp/ettin-decoder-150m --dtype bfloat16 --out ettin.jsonl
"""

import argparse
import gc
import json
import time

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "SELECT name, COUNT(*) FROM orders",
    "In 1905, Albert Einstein published",
    "The three primary colors are",
    "Photosynthesis converts sunlight into",
    "Once upon a time, in a small village,",
    "To install the package, run",
    "The derivative of x squared is",
    "Water boils at a temperature of",
    "The quick brown fox",
    "import numpy as np\nx = np.",
    "The largest planet in the solar system is",
    "Machine learning is a field of",
    "Q: What is 12 times 12?\nA:",
    "The French Revolution began in",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--impl", default="transformers", help="vLLM model_impl: transformers | auto")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-util", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rec = {"model": args.model, "dtype": args.dtype, "impl": args.impl}
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = [tok(p)["input_ids"] for p in PROMPTS]

    # ---- Transformers reference: greedy, keeping each step's log-probs
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).cuda().eval()
    ref, ref_lp = [], []
    with torch.no_grad():
        for p in ids:
            out = hf.generate(torch.tensor([p], device="cuda"), max_new_tokens=args.max_tokens, do_sample=False,
                              output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
            ref.append(out.sequences[0, len(p):].tolist())
            ref_lp.append([s[0].float().log_softmax(-1).cpu() for s in out.scores])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        hf.generate(torch.tensor([ids[0]], device="cuda"), max_new_tokens=128, min_new_tokens=128, do_sample=False,
                    pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); rec["hf_b1_ms_per_tok"] = (time.perf_counter() - t0) / 128 * 1e3
    del hf; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(model=args.model, dtype=args.dtype, model_impl=args.impl, gpu_memory_utilization=args.gpu_util,
                  max_model_len=2048, seed=0)
    except Exception as e:  # noqa: BLE001 - the failure itself is the result
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        print(json.dumps(rec)); open(args.out, "a").write(json.dumps(rec) + "\n")
        raise
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate([{"prompt_token_ids": p} for p in ids], greedy)
    exact, gaps, firsts = 0, [], []
    for r, lps, o in zip(ref, ref_lp, outs):
        got = list(o.outputs[0].token_ids)
        n = min(len(r), len(got))
        i = next((k for k in range(n) if r[k] != got[k]), None)
        if i is None and len(r) == len(got):
            exact += 1
            continue
        i = n if i is None else i
        firsts.append(i)
        if i < len(lps) and i < len(got):  # HF log-prob gap between its own pick and vLLM's pick at the divergence
            gaps.append(round(float(lps[i][r[i]] - lps[i][got[i]]), 4))
    rec.update(exact=f"{exact}/{len(ids)}", first_divergence=firsts, hf_logprob_gap_at_divergence=gaps,
               sample=tok.decode(outs[0].outputs[0].token_ids))

    p1 = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    llm.generate([{"prompt_token_ids": ids[0]}], p1)  # warm-up
    t0 = time.perf_counter(); llm.generate([{"prompt_token_ids": ids[0]}], p1)
    rec["vllm_b1_ms_per_tok"] = (time.perf_counter() - t0) / 128 * 1e3
    batch = [{"prompt_token_ids": ids[k % len(ids)]} for k in range(256)]
    t0 = time.perf_counter(); llm.generate(batch, p1)
    rec["vllm_b256_tok_per_s"] = round(256 * 128 / (time.perf_counter() - t0))
    print(json.dumps(rec)); open(args.out, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
