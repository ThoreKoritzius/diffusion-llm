"""On-policy drafter training data for text-to-SQL with Qwen3-8B (Speculators format: input_ids + loss_mask).

Same recipe as scripts/spec_data.py on main (session 2), adapted to the chat-template prompt that sqlbench.py serves:
the target answers TRAINING rows of gretelai/synthetic_text_to_sql greedily in vLLM, and each row is the served
prompt followed by the target's own output, loss on the output only. Never touches the test split.

    python spec_sql_data.py --model Qwen/Qwen3-8B --n 12000 --out spec-sql8b/data.jsonl
"""
import argparse
import json
import os
from pathlib import Path

from sqlbench import MAX_NEW, STOP, prompt_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--n', type=int, default=12000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset('gretelai/synthetic_text_to_sql', split='train')
    ds = ds.filter(lambda ex: ex['sql_prompt'].strip() != '').select(range(a.n))
    prompts = [tok(prompt_text(tok, {'prompt': ex['sql_prompt'], 'context': ex['sql_context'] or ''}),
                   add_special_tokens=False)['input_ids'] for ex in ds]
    keep = [i for i, p in enumerate(prompts) if len(p) + MAX_NEW <= 2048]
    prompts = [prompts[i] for i in keep]
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get('VLLM_GPU_UTIL', '0.85')))
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, stop=STOP, include_stop_str_in_output=True)
    outs = llm.generate([{'prompt_token_ids': p} for p in prompts], params)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(a.out, 'w') as f:
        for p, o in zip(prompts, outs):
            gen = list(o.outputs[0].token_ids)  # ends with the closing fence, so the drafter learns where SQL ends
            n_tok += len(p) + len(gen)
            f.write(json.dumps({'input_ids': p + gen, 'loss_mask': [0] * len(p) + [1] * len(gen)}) + '\n')
    print(f'wrote {len(prompts)} rows ({n_tok / len(prompts):.0f} tokens/row) to {a.out}')


if __name__ == '__main__':
    main()
