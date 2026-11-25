"""Greedy target outputs (Qwen2.5-Coder-1.5B-Instruct, FP16, MPS, batch one).

The target's own greedy output is the ground truth for every lossless drafter:
speculative decoding with exact greedy verification must reproduce it token for token.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
TARGET = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
FENCE = '```python\n'


def build_prompt(tok, item):
    what = 'function' if item['regime'] == 'function' else 'file'
    user = (f'Revise the following Python {what} according to the change request. '
            f'Reply with only the complete revised {what} in a single Python code block.\n\n'
            f"Change request: {item['instruction']}\n\n```python\n{item['old'].rstrip()}\n```")
    text = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                   add_generation_prompt=True)
    return text + FENCE


@torch.inference_mode()
def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.float16,
                                                 attn_implementation='sdpa').eval().to('mps')
    items = json.loads((HERE / 'cohort.json').read_text())
    out_path = HERE / 'targets.jsonl'
    done = set()
    if out_path.exists():
        done = {json.loads(l)['id'] for l in out_path.open()}
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
    with out_path.open('a') as f:
        for k, item in enumerate(items):
            if item['id'] in done:
                continue
            prompt = build_prompt(tok, item)
            ids = tok(prompt, return_tensors='pt').input_ids.to('mps')
            limit = min(1400, int(1.6 * len(tok.encode(item['old'])) + 96))
            torch.mps.synchronize(); t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.mps.synchronize(); prefill = time.perf_counter() - t0
            gen, stop = [], 'limit'
            while len(gen) < limit:
                t = nxt.item()
                gen.append(t)
                if t in eos:
                    stop = 'eos'; break
                if '```' in tok.decode(gen[-4:]):
                    stop = 'fence'; break
                out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.mps.synchronize(); total = time.perf_counter() - t0
            f.write(json.dumps(dict(id=item['id'], prompt_ids=ids[0].tolist(), out_ids=gen, stop=stop,
                                    prefill_ms=prefill * 1e3, total_ms=total * 1e3)) + '\n')
            f.flush()
            print(k, item['id'], len(gen), stop, round(total, 1), 's', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
