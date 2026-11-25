"""Diagnose mid-sequence disagreements between speculative outputs and plain AR (cases from postprocess.py).

For each case, the shared prefix (prompt + AR output up to the first mismatch k) is replayed through the target in
three ways: one token at a time (AR decode path), 16-token blocks (speculative verify path), one full forward.
At position k we record top-2 token ids, logits and margin per path, and which output each path's argmax agrees
with. A small margin that flips between paths indicates a numerical near-tie; a large margin with a consistent
argmax that differs from one output indicates a real bug. If vllm_ar_repeat exists, AR-vs-AR reproducibility is added.

Usage: python diverge.py [--target Qwen/Qwen3-8B]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from common import HERE, cohort, prompt_text


@torch.inference_mode()
def logits_at(model, P, prefix, step):
    cache = DynamicCache()
    model(torch.tensor([P], device='cuda'), past_key_values=cache, use_cache=True, logits_to_keep=1)
    if not prefix:
        return model(torch.tensor([P], device='cuda'), logits_to_keep=1).logits[0, -1].float()
    out = None
    for s in range(0, len(prefix), step):
        out = model(torch.tensor([prefix[s:s + step]], device='cuda'), past_key_values=cache, use_cache=True)
    return out.logits[0, -1].float()


def top2(l):
    v, i = l.topk(2)
    return dict(top1=int(i[0]), top2=int(i[1]), l1=round(float(v[0]), 4), l2=round(float(v[1]), 4), margin=round(float(v[0] - v[1]), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='Qwen/Qwen3-8B')
    a = ap.parse_args()
    tag = a.target.split('/')[-1]
    res = HERE / 'results' / tag
    cases = json.loads((res / 'diverge_ids.json').read_text())
    items = {c['id']: c for c in cohort()}
    tok = AutoTokenizer.from_pretrained(a.target)
    model = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    V = {}
    for name in {c['config'] for c in cases} | {'ar', 'ar_repeat'}:
        p = res / f'vllm_{name}.jsonl'
        if p.exists():
            V[name] = {json.loads(l)['id']: json.loads(l)['text'] for l in p.open()}
    out = []
    for c in cases:
        P = tok.encode(prompt_text(tok, items[c['id']]), add_special_tokens=False)
        ar = tok.encode(V['ar'][c['id']], add_special_tokens=False)
        sp = tok.encode(V[c['config']][c['id']], add_special_tokens=False)
        k = c['k']
        rec = dict(id=c['id'], config=c['config'], k=k, ar_token=ar[k], spec_token=sp[k],
                   ar_token_str=tok.decode([ar[k]]), spec_token_str=tok.decode([sp[k]]))
        for path, step in (('one_token', 1), ('block16', 16), ('full', 10 ** 6)):
            t = top2(logits_at(model, P, ar[:k], step) if step < 10 ** 6 else
                     model(torch.tensor([P + ar[:k]], device='cuda'), logits_to_keep=1).logits[0, -1].float())
            t['agrees_with'] = 'ar' if t['top1'] == ar[k] else ('spec' if t['top1'] == sp[k] else 'neither')
            rec[path] = t
        if 'ar_repeat' in V and c['id'] in V['ar_repeat']:
            rec['ar_repeat_identical'] = V['ar_repeat'][c['id']] == V['ar'][c['id']]
        out.append(rec)
        print(c['id'], c['config'], 'k', k, {p: (rec[p]['margin'], rec[p]['agrees_with']) for p in ('one_token', 'block16', 'full')},
              'ar_repeat_identical', rec.get('ar_repeat_identical'), flush=True)
    (res / 'diverge.json').write_text(json.dumps(out, indent=1))
    m = [r['one_token']['margin'] for r in out]
    flips = sum(len({r[p]['top1'] for p in ('one_token', 'block16', 'full')}) > 1 for r in out)
    print(f'\n{len(out)} cases: median top-2 margin {sorted(m)[len(m) // 2] if m else float("nan")} (bf16 logits), '
          f'argmax changes between replay paths in {flips}, AR-repeat identical in '
          f"{sum(r.get('ar_repeat_identical') is True for r in out)}/{sum('ar_repeat_identical' in r for r in out)}")


if __name__ == '__main__':
    main()
