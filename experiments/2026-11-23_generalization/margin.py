"""Are disagreements between a speculative method and plain AR numerical near-ties? (runs on Mac MPS or GPU)

For each request whose text differs from AR: retokenize both outputs, find the first differing token k, run the
target once on prompt + AR[:k] and report the logit gap between AR's token and the method's token, and the top-2
margin. Near-ties (gap of a fraction of a logit) flip with kernel shapes and are not errors of the method; a large
gap means the method emitted a token the target does not prefer -> a real bug.

Usage: python margin.py KEY METHOD [COHORT ...] [--device mps]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import HERE, cohort, prompt_text, targets


def load(key, c, m):
    return {r['id']: r for r in map(json.loads, (HERE / 'results' / key / c / f'vllm_{m}.jsonl').open())}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('method')
    ap.add_argument('cohorts', nargs='*', default=['edit261', 'ceb150', 'diff104'])
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda')
    a = ap.parse_args()
    name = targets()[a.key]['model']
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(a.device).eval()
    rows = []
    for c in a.cohorts:
        items = {it['id']: it for it in cohort(c)}
        A, R = load(a.key, c, 'ar'), load(a.key, c, a.method)
        for i in sorted(set(A) & set(R)):
            if A[i]['text'] == R[i]['text']:
                continue
            ya = tok(A[i]['text'], add_special_tokens=False).input_ids
            yr = tok(R[i]['text'], add_special_tokens=False).input_ids
            k = next((t for t in range(min(len(ya), len(yr))) if ya[t] != yr[t]), None)
            if k is None:
                continue
            P = tok(prompt_text(tok, items[i]), add_special_tokens=False).input_ids
            x = torch.tensor([P + ya[:k]], device=a.device)
            lg = model(x, logits_to_keep=1).logits[0, -1].float()
            top = torch.topk(lg, 2)
            gap = float(lg[ya[k]] - lg[yr[k]])
            rows.append(dict(cohort=c, id=i, k=k, gap_ar_minus_method=gap, top2_margin=float(top.values[0] - top.values[1]),
                             top1_is=('ar' if int(top.indices[0]) == ya[k] else 'method' if int(top.indices[0]) == yr[k] else 'other'),
                             ar_tok=tok.decode([ya[k]]), method_tok=tok.decode([yr[k]])))
            r = rows[-1]
            print(f"{c} {i} k={k} gap={r['gap_ar_minus_method']:+.3f} top2={r['top2_margin']:.3f} top1={r['top1_is']} "
                  f"{r['ar_tok']!r} vs {r['method_tok']!r}", flush=True)
            if a.device == 'mps':
                torch.mps.empty_cache()
    out = HERE / 'results' / a.key / f'margin_{a.method}.json'
    out.write_text(json.dumps(rows, indent=1))
    gaps = sorted(abs(r['gap_ar_minus_method']) for r in rows)
    if gaps:
        print(f'{len(rows)} disagreements: |gap| median {gaps[len(gaps) // 2]:.3f}, max {gaps[-1]:.3f}; '
              f'{sum(g < 1.0 for g in gaps)} under 1 logit -> {out}')


if __name__ == '__main__':
    main()
