"""Paired timing repeat in the PyTorch harness: DFlash alone vs copy + DFlash router (min match 8).

Requests are fixed in advance: 18 per group (CanItEdit functions, EditPackFT functions, full files), spread over
prompt-length quantiles. Both paths are warmed up, then 3 repeats with alternating order. Whole-request latency
(prefill included) and per-cycle logs (incl. the drafter's queued context at each draft) are saved.

Usage: python paired_timing.py [--target Qwen/Qwen3-8B] [--reps 3] [--per-group 18]
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import HERE, cohort, group, max_new_tokens, prompt_text
from harness import run_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='Qwen/Qwen3-8B')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--per-group', type=int, default=18)
    a = ap.parse_args()
    from dflash.model import DFlashDraftModel
    tag = a.target.split('/')[-1]
    tok = AutoTokenizer.from_pretrained(a.target)
    target = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    draft = DFlashDraftModel.from_pretrained(f'z-lab/{tag}-DFlash-b16', dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    items = cohort()
    chosen = []
    for g in ('function (CanItEdit)', 'function (EditPackFT)', 'full file'):
        its = sorted([it for it in items if group(it) == g], key=lambda it: len(it['old']))
        idx = np.linspace(0, len(its) - 1, min(a.per_group, len(its))).round().astype(int)
        chosen += [its[i] for i in sorted(set(idx))]
    reqs = [(it, tok.encode(prompt_text(tok, it), add_special_tokens=False), max_new_tokens(tok, it)) for it in chosen]
    for it, P, lim in reqs[:3]:  # warm-up both paths
        run_spec(target, draft, tok, P, lim, 'dflash', 'cuda')
        run_spec(target, draft, tok, P, lim, 'hybrid_router', 'cuda', router_min=8)
    path = HERE / 'results' / tag / 'paired_timing.jsonl'
    with path.open('w') as f:
        for rep in range(a.reps):
            for j, (it, P, lim) in enumerate(reqs):
                order = ['dflash', 'hybrid_router'] if (rep + j) % 2 == 0 else ['hybrid_router', 'dflash']
                for mode in order:
                    out, cyc, meta = run_spec(target, draft, tok, P, lim, mode, 'cuda', router_min=8)
                    f.write(json.dumps(dict(id=it['id'], group=group(it), rep=rep, mode=mode, order=order.index(mode),
                                            out_tokens=len(out), cycles=cyc, **meta)) + '\n')
                f.flush()
            print('rep', rep, 'done', flush=True)
    rows = [json.loads(l) for l in path.open()]
    by = {}
    for r in rows:
        by.setdefault(r['id'], {}).setdefault(r['rep'], {})[r['mode']] = r['total_ms']
    ids = [i for i, reps in by.items() if all(len(v) == 2 for v in reps.values())]
    d = np.array([sum(v['dflash'] for v in by[i].values()) for i in ids])
    h = np.array([sum(v['hybrid_router'] for v in by[i].values()) for i in ids])
    boot = [d[k].sum() / h[k].sum() for k in np.random.default_rng(0).integers(0, len(ids), (4000, len(ids)))]
    print(f'paired speedup of router-8 over DFlash: {d.sum() / h.sum():.3f}x '
          f'(95% CI {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}, bootstrap over {len(ids)} requests, {a.reps} repeats each)')


if __name__ == '__main__':
    main()
