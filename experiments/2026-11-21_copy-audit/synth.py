"""Synthetic-drafter sweep: tokens per target call as a function of drafter quality, with and without copy.

Drafter model: per-token acceptance probability p (i.i.d.), block of 15 drafted tokens (DFlash block 16 = 15 drafts
+ 1 target token). Exact replay against recorded target outputs, same suffix-copy proposer as sim.py, best-branch
tree verification for the hybrid. Validated against the measured Qwen2.5-Coder-0.5B AR drafter (draft_A.json).

GPU cost model (in units of one plain AR decode step), calibrated on DFlash paper Table 3 (Qwen3-8B, B200, SGLang,
concurrency 1: 229 tok/s AR vs 955 tok/s DFlash at tau 6.50 -> one draft+verify cycle = 1.55 AR steps).
"""
import json
import random
from pathlib import Path

import numpy as np

from sim import SuffixIndex, align, lcp, load

HERE = Path(__file__).resolve().parent
KD = 15


def replay(P, Y, policy, p, rng):
    sx = SuffixIndex(P); sx.extend(Y[:1])
    i, n_cyc, n_draft = 1, 0, 0
    while i < len(Y):
        a_copy, src, cp = 0, None, []
        if policy in ('copy', 'hybrid'):
            cp, src = sx.propose(32)
            a_copy = lcp(cp, Y, i)
        a_d = -1
        if policy == 'draft' or (policy == 'hybrid' and a_copy < KD):
            a_d = 0
            while a_d < KD and rng.random() < p:
                a_d += 1
            a_d = min(a_d, len(Y) - i)
            n_draft += 1
        a = max(a_copy, a_d)
        step = min(a + 1, len(Y) - i)
        sx.extend(Y[i:i + step]); sx.ptr = (src + a) if (cp and a == a_copy and a) else None
        i += step; n_cyc += 1
    return n_cyc, n_draft


def main():
    cohort, rows = load()
    rows = [r for r in rows if r['stop'] != 'limit' and len(r['out_ids']) >= 8]
    A = {a['id']: a for a in json.loads((HERE / 'draft_A.json').read_text())['items']}
    groups = {'function': [r for r in rows if cohort[r['id']]['regime'] == 'function'],
              'file': [r for r in rows if cohort[r['id']]['regime'] == 'file']}
    ps = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.93, 0.95]
    out = {}
    for g, its in groups.items():
        out[g] = {}
        toks = sum(len(r['out_ids']) - 1 for r in its)
        for p in ps:
            res = {}
            for pol in ('draft', 'hybrid'):
                cyc = drafted = 0
                for seed in range(3):
                    rng = random.Random(seed)
                    for r in its:
                        c, d = replay(r['prompt_ids'], r['out_ids'], pol, p, rng)
                        cyc += c; drafted += d
                res[pol] = dict(tpc=3 * toks / cyc, draft_frac=drafted / cyc)
            out[g][p] = res
        c, _ = zip(*[replay(r['prompt_ids'], r['out_ids'], 'copy', 0, random.Random(0)) for r in its])
        out[g]['copy'] = toks / sum(c)
        # measured 0.5B AR drafter on the same items (16-token drafts), for validation
        m = [A[r['id']] for r in its if r['id'] in A]
        out[g]['measured'] = {pol: sum(a['L'] - 1 for a in m) / sum(len(a[pol]) for a in m) for pol in ('ar_draft', 'hybrid')}
        out[g]['measured_ids'] = len(m)
    (HERE / 'synth.json').write_text(json.dumps(out))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
