"""CPU-only follow-ups on existing logs -> results/<tag>/postprocess.md and diverge_ids.json.

1. Measured hybrid gain in the PyTorch harness: paired sum(DFlash total_ms) / sum(router total_ms) on identical
   requests, with a bootstrap interval, overall and for single-function vs full-file edits. Eager PyTorch, so this is a
   measured same-harness comparison, not a serving-engine speedup.
2. Drafter resume cost: in router mode the drafter ingests every target hidden state queued during a copy run.
   Drafted-cycle time is related to that context length and compared with ordinary DFlash cycles.
3. Output disagreement with plain AR in vLLM: identical / stop-or-length only (one output is a prefix of the other) /
   mid-sequence mismatch. Writes up to 20 mid-sequence cases for the GPU diagnosis (diverge.py).
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from common import HERE, cohort, group

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
rng = np.random.default_rng(0)


def jl(name):
    p = RES / f'{name}.jsonl'
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def boot_ratio(a, b, n=4000):
    a, b = np.array(a), np.array(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return a.sum() / b.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def ctx_lengths(r):
    """Queued hidden-state count at each drafted cycle (logged as ctx_len, else reconstructed from the cycle log)."""
    out, pend = [], len(r['prompt'])
    for c in r['cycles']:
        if c['drafted']:
            out.append((c.get('ctx_len', pend), c.get('cycle_ms', c.get('draft_ms', 0) + c.get('verify_ms', 0)), c.get('calls', 1)))
            pend = 0
        pend += c['acc'] + 1
    return out


def main():
    items = {c['id']: c for c in cohort()}
    tok = AutoTokenizer.from_pretrained(f'Qwen/{TAG}')
    L = [f'# Post-processing: {TAG}\n']
    # 1. measured harness wall clock
    D = jl('harness_dflash')
    L += ['## 1. Measured in the PyTorch harness: DFlash total time / hybrid total time (same requests, prefill included)\n',
          '| Hybrid | Scope | n | Speedup over DFlash | 95% bootstrap CI |', '|---|---|---:|---:|---|']
    for name in ('hybrid_router4', 'hybrid_router8', 'hybrid_router16', 'hybrid_oracle'):
        Hm = jl('harness_' + name)
        for scope, sel in (('all', lambda g: True), ('single function', lambda g: g != 'full file'), ('full file', lambda g: g == 'full file')):
            ids = [i for i in Hm if i in D and sel(group(items[i]))]
            if not ids:
                continue
            r, lo, hi = boot_ratio([D[i]['total_ms'] for i in ids], [Hm[i]['total_ms'] for i in ids])
            L.append(f'| {name}{" (extra verify calls, not deployable timing)" if name == "hybrid_oracle" else ""} | {scope} | {len(ids)} | {r:.2f}× | {lo:.2f}–{hi:.2f} |')
    # 2. resume cost
    L += ['\n## 2. Drafter resume cost (drafted cycles: draft + 16-token verify)\n',
          '| Source | Queued context at draft | n cycles | Median cycle ms |', '|---|---|---:|---:|']
    bins = [(1, 16), (17, 48), (49, 128), (129, 10 ** 6)]
    for name in ('harness_dflash', 'hybrid_router8'):
        Hm = D if name == 'harness_dflash' else jl('harness_' + name)
        rows = [x for r in Hm.values() for x in ctx_lengths(r)[1:] if x[2] == 1]  # skip first cycle (prompt context)
        for lo_, hi_ in bins:
            sel = [ms for c, ms, _ in rows if lo_ <= c <= hi_]
            if sel:
                L.append(f'| {name} | {lo_}–{hi_ if hi_ < 10 ** 6 else "∞"} | {len(sel)} | {np.median(sel):.1f} |')
        if rows:
            x = np.array([c for c, _, _ in rows], float); y = np.array([ms for _, ms, _ in rows])
            slope = np.polyfit(x, y, 1)[0]
            L.append(f'| {name} | linear fit | {len(rows)} | {slope * 100:.2f} ms per 100 queued states |')
    # 3. disagreement classes
    V = {p.stem[5:]: {json.loads(l)['id']: json.loads(l) for l in p.open()} for p in sorted(RES.glob('vllm_*.jsonl'))}
    AR = V.get('ar', {})
    L += ['\n## 3. vLLM output vs plain AR\n', '| Config | n | Identical | Stop/length only | Mid-sequence mismatch | Median first-mismatch position |',
          '|---|---:|---:|---:|---:|---:|']
    picks = []
    for name, R in V.items():
        if name == 'ar':
            continue
        same = stop = mid = 0
        pos = []
        for i, r in R.items():
            if i not in AR:
                continue
            a = tok.encode(AR[i]['text'], add_special_tokens=False)
            b = tok.encode(r['text'], add_special_tokens=False)
            if a == b:
                same += 1
                continue
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
            if k is None:
                stop += 1
            else:
                mid += 1
                pos.append(k)
                if name in ('dflash15', 'ngram31'):
                    picks.append(dict(id=i, config=name, k=k))
        n = same + stop + mid
        L.append(f'| {name} | {n} | {same} | {stop} | {mid} | {int(np.median(pos)) if pos else "–"} |')
    seen, chosen = set(), []
    for p in sorted(picks, key=lambda p: (p['config'] != 'dflash15', p['id'])):
        if p['id'] not in seen and len(chosen) < 20:
            seen.add(p['id']); chosen.append(p)
    (RES / 'diverge_ids.json').write_text(json.dumps(chosen))
    L.append(f'\n{len(chosen)} mid-sequence cases written to diverge_ids.json for the GPU diagnosis.')
    (RES / 'postprocess.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
