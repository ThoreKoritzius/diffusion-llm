"""Summarize DFlash / hybrid / AR runs on the edit cohort. Usage: python analyze.py [TAG]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / '2026-11-21_copy-audit'
sys.path.insert(0, str(AUDIT))
from sim import align, replay  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-4B'
rng = np.random.default_rng(0)
# GPU cost model in AR-step units (see copy-audit README): drafted cycle, copy-only cycle, prefill per prompt token
C_DRAFT, C_COPY, C_PRE = 1.55, 1.25, 1 / 250


def load(mode):
    p = HERE / f'{mode}_{TAG}.jsonl'
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def boot(xs, f, n=2000):
    idx = np.arange(len(xs))
    return np.percentile([f([xs[j] for j in rng.choice(idx, len(idx))]) for _ in range(n)], [2.5, 97.5])


def main():
    cohort = {c['id']: c for c in json.loads((AUDIT / 'cohort.json').read_text())}
    D, H, A = load('dflash'), load('hybrid'), load('ar')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f'Qwen/{TAG}')
    norm = lambda s: '\n'.join(l.rstrip() for l in s.strip().splitlines())
    lines = [f'# DFlash on code edits, target {TAG} (MLX, bf16, greedy, M4 Pro)\n']
    rows = []
    for i, r in D.items():
        L = len(r['out'])
        if L < 8 or L >= 1400 or not any(('```' in tok.decode(r['out'][-6:]), r['out'][-1] in (tok.eos_token_id,))):
            continue
        lab = align(r['prompt'], r['out'])
        text = tok.decode(r['out'], skip_special_tokens=True).split('```')[0]
        c = cohort[i]
        g = 'full file' if c['regime'] == 'file' else ('function (CanItEdit)' if c['source'] == 'canitedit' else 'function (EditPackFT)')
        rec = dict(id=i, g=g, L=L, P=len(r['prompt']), lab=lab, noop=norm(text) == norm(c['old']), d=r)
        rec['copy_calls'] = len(replay(r['prompt'], r['out'], lab, 'suffix', 32))
        if i in H:
            rec['h'] = H[i]
        rows.append(rec)
    lines.append(f'{len(rows)} completed edits; {sum(r["noop"] for r in rows)} outputs reproduce the old code unchanged.\n')
    lines += ['## Tokens per target call and DFlash acceptance\n',
              '| Group | n | New-code share | Copy only | DFlash alone | DFlash accepted/guess on new code | Copy + DFlash (measured) | Drafter accepted/guess when needed | H100 projection: copy / DFlash / hybrid |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    groups = sorted({r['g'] for r in rows})
    for sub, label in ((False, 'all'), (True, 'actual edits')):
        for g in groups:
            its = [r for r in rows if r['g'] == g and (not sub or not r['noop'])]
            if not its:
                continue
            toks = sum(r['L'] - 1 for r in its)
            share = np.median([np.mean(np.array(r['lab']) < 0) for r in its])
            copy = toks / sum(r['copy_calls'] for r in its)
            dfl = toks / sum(len(r['d']['cycles']) for r in its)
            nov = [cy['acc'] for r in its for cy in r['d']['cycles'] if r['lab'][min(cy['n'], r['L'] - 1)] < 0]
            hs = [r for r in its if 'h' in r and not r['h']['diverged']]
            hyb = sum(len(r['h']['out']) - 1 for r in hs) / max(1, sum(len(r['h']['cycles']) for r in hs))
            need = [cy['a_draft'] for r in hs for cy in r['h']['cycles'] if cy['drafted'] and cy['a_draft'] >= 0]
            pre = sum(r['P'] * C_PRE for r in its)
            ar_cost = toks + pre
            proj_copy = ar_cost / (sum(r['copy_calls'] for r in its) * C_COPY + pre)
            proj_d = ar_cost / (sum(len(r['d']['cycles']) for r in its) * C_DRAFT + pre)
            if hs:
                htoks = sum(len(r['h']['out']) - 1 for r in hs)
                hpre = sum(r['P'] * C_PRE for r in hs)
                hcost = sum(C_DRAFT if cy['drafted'] else C_COPY for r in hs for cy in r['h']['cycles'])
                proj_h = f'{(htoks + hpre) / (hcost + hpre):.1f}×'
            else:
                proj_h = '–'
            lo, hi = boot(nov, np.mean) if nov else (0, 0)
            lines.append(f'| {g} ({label}) | {len(its)} | {share:.0%} | {copy:.1f} | {dfl:.1f} | {np.mean(nov):.2f} ({lo:.2f}–{hi:.2f}) | '
                         f'{hyb:.1f} | {np.mean(need) if need else float("nan"):.2f} | {proj_copy:.1f}× / {proj_d:.1f}× / {proj_h} |')
    # Mac wall clock: AR subset vs DFlash / hybrid on the same items
    common = [r for r in rows if r['id'] in A]
    if common:
        ar_t = sum(A[r['id']]['total_ms'] for r in common)
        d_t = sum(r['d']['total_ms'] for r in common)
        h_items = [r for r in common if 'h' in r]
        lines += ['\n## Mac wall clock (M4 Pro, batch one, includes prefill)\n',
                  f'{len(common)} edits timed with plain AR. DFlash speedup {ar_t / d_t:.2f}×.']
        if h_items:
            ar_h = sum(A[r['id']]['total_ms'] for r in h_items)
            lines.append(f'Hybrid on {len(h_items)} of them: {ar_h / sum(r["h"]["total_ms"] for r in h_items):.2f}× '
                         '(verifies only the chosen branch, so slightly optimistic versus a real two-branch tree).')
        same = sum(A[r['id']]['out'][:len(r['d']['out'])] == r['d']['out'][:len(A[r['id']]['out'])] for r in common)
        lines.append(f'AR and DFlash outputs agree on the common prefix in {same}/{len(common)} edits (bf16 numerics).')
        dms = [cy['draft_ms'] for r in common for cy in r['d']['cycles']]
        vms = [cy['verify_ms'] for r in common for cy in r['d']['cycles']]
        ar_step = np.median([(A[r['id']]['total_ms'] - A[r['id']]['prefill_ms']) / max(1, len(A[r['id']]['out']) - 1) for r in common])
        lines.append(f'Median per cycle: drafter {np.median(dms):.1f} ms, 16-token verify {np.median(vms):.1f} ms; AR step {ar_step:.1f} ms.')
    lines.append('\nH100 projection uses 1.55 AR steps per drafted cycle (DFlash paper, B200 SGLang), 1.25 per copy-only cycle, and '
                 '4 AR steps of prefill per 1,000 prompt tokens. It is a model, not a measurement.')
    (HERE / f'results_{TAG}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
