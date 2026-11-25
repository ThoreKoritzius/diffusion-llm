"""Matched comparison on identical Qwen3-4B edits: copy, DFlash, AR drafters, and their copy hybrids.

H100 cost model in plain-AR-step units (batch one):
  copy-only cycle 1.25; DFlash drafted cycle 1.55 (DFlash paper B200/SGLang calibration);
  AR-drafter cycle 1.25 + gamma * r, r = drafter step / target step. r is a range: parameter ratio as the
  optimistic end, kernel-overhead-dominated small-model decode as the pessimistic end.
Prefill: 4 AR steps per 1,000 prompt tokens, identical for all methods.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-21_copy-audit'))
from sim import align, replay  # noqa: E402

R = {'Qwen3-0.6B': (0.15, 0.30, 0.45), 'Qwen3-1.7B': (0.43, 0.50, 0.60)}
C_COPY, C_DF, C_PRE = 1.25, 1.55, 1 / 250


def jl(p):
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def main():
    cohort = {c['id']: c for c in json.loads((HERE.parent / '2026-11-21_copy-audit' / 'cohort.json').read_text())}
    D, H = jl(HERE / 'dflash_Qwen3-4B.jsonl'), jl(HERE / 'hybrid_Qwen3-4B.jsonl')
    A = {t: jl(HERE / f'ardraft_{t}.jsonl') for t in R}
    ids = [i for i in D if i in H and not H[i]['diverged'] and all(i in a for a in A.values())
           and 8 <= len(D[i]['out']) < 1400]
    lines = ['# Matched drafter comparison on Qwen3-4B edits\n',
             f'{len(ids)} edits present in every run (DFlash hybrid runs that diverged through bf16 numerics excluded).\n']
    for g, sel in (('functions', lambda c: c['regime'] == 'function'), ('full files', lambda c: c['regime'] == 'file')):
        its = [i for i in ids if sel(cohort[i])]
        toks = sum(len(D[i]['out']) - 1 for i in its)
        pre = sum(len(D[i]['prompt']) * C_PRE for i in its)
        ar = toks + pre
        lines += [f'## {g} (n={len(its)})\n',
                  '| Method | Tokens per target call | Accepted per draft on new code | H100 projection (optimistic / mid / pessimistic drafter cost) |',
                  '|---|---:|---:|---|']
        copy_calls = sum(len(replay(D[i]['prompt'], D[i]['out'], align(D[i]['prompt'], D[i]['out']), 'suffix', 32)) for i in its)
        lines.append(f'| Copy only | {toks / copy_calls:.1f} | – | {ar / (copy_calls * C_COPY + pre):.1f}× |')
        # DFlash alone and DFlash hybrid (measured with the real target)
        for label, src, key in (('DFlash alone', D, None), ('Copy + DFlash', H, None)):
            cyc = [c for i in its for c in src[i]['cycles']]
            labs = {i: align(src[i]['prompt'], src[i]['out']) for i in its}
            nov = [c['acc'] if src is D else c['a_draft'] for i in its for c in src[i]['cycles']
                   if c['drafted'] and labs[i][min(c['n'], len(labs[i]) - 1)] < 0 and (src is D or c['a_draft'] >= 0)]
            cost = sum(C_DF if c['drafted'] else C_COPY for c in cyc)
            tt = sum(len(src[i]['out']) - 1 for i in its)
            lines.append(f'| {label} | {tt / len(cyc):.1f} | {np.mean(nov):.2f} | {(tt + pre) / (cost + pre):.1f}× |')
        for t, rs in R.items():
            for key, label, gamma in (('alone7', f'{t} alone, γ=7', 7), ('hybrid7', f'Copy + {t}, γ=7', 7),
                                      ('hybrid15', f'Copy + {t}, γ=15', 15)):
                cyc = [c for i in its for c in A[t][i][key]]
                nov = [c['a_draft'] for c in cyc if c['drafted'] and c['novel']]
                proj = ' / '.join(f"{ar / (sum(C_COPY + (gamma * r if c['drafted'] else 0) for c in cyc) + pre):.1f}×" for r in rs)
                lines.append(f'| {label} | {toks / len(cyc):.1f} | {np.mean(nov):.2f} | {proj} |')
        lines.append('')
    lines.append('DFlash rows use the fixed calibrated cycle cost, so their three projections coincide. r ranges: '
                 + ', '.join(f'{t}: {rs}' for t, rs in R.items()) + '.')
    (HERE / 'compare_Qwen3-4B.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
