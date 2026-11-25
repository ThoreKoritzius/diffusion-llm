"""Cross-target summary of the generalization run -> results/summary.md

Per (target, cohort): measured vLLM speedups, tau and cycle costs; measured harness gain of copy + DFlash (router)
over DFlash with a bootstrap CI; projected hybrid speedup from measured costs; DFlash acceptance on new code.
Then the pre-registered predictions (predictions.json) against the measurements.
"""
import json

import numpy as np

from common import COHORTS, HERE, cohort, targets
from predict import labels

rng = np.random.default_rng(0)
RES = HERE / 'results'


def jl(p):
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def ratio_ci(num, den, n=4000):
    num, den = np.array(num), np.array(den)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def main():
    pred = json.loads((HERE / 'predictions.json').read_text())
    L = ['# Generalization run: summary\n']
    pre_rows, meas = [], {}
    for key in targets():
        for c in COHORTS:
            d = RES / key / c
            AR = jl(d / 'vllm_ar.jsonl')
            if not AR:
                continue
            V = {p.stem[5:]: jl(p) for p in sorted(d.glob('vllm_*.jsonl'))}
            step = {i: (r['latency_ms'] - r['ttft_ms']) / (r['out_tokens'] - 1) for i, r in AR.items() if r['out_tokens'] > 8}
            row, cost = {}, {}
            for m, R in V.items():
                ids = [i for i in R if i in step]
                if not ids or m == 'ar':
                    continue
                spd = sum(AR[i]['latency_ms'] for i in ids) / sum(R[i]['latency_ms'] for i in ids)
                drafts = sum(R[i]['drafts'] for i in ids)
                tpc = sum(R[i]['out_tokens'] for i in ids) / max(1, drafts)
                cost[m] = float(np.median([(R[i]['latency_ms'] - R[i]['ttft_ms']) / R[i]['drafts'] / step[i] for i in ids if R[i]['drafts'] > 3] or [np.nan]))
                same = np.mean([R[i]['text'] == AR[i]['text'] for i in ids])
                row[m] = dict(speedup=spd, tpc=tpc, cost=cost[m], same=same, n=len(ids))
            D, Hr = jl(d / 'harness_dflash.jsonl'), jl(d / 'harness_hybrid_router8.jsonl')
            gain = proj = nov = None
            if D and Hr:
                ids = [i for i in Hr if i in D]
                gain = ratio_ci([D[i]['total_ms'] for i in ids], [Hr[i]['total_ms'] for i in ids])
                if 'dflash' in cost and 'ngram31' in cost:
                    ids2 = [i for i in ids if i in step]
                    new_t = sum(AR[i]['ttft_ms'] + step[i] * sum(cost['dflash'] if cy['drafted'] else cost['ngram31'] for cy in Hr[i]['cycles']) for i in ids2)
                    proj = sum(AR[i]['latency_ms'] for i in ids2) / new_t
                acc = []
                for i, r in D.items():
                    lab = labels(r['prompt'], r['out'])
                    acc += [cy['acc'] for cy in r['cycles'] if lab[min(cy['n'], len(lab) - 1)] < 0]
                nov = float(np.mean(acc)) if acc else None
            meas[(key, c)] = dict(row=row, gain=gain, proj=proj, nov=nov)
            p = pred['predictions'].get(f'{key}/{c}')
            if p:
                for m in ('ngram31', 'draft7', 'eagle7', 'dflash'):
                    if m in row and m in p:
                        pre_rows.append((key, c, p['kind'], m, p[m]['tokens_per_call'], row[m]['tpc'], p[m]['speedup_a100_costs'], row[m]['speedup']))
    methods = ['ngram31', 'eagle7', 'draft7', 'dflash']
    L += ['## 1. Measured end-to-end speedup over plain AR (vLLM, batch one, greedy)\n',
          '| Target | Cohort | ' + ' | '.join(methods) + ' | Copy + DFlash projected | Router vs DFlash, measured (95% CI) | DFlash accepted/guess on new code |',
          '|---|---|' + '---:|' * (len(methods) + 3)]
    for (key, c), v in meas.items():
        cells = [f"{v['row'][m]['speedup']:.2f}×" if m in v['row'] else '–' for m in methods]
        g = f"{v['gain'][0]:.2f}× ({v['gain'][1]:.2f}–{v['gain'][2]:.2f})" if v['gain'] else '–'
        pj = f"{v['proj']:.2f}×" if v['proj'] else '–'
        nv = f"{v['nov']:.2f}" if v['nov'] is not None else '–'
        L.append(f"| {key} | {c} | " + ' | '.join(cells) + f" | {pj} | {g} | {nv} |")
    L += ['\n## 2. Cycle costs (AR steps) and agreement with plain AR\n', '| Target | Cohort | ' + ' | '.join(methods) + ' | Output identical to AR (dflash) |',
          '|---|---|' + '---:|' * (len(methods) + 1)]
    for (key, c), v in meas.items():
        same = f"{v['row']['dflash']['same']:.0%}" if 'dflash' in v['row'] else '–'
        L.append(f"| {key} | {c} | " + ' | '.join(f"{v['row'][m]['cost']:.2f}" if m in v['row'] else '–' for m in methods) + f" | {same} |")
    L += ['\n## 3. Pre-registered predictions vs measured\n',
          f"Predictions created {pred['created_utc']} (sha256 {pred['sha256'][:12]}…), before any decoding of these cohorts. "
          'Speedup predictions use A100 cost constants; the GPU here may differ, so tokens per call is the hardware-independent test.\n',
          '| Target | Cohort | Kind | Method | Tokens/call predicted | measured | error | Speedup predicted (A100 costs) | measured |',
          '|---|---|---|---|---:|---:|---:|---:|---:|']
    errs = {}
    for key, c, kind, m, tp, tm, sp, sm in pre_rows:
        e = (tp - tm) / tm
        errs.setdefault(kind, []).append(abs(e))
        L.append(f'| {key} | {c} | {kind} | {m} | {tp:.1f} | {tm:.1f} | {e:+.0%} | {sp:.2f}× | {sm:.2f}× |')
    for kind, e in errs.items():
        L.append(f'\nMedian absolute tokens-per-call error, {kind}: {np.median(e):.0%} over {len(e)} predictions.')
    (RES / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
