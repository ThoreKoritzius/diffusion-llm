"""Summarize the H100 runs: measured baselines, calibrated cycle costs, projected hybrids, decision gates.

Usage: python analyze_h100.py [TARGET_TAG]   (default Qwen3-8B) -> results/<tag>/summary.md
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import HERE, cohort, group

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
GROUPS = ['function (CanItEdit)', 'function (EditPackFT)', 'full file']


def jl(p):
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def main():
    items = {c['id']: c for c in cohort()}
    V = {p.stem[5:]: jl(p) for p in sorted(RES.glob('vllm_*.jsonl'))}
    if 'ar' not in V:
        raise SystemExit('need results for the ar config first')
    AR = V['ar']
    step = {i: (r['latency_ms'] - r['ttft_ms']) / max(1, r['out_tokens'] - 1) for i, r in AR.items() if r['out_tokens'] > 8}
    L = [f'# GPU benchmark summary: {TAG} (GPU recorded in env.txt below)\n']
    env = RES / 'env.txt'
    if env.exists():
        L += ['```', env.read_text().strip(), '```\n']
    # 1. measured baselines
    L += ['## 1. Measured in vLLM (batch one, greedy)\n',
          '| Config | n | Speedup vs AR (end to end) | Decode-only speedup | Mean accepted length τ | Cycle cost (AR steps) | Output identical to AR |',
          '|---|---:|---:|---:|---:|---:|---:|']
    cost = {}
    for name, R in V.items():
        ids = [i for i in R if i in step]
        if not ids:
            continue
        e2e = sum(AR[i]['latency_ms'] for i in ids) / sum(R[i]['latency_ms'] for i in ids)
        dec = sum(AR[i]['latency_ms'] - AR[i]['ttft_ms'] for i in ids) / sum(R[i]['latency_ms'] - R[i]['ttft_ms'] for i in ids)
        drafts = sum(R[i]['drafts'] for i in ids)
        tau = 1 + sum(R[i]['accepted'] for i in ids) / drafts if drafts else 1.0
        cc = [((R[i]['latency_ms'] - R[i]['ttft_ms']) / R[i]['drafts']) / step[i] for i in ids if R[i]['drafts'] > 3]
        cost[name] = float(np.median(cc)) if cc else 1.0
        same = np.mean([R[i]['text'] == AR[i]['text'] for i in ids])
        L.append(f'| {name} | {len(ids)} | {e2e:.2f}× | {dec:.2f}× | {tau:.2f} | {cost[name]:.2f} | {same:.0%} |')
    L += ['\nPer group, end-to-end speedup vs AR:\n', '| Config | ' + ' | '.join(GROUPS) + ' |', '|---|' + '---:|' * len(GROUPS)]
    for name, R in V.items():
        cells = []
        for g in GROUPS:
            ids = [i for i in R if i in step and group(items[i]) == g]
            cells.append(f"{sum(AR[i]['latency_ms'] for i in ids) / sum(R[i]['latency_ms'] for i in ids):.2f}×" if ids else '–')
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    # 2. calibrated costs
    r = {}
    for d in ('draft06b7', 'draft17b7'):
        if d in cost and 'ngram7' in cost:
            r[d] = (cost[d] - cost['ngram7']) / 7
    L += ['\n## 2. Calibrated costs (median per edit, in plain AR decode steps)\n']
    L += [f'- {k}: {v:.2f} per cycle' for k, v in cost.items()]
    L += [f'- implied AR-drafter step cost r ({d}): {v:.3f} of a target step' for d, v in r.items()]
    # 3. projections for hybrids measured in the harness, using measured costs
    c_copy = cost.get('ngram31', cost.get('ngram15'))
    c_df = cost.get('dflash15')
    tree_extra = (cost['ngram31'] - cost['ngram15']) if {'ngram31', 'ngram15'} <= cost.keys() else 0.0
    L += ['\n## 3. Projected hybrids (acceptance from the PyTorch harness, costs from section 2)\n',
          '| Method | n | Tokens per target call | Accepted per draft on new code | Projected end-to-end speedup | Speedup vs best measured baseline |',
          '|---|---:|---:|---:|---:|---:|']
    best_name = max((n for n in V if n != 'ar'), key=lambda n: sum(AR[i]['latency_ms'] for i in V[n] if i in step) /
                    max(1e-9, sum(V[n][i]['latency_ms'] for i in V[n] if i in step)), default=None)

    def proj(rows, cyc_cost):
        ids = [i for i in rows if i in step]
        ar_t = sum(AR[i]['latency_ms'] for i in ids)
        new_t = sum(AR[i]['ttft_ms'] + step[i] * sum(cyc_cost(c) for c in rows[i]) for i in ids)
        best_t = sum(V[best_name][i]['latency_ms'] for i in ids if i in V[best_name]) if best_name else None
        return ids, ar_t / new_t, (best_t / new_t if best_t else float('nan'))

    rows_out = []
    if c_df and c_copy:
        for h, label, extra in (('hybrid_oracle', 'Copy + DFlash, two-branch tree', tree_extra), ('hybrid_router8', 'Copy + DFlash, router (deployable)', 0.0),
                                ('hybrid_router4', 'Copy + DFlash, router min 4', 0.0), ('hybrid_router16', 'Copy + DFlash, router min 16', 0.0), ('dflash', 'DFlash alone (harness)', 0.0)):
            H = jl(RES / f'harness_{h}.jsonl')
            H = {i: v for i, v in H.items() if not v.get('diverged')}
            if not H:
                continue
            cyc = {i: v['cycles'] for i, v in H.items()}
            ids, s, sb = proj(cyc, lambda c: (c_df + extra) if c['drafted'] else c_copy)
            tpc = sum(len(H[i]['out']) - 1 for i in ids) / sum(len(cyc[i]) for i in ids)
            from sim import align
            labs = {i: align(H[i]['prompt'], H[i]['out']) for i in ids}
            is_h = h.startswith('hybrid')
            nov = [(c['a_draft'] if is_h else c['acc']) for i in ids for c in cyc[i]
                   if c['drafted'] and labs[i][min(c['n'], len(labs[i]) - 1)] < 0 and (not is_h or c['a_draft'] >= 0)]
            rows_out.append((label, len(ids), tpc, np.mean(nov) if nov else float('nan'), s, sb))
        for drafter, rk in (('Qwen3-0.6B', 'draft06b7'), ('Qwen3-1.7B', 'draft17b7')):
            A = jl(RES / f'harness_ardraft_{drafter}.jsonl')
            if not A or rk not in r:
                continue
            for key, gamma in (('hybrid7', 7), ('hybrid15', 15)):
                base = cost.get('ngram7' if gamma == 7 else 'ngram15', c_copy)
                cyc = {i: v[key] for i, v in A.items()}
                ids, s, sb = proj(cyc, lambda c: (base + gamma * r[rk]) if c['drafted'] else c_copy)
                tpc = sum(A[i]['L'] - 1 for i in ids) / sum(len(cyc[i]) for i in ids)
                nov = [c['a_draft'] for i in ids for c in cyc[i] if c['drafted'] and c['novel']]
                rows_out.append((f'Copy + {drafter}, γ={gamma} (EfficientEdit-style)', len(ids), tpc, np.mean(nov), s, sb))
    for label, n, tpc, nov, s, sb in rows_out:
        L.append(f'| {label} | {n} | {tpc:.1f} | {nov:.2f} | {s:.2f}× | {sb:.2f}× |')
    L.append(f'\nBest measured baseline: {best_name}. Tree verification adds {tree_extra:.2f} AR steps per drafted cycle '
             '(ngram31 minus ngram15 cost).')
    # 4. gates
    L += ['\n## 4. Decision gates\n']
    dfl = {r_[0]: r_ for r_ in rows_out}
    g1 = 'dflash15' in cost and cost['dflash15'] <= 1.8
    L.append(f"- G1 cost model holds (DFlash cycle ≤ 1.8 AR steps; DFlash paper on B200: 1.55): {'PASS' if g1 else 'FAIL'} "
             f"({cost.get('dflash15', float('nan')):.2f})")
    hyb = dfl.get('Copy + DFlash, router (deployable)') or dfl.get('Copy + DFlash, two-branch tree')
    if hyb:
        L.append(f"- G2 deployable copy + DFlash ≥ 1.15× best measured baseline: {'PASS' if hyb[5] >= 1.15 else 'FAIL'} ({hyb[5]:.2f}×)")
    ar_h = [x for x in rows_out if 'EfficientEdit' in x[0]]
    if hyb and ar_h:
        best_ar = max(ar_h, key=lambda x: x[4])
        L.append(f"- G3 one-pass beats best AR-drafter hybrid at measured cost: {'PASS' if hyb[4] > best_ar[4] else 'FAIL'} "
                 f'({hyb[4]:.2f}× vs {best_ar[4]:.2f}× for {best_ar[0]})')
    (RES / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
