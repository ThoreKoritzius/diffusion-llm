"""Aggregate the copy-acceptance audit into ceilings, realistic drafters and a Mac latency model."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def boot(items, f, n=2000):
    idx = np.arange(len(items))
    vals = [f([items[j] for j in rng.choice(idx, len(idx))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def cost_model():
    rows = json.loads((HERE / 'cost.json').read_text())
    ws = sorted({r['w'] for r in rows})
    med = [float(np.median([r['ms'] for r in rows if r['w'] == w])) for w in ws]
    def c(w):
        return float(np.interp(w, ws, med)) if w <= ws[-1] else med[-1] + (w - ws[-1]) * (med[-1] - med[-2]) / (ws[-1] - ws[-2])
    return c, dict(zip(ws, med))


def noop_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')
    cohort = {c['id']: c for c in json.loads((HERE / 'cohort.json').read_text())}
    norm = lambda s: '\n'.join(l.rstrip() for l in s.strip().splitlines())
    out = set()
    for l in (HERE / 'targets.jsonl').open():
        r = json.loads(l)
        text = tok.decode(r['out_ids'], skip_special_tokens=True).split('```')[0]
        if norm(text) == norm(cohort[r['id']]['old']):
            out.add(r['id'])
    return out


def groups(res):
    g = {'function (CanItEdit)': [r for r in res if r['regime'] == 'function' and r['id'].startswith('cie')],
         'function (EditPackFT)': [r for r in res if r['source'] == 'editpackft'],
         'full file (CanItEdit)': [r for r in res if r['regime'] == 'file']}
    return g


def main():
    res = [r for r in json.loads((HERE / 'sim.json').read_text())]
    n_limit = sum(r['stop'] == 'limit' for r in res)
    res = [r for r in res if r['stop'] != 'limit']  # degenerate repetition until the cap
    c, table = cost_model()
    lines = [f'# Copy-acceptance audit results\n',
             f'Target Qwen2.5-Coder-1.5B-Instruct, greedy FP16, MPS, batch one. {len(res)} edits kept; '
             f'{n_limit} dropped because the target hit the generation cap (repetition).\n',
             '## Verification cost per target call (median ms, cached context)\n',
             '| Width | ' + ' | '.join(str(w) for w in table) + ' |', '|---' * (len(table) + 1) + '|',
             '| ms | ' + ' | '.join(f'{v:.1f}' for v in table.values()) + ' |\n']
    pols = [('ar', 'Greedy AR'), ('ngram32', 'Prompt lookup (3-gram, K=32)'), ('suffix32', 'Suffix-match copy (K=32)'),
            ('suffix64', 'Suffix-match copy (K=64)'),
            ('copy_best32', 'Copy ceiling: perfect resync (K=32)'),
            ('oracle_fill4_32', 'Copy + perfect 4-token filler on edited spans'),
            ('oracle_fill8_32', 'Copy + perfect 8-token filler'),
            ('oracle_fill16_32', 'Copy + perfect 16-token filler'),
            ('oracle_fill32_32', 'Copy + perfect 32-token filler'),
            ('perfect32', 'Perfect drafter, K=32 everywhere')]
    def lat(r, p):
        return r['prefill_ms'] + sum(c(w) for w in r[p]['widths'])
    for name, items in groups(res).items():
        L = np.array([r['L'] for r in items])
        nov = np.array([r['novel'] / r['L'] for r in items])
        lines += [f'## {name}: n={len(items)}, median output {int(np.median(L))} tokens, '
                  f'median edited-token share {np.median(nov):.0%}\n',
                  '| Policy | Tokens per target call | 95% CI | Call speedup (median item) | Share of calls on edited spans | Modeled Mac latency speedup |',
                  '|---|---:|---:|---:|---:|---:|']
        ar_lat = sum(lat(r, 'ar') for r in items)
        for p, label in pols:
            tpc = lambda its: sum(r['L'] - 1 for r in its) / sum(r[p]['calls'] for r in its)
            lo, hi = boot(items, tpc)
            med = np.median([(r['L'] - 1) / r[p]['calls'] for r in items])
            share = sum(r[p]['novel_calls'] for r in items) / sum(r[p]['calls'] for r in items)
            lines.append(f'| {label} | {tpc(items):.2f} | {lo:.2f}–{hi:.2f} | {med:.2f}× | {share:.0%} | '
                         f'{ar_lat / sum(lat(r, p) for r in items):.2f}× |')
        meas = sum(r['ar_ms'] for r in items) / ar_lat
        lines.append(f'\nMeasured AR wall time / modeled AR time: {meas:.2f}. '
                     f'Latency includes prefill; speedups are ratios of total time over the group.\n')
    # Robustness: drop outputs that reproduce the old code unchanged (target failed to edit)
    noop = noop_ids()
    sel = [('suffix32', 'Suffix-match copy'), ('copy_best32', 'Perfect resync'),
           ('oracle_fill8_32', '+ perfect 8-token filler'), ('oracle_fill16_32', '+ perfect 16-token filler'),
           ('perfect32', 'Perfect drafter')]
    lines += ['## Actual edits only (target output differs from the old code)\n',
              '| Group | n | ' + ' | '.join(l for _, l in sel) + ' |', '|---|---:|' + '---:|' * len(sel)]
    for name, items in groups(res).items():
        its = [r for r in items if r['id'] not in noop]
        ar_lat = sum(lat(r, 'ar') for r in its)
        cells = [f"{sum(r['L'] - 1 for r in its) / sum(r[p]['calls'] for r in its):.1f} tok/call, "
                 f"{ar_lat / sum(lat(r, p) for r in its):.1f}× Mac" for p, _ in sel]
        lines.append(f'| {name} | {len(its)} | ' + ' | '.join(cells) + ' |')
    lines.append(f'\n{len(noop)} of {len(res)} target outputs reproduced the old code unchanged.\n')
    # Part A: real drafters
    pA = HERE / 'draft_A.json'
    if pA.exists():
        A = json.loads(pA.read_text())
        t = A['timing']
        simd = {r['id']: r for r in res}
        items = [a for a in A['items'] if a['id'] in simd]
        lines += ['## Real drafter: Qwen2.5-Coder-0.5B-Instruct (16-token drafts)\n',
                  f"Drafter forward cost: 1 token {t['forward_1_ms']:.1f} ms, 16 tokens {t['forward_16_ms']:.1f} ms. "
                  'AR drafting charges 16 sequential steps; the one-pass projection charges one 16-token forward at the '
                  'same acceptance (an assumption about a future diffusion drafter, not a measurement).\n',
                  '| Group | Policy | Tokens per target call | Accepted on edited-span cycles | Mac model, AR drafting | Mac model, one-pass drafting | Flat verify, one pass = 5% of target step | = 20% |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for gname, its in [('function', [a for a in items if a['regime'] == 'function']),
                           ('full file', [a for a in items if a['regime'] == 'file'])]:
            ar_lat = sum(lat(simd[a['id']], 'ar') for a in its)
            for p in ('ar_draft', 'hybrid'):
                calls = sum(len(a[p]) for a in its)
                tpc = sum(a['L'] - 1 for a in its) / calls
                nov = [cy['a'] for a in its for cy in a[p] if cy['region'] == 'novel']
                def L_(draft_cost):
                    return sum(simd[a['id']]['prefill_ms'] + sum(c(cy['width']) + (draft_cost if cy['drafted'] else 0)
                                                                for cy in a[p]) for a in its)
                ar_d = t['forward_16_ms'] + 16 * t['forward_1_ms']
                one = 2 * t['forward_16_ms']
                flat = lambda rr: sum(a['L'] - 1 for a in its) / sum(1 + rr * cy['drafted'] for a in its for cy in a[p])
                lines.append(f"| {gname} (n={len(its)}) | {p} | {tpc:.2f} | {np.mean(nov):.2f} | "
                             f'{ar_lat / L_(ar_d):.2f}× | {ar_lat / L_(one):.2f}× | {flat(0.05):.1f}× | {flat(0.2):.1f}× |')
            sx = sum(simd[a['id']]['suffix32']['calls'] for a in its)
            sx_t = sum(a['L'] - 1 for a in its) / sx
            lines.append(f"| {gname} | suffix32 (reference) | {sum(a['L'] - 1 for a in its) / sx:.2f} | – | "
                         f"{ar_lat / sum(lat(simd[a['id']], 'suffix32') for a in its):.2f}× | – | {sx_t:.1f}× | {sx_t:.1f}× |")
        lines.append('')
    pB = HERE / 'draft_B.json'
    if pB.exists():
        B = json.loads(pB.read_text())
        keep = {r['id'] for r in res}
        B = [b for b in B if b['id'] in keep and b['points']]
        pts = [p for b in B for p in b['points']]
        lines += ['## Right-context ablation on edited spans (Qwen2.5-Coder-0.5B base, FIM, function regime)\n',
                  f'{len(pts)} edited-span cycle starts in {len(B)} edits. Accepted tokens per draft (max 16).\n',
                  '| Suffix given to the drafter | Mean accepted | 95% CI (by edit) | ≥4 accepted | Better than no suffix | Worse than no suffix |',
                  '|---|---:|---:|---:|---:|---:|']
        for k, label in [('fim_none', 'None'), ('fim_real', 'Remaining old source (deployable)'),
                         ('fim_oracle', 'Old source from the true next anchor (oracle)')]:
            f = lambda bs: np.mean([p[k] for b in bs for p in b['points']])
            lo, hi = boot(B, f)
            lines.append(f"| {label} | {f(B):.2f} | {lo:.2f}–{hi:.2f} | {np.mean([p[k] >= 4 for p in pts]):.0%} | "
                         f"{np.mean([p[k] > p['fim_none'] for p in pts]):.0%} | {np.mean([p[k] < p['fim_none'] for p in pts]):.0%} |")
        lines.append('')
    (HERE / 'results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
