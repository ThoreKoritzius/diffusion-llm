"""Speedup as a curve, not a single number: by copy share and by output length (all measured in vLLM).

Copy share of a request = share of the characters of the plain-AR output that lie in lines copied verbatim from the
input file (line-level difflib match), i.e. how much of the answer is unchanged code. Each point = total AR latency /
total method latency over the requests in that bin (pooled over all cohorts present), with a bootstrap 95% CI band.

  figures/curves_<key>.png/.pdf   paper: (a) speedup vs copy share, (b) speedup vs output length
  figures/curve_social_<key>.png  one plain-language panel (speedup vs share of the file left unchanged)
  results/<key>/curves.json       the binned numbers
Usage: python curves.py [KEY]
"""
import difflib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from common import HERE, cohort

KEY = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-8b'
RES, OUT = HERE / 'results' / KEY, HERE / 'figures'
COHORTS = ['edit261', 'ceb150', 'diff104', 'apply100']
SURFACE, TEXT, TEXT2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
# validated categorical palette (blue / orange / aqua), fixed order: existing methods, ours, copy-only
METHODS = [('ngram63', 'Copy only (prompt lookup, 63)', '#1baf7a'), ('ngram31', 'Copy only (prompt lookup, 31)', '#1baf7a'),
           ('dflash', 'DFlash', '#2a78d6'), ('copyflash63', 'Copy + DFlash (ours)', '#eb6834')]
SHARE_BINS = [(0, .5, '<50%'), (.5, .8, '50–80%'), (.8, .9, '80–90%'), (.9, .95, '90–95%'), (.95, .98, '95–98%'),
              (.98, 1.01, '≥98%')]
LEN_BINS = [(0, 100, '<100'), (100, 200, '100–200'), (200, 400, '200–400'), (400, 800, '400–800'),
            (800, 1600, '800–1.6k'), (1600, 10 ** 6, '≥1.6k')]
NAMES = {'qwen3-8b': 'Qwen3-8B', 'qwen3-4b': 'Qwen3-4B', 'llama31-8b': 'Llama-3.1-8B', 'qwen3coder-30b': 'Qwen3-Coder-30B-A3B'}


def load(c, m):
    p = RES / c / f'vllm_{m}.jsonl'
    return {r['id']: r for r in map(json.loads, p.open())} if p.exists() else {}


def copy_share(old, out):
    a, b = old.splitlines(), out.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    copied = sum(len(line) + 1 for blk in sm.get_matching_blocks() for line in b[blk.b:blk.b + blk.size])
    return copied / max(1, sum(len(line) + 1 for line in b))


def requests():
    rows = []
    for c in COHORTS:
        A = load(c, 'ar')
        if not A:
            continue
        items = {it['id']: it for it in cohort(c)}
        lat = {m: load(c, m) for m, _, _ in METHODS}
        for i, r in A.items():
            row = dict(id=i, cohort=c, out=r['out_tokens'], share=copy_share(items[i]['old'], r['text']), ar=r['latency_ms'])
            for m in lat:
                if i in lat[m]:
                    row[m] = lat[m][i]['latency_ms']
            rows.append(row)
    return rows


def binned(rows, key, bins, method, boot=4000):
    out = []
    rng = np.random.default_rng(0)
    for lo, hi, label in bins:
        R = [r for r in rows if lo <= r[key] < hi and method in r]
        if len(R) < 5:
            out.append(None); continue
        num, den = np.array([r['ar'] for r in R]), np.array([r[method] for r in R])
        idx = rng.integers(0, len(R), (boot, len(R)))
        bs = num[idx].sum(1) / den[idx].sum(1)
        out.append(dict(bin=label, n=len(R), speedup=float(num.sum() / den.sum()), lo=float(np.percentile(bs, 2.5)),
                        hi=float(np.percentile(bs, 97.5))))
    return out


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ('Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans') if f in have), 'DejaVu Sans')


def methods_present(rows):
    have = [m for m in METHODS if any(m[0] in r for r in rows)]
    if any(m[0] == 'ngram63' for m in have):  # one copy-only line: the widest window measured
        have = [m for m in have if m[0] != 'ngram31']
    return have


def panel(ax, rows, key, bins, title, xlabel, methods, ymax):
    bins = [b for b in bins if sum(b[0] <= r[key] < b[1] for r in rows) >= 5]  # no empty bins on the axis
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, colors=TEXT2)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    x = np.arange(len(bins))
    res = {}
    for m, label, color in methods:
        b = binned(rows, key, bins, m)
        res[m] = b
        pts = [(xi, v) for xi, v in zip(x, b) if v]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.fill_between(xs, [p[1]['lo'] for p in pts], [p[1]['hi'] for p in pts], color=color, alpha=0.14, lw=0)
        ax.plot(xs, [p[1]['speedup'] for p in pts], color=color, lw=2, zorder=3)
        ax.plot(xs, [p[1]['speedup'] for p in pts], 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=4, label=label)
        last = pts[-1]
        ax.text(last[0] + 0.12, last[1]['speedup'], f"{last[1]['speedup']:.1f}×", va='center', ha='left', fontsize=10,
                color=TEXT, fontweight='bold' if m == 'copyflash63' else 'normal')
    ns = [max((v['n'] for b in res.values() if (v := b[i])), default=0) for i in range(len(bins))]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lab}\nn = {n}' for (_, _, lab), n in zip(bins, ns)], fontsize=9.5, color=TEXT2)
    ax.set_xlim(-0.4, len(bins) - 0.4)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f'{t:.0f}×'))
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=10.5, labelpad=8)
    ax.set_title(title, loc='left', color=TEXT, fontsize=12, pad=12)
    return res


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': font()})
    rows = requests()
    methods = methods_present(rows)
    cohorts = [c for c in COHORTS if any(r['cohort'] == c for r in rows)]
    ymax = max(v['hi'] for m, _, _ in methods for key, bins in (('share', SHARE_BINS), ('out', LEN_BINS))
               for v in binned(rows, key, bins, m) if v) * 1.12
    fig = plt.figure(figsize=(13.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, wspace=0.16, left=0.055, right=0.965, top=0.74, bottom=0.17)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    ra = panel(a, rows, 'share', SHARE_BINS, '(a) By how much of the output is unchanged code',
               'Share of the output copied verbatim from the input', methods, ymax)
    rb = panel(b, rows, 'out', LEN_BINS, '(b) By output length', 'Output length (tokens)', methods, ymax)
    a.set_ylabel('Speedup over plain AR (measured)', color=TEXT2, fontsize=10.5)
    a.legend(loc='upper left', frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=1, fontsize=9.5, labelcolor=TEXT2)
    n = len(rows)
    fig.text(0.012, 0.955, 'The speedup depends on how much of the file stays the same, and grows with its length',
             fontsize=13.5, fontweight='bold', color=TEXT, ha='left')
    fig.text(0.012, 0.9, f'{NAMES[KEY]}, H100, vLLM, batch 1, greedy. {n} requests pooled over {", ".join(cohorts)}. '
             'Each point: total plain-AR latency / total method latency in the bin; band: 95% bootstrap CI.',
             fontsize=9.5, color=TEXT2, ha='left')
    fig.text(0.012, 0.862, 'Copy share is measured on the plain-AR output (lines identical to lines of the input, weighted by '
             'length); the same split applies to every method.', fontsize=9.5, color=TEXT2, ha='left')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'curves_{KEY}.{ext}', dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # social: one panel, plain words
    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURFACE)
    fig.text(0.055, 0.905, 'The less code changes, the bigger the speed-up', fontsize=28, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.85, f'Speed-up over standard decoding by how much of the file an edit leaves unchanged '
             f'({NAMES[KEY]}, {n} real edits, one H100).', fontsize=14, color=TEXT2)
    ax = fig.add_axes([0.075, 0.2, 0.86, 0.57])
    plt.rcParams.update({'font.size': 12})
    panel(ax, rows, 'share', SHARE_BINS, '', 'Share of the output that is unchanged code', methods, ymax)
    ax.legend(loc='upper left', frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=1, fontsize=12.5, labelcolor=TEXT2)
    fig.text(0.055, 0.045, 'All measured end to end in vLLM. Each point pools the edits in its range (n below each label); '
             'shaded: 95% CI. Outputs match standard decoding up to floating-point ties.', fontsize=10, color=MUTED)
    fig.savefig(OUT / f'curve_social_{KEY}.png', dpi=200, facecolor=SURFACE)
    plt.close(fig)
    (RES / 'curves.json').write_text(json.dumps({'copy_share': ra, 'output_length': rb}, indent=1))
    for name, res in (('copy share', ra), ('output length', rb)):
        print(f'\nby {name}')
        for m, b in res.items():
            print(f'  {m:>12}: ' + '  '.join(f"{v['bin']} {v['speedup']:.1f}x(n={v['n']})" for v in b if v))


if __name__ == '__main__':
    main()
