"""Figures from the H100 vLLM runs: every bar is measured end to end (no projections).

  figures/hero_<key>.png/.pdf     paper figure: (a) speedup over AR on edit261, (b) paired gain over DFlash, 95% CI
  figures/social_<key>.png        16:9 plain-language chart
  figures/explainer_<key>.png     idea (LLM passes) on top, chart below

Speedup = total AR latency / total method latency over the requests both have (prefill included), bootstrap 95% CI
over requests. Usage: python figures_h100.py [KEY]   (default qwen3-8b)
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from common import HERE

KEY = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-8b'
RES = HERE / 'results' / KEY
OUT = HERE / 'figures'
SURFACE, TEXT, TEXT2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
EXIST, OURS = '#2a78d6', '#eb6834'
COPY, COPY_BG, DRAFT, DRAFT_BG, PASS = '#bdbbb4', '#efeee9', '#1baf7a', '#d6f2e7', '#6f6e69'
MONO = 'Menlo'
TARGET_NAME = {'qwen3-8b': 'Qwen3-8B', 'qwen3-4b': 'Qwen3-4B', 'llama31-8b': 'Llama-3.1-8B',
               'qwen3coder-30b': 'Qwen3-Coder-30B-A3B'}[KEY]


def load(cohort, name):
    p = RES / cohort / f'vllm_{name}.jsonl'
    return {r['id']: r for r in map(json.loads, p.open())} if p.exists() else {}


def ratio_ci(num, den, n=4000):
    num, den = np.array(num), np.array(den)
    idx = np.random.default_rng(0).integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def speedup(cohort, name, base='ar'):
    A, R = load(cohort, base), load(cohort, name)
    ids = [i for i in R if i in A]
    est, lo, hi = ratio_ci([A[i]['latency_ms'] for i in ids], [R[i]['latency_ms'] for i in ids])
    return est, lo, hi, len(ids)


def tokens_per_call(cohort, name):
    R = load(cohort, name)
    d = sum(r['drafts'] for r in R.values())
    return (sum(r['accepted'] for r in R.values()) + d) / d


def same_as_ar(cohort, name):
    A, R = load(cohort, 'ar'), load(cohort, name)
    ids = [i for i in R if i in A]
    return sum(A[i]['text'] == R[i]['text'] for i in ids) / len(ids)


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ('Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans') if f in have), 'DejaVu Sans')


def bar(ax, y, x, h, color, aspect=2.2):
    """Bar from 0 to x: square at the baseline, rounded data end."""
    r = min(0.12, x / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), x, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                mutation_aspect=aspect, fc=color, ec='none'))
    ax.add_patch(Rectangle((0, y - h / 2), x - r, h, fc=color, ec='none'))


METHODS = [  # vLLM name, plain label, how it works, ours
    ('draft7', 'Small draft model', 'a 0.6B model guesses ahead', False),
    ('eagle7', 'EAGLE-3', 'a light head guesses ahead', False),
    ('ngram31', 'Copy from the old code', 'reuses unchanged lines', False),
    ('dflash', 'DFlash', 'diffusion drafts 16 tokens at once', False),
    ('copyflash63', 'Copy + DFlash (ours)', 'copy old lines, diffuse new code', True),
]


def chart_rows():
    rows = [('Standard decoding', 1.0, 'one token per LLM pass', False)]
    for name, label, how, ours in METHODS:
        if load('edit261', name):
            rows.append((label, speedup('edit261', name)[0], how, ours))
    return rows


def draw_chart(ax, rows, big):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.7, len(rows) - 0.35)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for y, (name, v, how, ours) in enumerate(rows):
        bar(ax, y, v, 0.57, OURS if ours else EXIST)
        ax.text(v + 0.14, y + (0.06 if ours else 0), f'{v:.1f}×', va='center', ha='left',
                fontsize=(21 if ours else 17) * big, fontweight='bold', color=TEXT)
        ax.text(-0.25, y + 0.14, name, va='center', ha='right', fontsize=15.5 * big, color=TEXT,
                fontweight='bold' if ours else 'normal')
        ax.text(-0.25, y - 0.2, how, va='center', ha='right', fontsize=11.5 * big, color=MUTED)


def footnotes(fig, y0, gap, size=10):
    v, lo, hi, n = speedup('edit261', 'copyflash63')
    g, glo, ghi, _ = speedup('edit261', 'copyflash63', base='dflash')
    fig.text(0.055, y0, f'All bars measured end to end in vLLM on {n} edits. Ours: {v:.2f}× (95% CI {lo:.1f}–{hi:.1f}), '
             f'{g:.2f}× faster than DFlash alone on the same edits (CI {glo:.2f}–{ghi:.2f}).', fontsize=size, color=MUTED)
    fig.text(0.055, y0 - gap, 'Greedy decoding, batch 1; every token is checked by the LLM, '
             'so outputs match standard decoding up to floating-point ties.', fontsize=size, color=MUTED)


def social():
    rows = chart_rows()
    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURFACE)
    fig.text(0.055, 0.905, 'LLMs can edit code 9× faster, with the same output', fontsize=30, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.845, f'Speed-up over standard decoding when an LLM rewrites code ({TARGET_NAME}, 261 real edits, one H100).',
             fontsize=14.5, color=TEXT2)
    fig.text(0.055, 0.805, 'Unchanged lines are copied from the old code, new code is drafted by a diffusion model, '
             'and the LLM checks every token.', fontsize=14.5, color=TEXT2)
    draw_chart(fig.add_axes([0.255, 0.13, 0.67, 0.63]), rows, 1.0)
    footnotes(fig, 0.065, 0.032)
    fig.savefig(OUT / f'social_{KEY}.png', dpi=200, facecolor=SURFACE)
    plt.close(fig)


NEW = [('def average(values):', 'copy'),
       ('    values = [v for v in values if v is not None]', 'draft'),
       ('    if not values:', 'draft'),
       ('        return 0', 'draft'),
       ('    total = sum(values)', 'copy'),
       ('    return total / len(values)', 'copy')]
TOKENS = 42  # Qwen3 tokens of the six lines above
CHUNKS = [('copy', 4), ('draft', 16), ('draft', 9), ('copy', 13)]  # one LLM pass each (idealised: all accepted)


def chip(fig, x, y, color, label):
    fig.patches.append(FancyBboxPatch((x, y - 0.006), 0.018, 0.012, boxstyle='round,pad=0,rounding_size=0.003',
                                      transform=fig.transFigure, fc=color, ec='none'))
    fig.text(x + 0.026, y, label, fontsize=13, color=TEXT2, va='center')


def strip(ax, y, segments, h, gap):
    x = 0.0
    for n, c in segments:
        w = n / TOKENS
        ax.add_patch(FancyBboxPatch((x + gap / 2, y), w - gap, h, boxstyle='round,pad=0,rounding_size=0.004',
                                    mutation_aspect=4, fc=c, ec='none'))
        x += w


def explainer():
    rows = chart_rows()
    tpc = tokens_per_call('edit261', 'copyflash63')
    fig = plt.figure(figsize=(12, 13.5), dpi=200, facecolor=SURFACE)
    fig.text(0.055, 0.955, 'Editing code 9× faster, with the same output', fontsize=30, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.925, 'Most of an edited file stays the same. Copy those lines, let a diffusion model draft only the new ones,',
             fontsize=14.5, color=TEXT2)
    fig.text(0.055, 0.906, 'and have the LLM check whole chunks at once instead of writing one token at a time.', fontsize=14.5, color=TEXT2)
    fig.text(0.055, 0.858, 'How it works', fontsize=17, fontweight='bold', color=TEXT)
    chip(fig, 0.055, 0.83, COPY, 'copied from the old code')
    chip(fig, 0.33, 0.83, DRAFT, 'new code, drafted by diffusion')
    code = fig.add_axes([0.055, 0.61, 0.5, 0.2]); code.axis('off'); code.set_xlim(0, 1); code.set_ylim(0, 1)
    step = 1 / len(NEW)
    for k, (line, src) in enumerate(NEW):
        y = 1 - (k + 1) * step
        code.add_patch(Rectangle((0, y + 0.012), 1, step - 0.024, fc=COPY_BG if src == 'copy' else DRAFT_BG, ec='none'))
        if src == 'draft':
            code.add_patch(Rectangle((0, y + 0.012), 0.008, step - 0.024, fc=DRAFT, ec='none'))
        code.text(0.03, y + step / 2, line, family=MONO, fontsize=11.2, color=TEXT, va='center')
    pas = fig.add_axes([0.6, 0.61, 0.345, 0.2]); pas.axis('off'); pas.set_xlim(0, 1); pas.set_ylim(0, 1)
    pas.text(0, 0.97, 'LLM passes to write this function', fontsize=13.5, fontweight='bold', color=TEXT, va='top')
    pas.text(0, 0.78, 'Standard decoding', fontsize=12.5, color=TEXT2, va='bottom')
    pas.text(1, 0.78, f'{TOKENS} passes', fontsize=12.5, color=TEXT, fontweight='bold', va='bottom', ha='right')
    strip(pas, 0.6, [(1, PASS)] * TOKENS, 0.15, 0.006)
    pas.text(0, 0.36, 'Copy + DFlash', fontsize=12.5, color=TEXT2, va='bottom')
    pas.text(1, 0.36, f'{len(CHUNKS)} passes', fontsize=12.5, color=TEXT, fontweight='bold', va='bottom', ha='right')
    strip(pas, 0.18, [(n, COPY if s == 'copy' else DRAFT) for s, n in CHUNKS], 0.15, 0.012)
    pas.text(0, 0.02, 'one block = one pass of the LLM', fontsize=11, color=MUTED, va='bottom')
    fig.text(0.055, 0.578, 'The LLM checks each chunk in a single pass and keeps only what it would have written itself, so the output is unchanged.',
             fontsize=12.5, color=TEXT2)
    fig.text(0.055, 0.558, f'Idealised example (every guess accepted). On the 261 real edits the LLM keeps {tpc:.0f} tokens per pass on average.',
             fontsize=11, color=MUTED)
    fig.text(0.055, 0.5, 'Speed-up over standard decoding', fontsize=17, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.481, f'{TARGET_NAME}, 261 real code edits, one H100. Every bar measured end to end in vLLM.',
             fontsize=12.5, color=TEXT2)
    draw_chart(fig.add_axes([0.255, 0.085, 0.67, 0.375]), rows, 0.97)
    footnotes(fig, 0.047, 0.018)
    fig.savefig(OUT / f'explainer_{KEY}.png', dpi=200, facecolor=SURFACE)
    plt.close(fig)


def hero():
    rows = [r for r in chart_rows()[1:]]
    labels = {'Small draft model': 'Qwen3-0.6B draft model, γ = 7', 'EAGLE-3': 'EAGLE-3, 7 tokens',
              'Copy from the old code': 'Prompt lookup (copy), 31 tokens', 'DFlash': 'DFlash (one-pass diffusion)',
              'Copy + DFlash (ours)': 'Copy + DFlash router, 63-token window'}
    if load('edit261', 'copyflash'):
        rows.insert(len(rows) - 1, ('Copy + DFlash router, 31-token window', speedup('edit261', 'copyflash')[0], '', True))
    rows = [(labels.get(r[0], r[0]),) + r[1:] for r in rows]
    gains = []
    for cohort, cname in (('edit261', 'Code edits'), ('ceb150', 'Bug fixes'), ('diff104', 'Diff output')):
        for m, w in (('copyflash', 31), ('copyflash47', 47), ('copyflash63', 63)):
            if load(cohort, m) and load(cohort, 'dflash'):
                g, lo, hi, n = speedup(cohort, m, base='dflash')
                gains.append((f'{cname} ({n}), {w}-token window', g, lo, hi))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': GRID,
                         'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT})
    fig = plt.figure(figsize=(13.6, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.62, left=0.235, right=0.975, top=0.74, bottom=0.13)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ('top', 'right', 'left'):
            a.spines[s].set_visible(False)
        a.spines['bottom'].set_color(GRID)
        a.tick_params(axis='y', length=0)
        a.tick_params(axis='x', color=GRID)
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(0, 11, 2)); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for y, (label, v, _, ours) in enumerate(rows):
        bar(ax, y, v, 0.56, OURS if ours else EXIST, aspect=0.9)
        ax.text(v + 0.15, y, f'{v:.2f}×', va='center', ha='left', color=TEXT, fontsize=9.5,
                fontweight='bold' if ours else 'normal')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[3]:
            t.set_fontweight('bold')
    ax.axvline(1, color=TEXT2, lw=0.8)
    ax.set_xlabel('End-to-end speedup over plain autoregressive decoding (measured)')
    ax.set_title('(a) Speedup over plain AR on 261 real code edits', loc='left', color=TEXT, fontsize=11, pad=10)
    ax.legend(handles=[Patch(fc=EXIST, ec='none', label='Existing method'),
                       Patch(fc=OURS, ec='none', label='Copy + DFlash (this work)')],
              loc='lower right', frameon=False, fontsize=8.5, labelcolor=TEXT2)
    lo_x = min(0.95, min(g[2] for g in gains) - 0.03)
    bx.set_xlim(lo_x, max(g[3] for g in gains) + 0.2); bx.set_ylim(-0.6, len(gains) - 0.4)
    bx.xaxis.grid(True, color=GRID, lw=0.8); bx.set_axisbelow(True)
    bx.axvline(1.0, color=TEXT2, lw=0.8)
    for y, (label, g, lo, hi) in enumerate(reversed(gains)):
        bx.plot([lo, hi], [y, y], color=OURS, lw=2, solid_capstyle='round')
        bx.plot([g], [y], 'o', ms=7, color=OURS, mec=SURFACE, mew=2)
        bx.text(hi + 0.01, y, f'{g:.2f}×  [{lo:.2f}–{hi:.2f}]', va='center', color=TEXT, fontsize=8.8)
    bx.set_yticks(range(len(gains))); bx.set_yticklabels([g[0] for g in reversed(gains)])
    bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f'{t:.1f}×'))
    bx.set_xlabel('Copy + DFlash vs DFlash alone, same requests')
    bx.set_title('(b) Measured gain over DFlash (95% CI)', loc='left', color=TEXT, fontsize=11, pad=10)
    v, lo, hi, n = speedup('edit261', 'copyflash63')
    fig.text(0.012, 0.955, f'Copy unchanged code, draft new code in one diffusion pass: {v:.1f}× faster lossless decoding on code edits',
             fontsize=13, fontweight='bold', color=TEXT, ha='left')
    fig.text(0.012, 0.895, f'{TARGET_NAME}, H100, vLLM, batch 1, greedy; output equals the target\'s own greedy decoding up to bf16 near-ties '
             f'({same_as_ar("edit261", "copyflash63"):.0%} byte-identical, DFlash {same_as_ar("edit261", "dflash"):.0%}).',
             fontsize=9.5, color=TEXT2, ha='left')
    fig.text(0.012, 0.852, 'Speedup = total plain-AR latency / total method latency, prefill included; all methods measured end to end '
             f'in vLLM. Ours: {v:.2f}× (95% CI {lo:.2f}–{hi:.2f}).', fontsize=9.5, color=TEXT2, ha='left')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'hero_{KEY}.{ext}', dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return rows, gains


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': font()})
    social()
    explainer()
    rows, gains = hero()
    print(json.dumps({'a': [(r[0], round(r[1], 2)) for r in rows],
                      'b': [(g[0], round(g[1], 3), round(g[2], 3), round(g[3], 3)) for g in gains]}, indent=1))


if __name__ == '__main__':
    main()
