"""Hero figure from the A100 results (all numbers read from results/<tag>/, none typed in).

(a) End-to-end speedup over plain AR. Solid = measured end to end in vLLM; hatched = projected from harness
    acceptance x measured vLLM cycle costs (the hybrid is not yet implemented in a serving engine).
(b) Measured gain of copy + DFlash (router, min match 8) over DFlash alone in the PyTorch harness, with 95%
    bootstrap intervals over edits.
Usage: python hero_figure.py [TAG] -> figures/hero_<tag>.png / .pdf
"""
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from common import HERE, cohort, group

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
SURFACE, TEXT, TEXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
EXIST, EXIST_INK = '#2a78d6', '#1c5cab'   # categorical slot 1 + a darker step of its ramp for hatch ink
OURS, OURS_INK = '#eb6834', '#b4451c'     # categorical slot 2
rng = np.random.default_rng(0)


def jl(name):
    return {json.loads(l)['id']: json.loads(l) for l in (RES / f'{name}.jsonl').open()}


def measured(name, AR):
    R = jl(f'vllm_{name}')
    ids = [i for i in R if i in AR and AR[i]['out_tokens'] > 8]
    return sum(AR[i]['latency_ms'] for i in ids) / sum(R[i]['latency_ms'] for i in ids)


def projected(label):
    text = (RES / 'summary.md').read_text()
    m = re.search(rf'\| {re.escape(label)} \| \d+ \| [\d.]+ \| [\d.]+ \| ([\d.]+)× \|', text)
    return float(m.group(1))


def ratio_ci(num, den, n=4000):
    num, den = np.array(num), np.array(den)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def hbar(ax, y, x, h, color, hatch_ink=None):
    """Bar from 0 to x: square at the baseline, rounded data end."""
    r = min(0.09 * ax.get_xlim()[1] / 10, x / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), x, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                mutation_aspect=h / r * 0.35 if r else 1, fc=color, ec='none', lw=0))
    ax.add_patch(Rectangle((0, y - h / 2), x - r, h, fc=color, ec='none', lw=0))
    if hatch_ink:
        ax.add_patch(Rectangle((0, y - h / 2), x - r * 0.5, h, fc='none', ec=hatch_ink, lw=0, hatch='////'))


def main():
    items = {c['id']: c for c in cohort()}
    AR = jl('vllm_ar')
    rows = [  # label, speedup, ours, projected
        ('Qwen3-1.7B draft model, γ = 7', measured('draft17b7', AR), False, False),
        ('Qwen3-0.6B draft model, γ = 7', measured('draft06b7', AR), False, False),
        ('EAGLE-3, 7 tokens', measured('eagle3_7', AR), False, False),
        ('Prompt lookup (copy), 31 tokens', measured('ngram31', AR), False, False),
        ('Copy + 0.6B draft (EfficientEdit-style)', projected('Copy + Qwen3-0.6B, γ=7 (EfficientEdit-style)'), False, True),
        ('DFlash (one-pass diffusion)', measured('dflash15', AR), False, False),
        ('Copy + DFlash, router', projected('Copy + DFlash, router (deployable)'), True, True),
        ('Copy + DFlash, two-branch', projected('Copy + DFlash, two-branch tree'), True, True),
    ]
    D, H = jl('harness_dflash'), jl('harness_hybrid_router8')
    ci_rows = []
    pt = [json.loads(l) for l in (RES / 'paired_timing.jsonl').open()]
    by = {}
    for r in pt:
        by.setdefault(r['id'], {}).setdefault(r['mode'], 0.0)
        by[r['id']][r['mode']] += r['total_ms']
    pids = [i for i, v in by.items() if len(v) == 2]
    ci_rows.append((f'Paired repeat, {len(pids)} edits × 3',) + ratio_ci([by[i]['dflash'] for i in pids], [by[i]['hybrid_router'] for i in pids]))
    for label, sel in (('All edits', lambda g: True), ('Single functions', lambda g: g != 'full file'), ('Full files', lambda g: g == 'full file')):
        ids = [i for i in H if i in D and sel(group(items[i]))]
        ci_rows.append((f'{label} ({len(ids)})',) + ratio_ci([D[i]['total_ms'] for i in ids], [H[i]['total_ms'] for i in ids]))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'hatch.linewidth': 0.9,
                         'axes.edgecolor': GRID, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT})
    fig = plt.figure(figsize=(13.6, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.55, left=0.215, right=0.985, top=0.72, bottom=0.13)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ('top', 'right', 'left'):
            a.spines[s].set_visible(False)
        a.spines['bottom'].set_color(GRID)
        a.tick_params(axis='y', length=0)
        a.tick_params(axis='x', color=GRID)
    # panel a
    xmax = 11
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(0, 11, 2)); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for y, (label, v, ours, proj) in enumerate(rows):
        c, ink = (OURS, OURS_INK) if ours else (EXIST, EXIST_INK)
        hbar(ax, y, v, 0.56, c, ink if proj else None)
        ax.text(v + 0.15, y, f'{v:.1f}×' + ('  projected' if proj else ''), va='center', ha='left',
                color=TEXT if not proj else TEXT2, fontsize=9.5, fontweight='bold' if ours else 'normal')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2]:
            t.set_fontweight('bold')
    ax.axvline(1, color=TEXT2, lw=0.8)
    ax.text(1.08, len(rows) - 0.42, 'plain AR = 1×', color=TEXT2, fontsize=8.5, va='bottom')
    ax.set_xlabel('End-to-end speedup over plain autoregressive decoding')
    ax.set_title('(a) Speedup over plain AR on 261 real code edits', loc='left', color=TEXT, fontsize=11, pad=10)
    ax.legend(handles=[Patch(fc=EXIST, ec='none', label='Existing method'), Patch(fc=OURS, ec='none', label='Copy + DFlash (this work)'),
                       Patch(fc='#d9d8d4', ec=TEXT2, hatch='////', lw=0, label='Projected from measured cycle costs')],
              loc='lower right', frameon=False, fontsize=8.5, labelcolor=TEXT2)
    # panel b
    bx.set_xlim(0.95, 1.75); bx.set_ylim(-0.6, len(ci_rows) - 0.4)
    bx.xaxis.grid(True, color=GRID, lw=0.8); bx.set_axisbelow(True)
    bx.axvline(1.0, color=TEXT2, lw=0.8)
    bx.text(1.005, len(ci_rows) - 0.45, 'DFlash alone = 1×', color=TEXT2, fontsize=8.5, va='bottom')
    for y, (label, r, lo, hi) in enumerate(reversed(ci_rows)):
        bx.plot([lo, hi], [y, y], color=OURS, lw=2, solid_capstyle='round')
        bx.plot([r], [y], 'o', ms=8, color=OURS, mec=SURFACE, mew=2)
        bx.text(hi + 0.012, y, f'{r:.2f}×  [{lo:.2f}–{hi:.2f}]', va='center', color=TEXT, fontsize=9)
    bx.set_yticks(range(len(ci_rows))); bx.set_yticklabels([r[0] for r in reversed(ci_rows)])
    bx.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    bx.set_xticklabels([f'{t:.1f}×' for t in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
    bx.set_xlabel('Copy + DFlash (router) vs DFlash alone, measured')
    bx.set_title('(b) Measured gain over DFlash (95% CI)', loc='left', color=TEXT, fontsize=11, pad=10)
    fig.text(0.012, 0.955, 'Copy unchanged code, draft new code in one diffusion pass: the fastest lossless decoder on code edits',
             fontsize=13, fontweight='bold', color=TEXT, ha='left')
    env = (RES / 'env.txt').read_text()
    gpu = 'A100-SXM4-40GB' if 'A100' in env else 'GPU'
    fig.text(0.012, 0.895, f'Qwen3-8B, {gpu}, batch 1, greedy; output equals the target\'s own greedy decoding up to bf16 near-ties. '
             'CanItEdit + EditPackFT: single-function and full-file rewrites.', fontsize=9.5, color=TEXT2, ha='left')
    fig.text(0.012, 0.852, 'Speedup = total plain-AR latency / total method latency, prefill included. Solid: measured end to end in vLLM. '
             'Hatched: harness acceptance × measured vLLM cycle costs.', fontsize=9.5, color=TEXT2, ha='left')
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'hero_{TAG}.{ext}', dpi=200, facecolor=SURFACE)
    print(json.dumps({'a': [(r[0], round(r[1], 2)) for r in rows], 'b': [(r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)) for r in ci_rows]}, indent=1))


if __name__ == '__main__':
    main()
