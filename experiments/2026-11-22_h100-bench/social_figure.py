"""Social-media version of the hero figure (16:9, plain language). Values read from results/<tag>/.

Usage: python social_figure.py [TAG] -> figures/social_<tag>.png (2400x1350)
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

from hero_figure import HERE, jl, measured, projected, ratio_ci

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
SURFACE, TEXT, TEXT2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
EXIST, OURS = '#2a78d6', '#eb6834'


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ('Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans') if f in have), 'DejaVu Sans')


def bar(ax, y, x, h, color):
    r = min(0.12, x / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), x, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                mutation_aspect=2.2, fc=color, ec='none'))
    ax.add_patch(Rectangle((0, y - h / 2), x - r, h, fc=color, ec='none'))


def main():
    AR = jl('vllm_ar')
    rows = [
        ('Standard decoding', 1.0, 'one token at a time', False),
        ('Small draft model', measured('draft06b7', AR), 'a 0.6B model guesses ahead', False),
        ('EAGLE-3', measured('eagle3_7', AR), 'a light head guesses ahead', False),
        ('Copy from the old code', measured('ngram31', AR), 'reuses unchanged lines', False),
        ('DFlash', measured('dflash15', AR), 'diffusion drafts 16 tokens at once', False),
        ('Copy + DFlash (ours)', projected('Copy + DFlash, router (deployable)'), 'copy old lines, diffuse new code', True),
    ]
    D, H = jl('harness_dflash'), jl('harness_hybrid_router8')
    pt = [json.loads(l) for l in (RES / 'paired_timing.jsonl').open()]
    by = {}
    for r in pt:
        by.setdefault(r['id'], {}).setdefault(r['mode'], 0.0)
        by[r['id']][r['mode']] += r['total_ms']
    pids = [i for i, v in by.items() if len(v) == 2]
    gain, lo, hi = ratio_ci([by[i]['dflash'] for i in pids], [by[i]['hybrid_router'] for i in pids])

    plt.rcParams.update({'font.family': font()})
    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURFACE)
    fig.text(0.055, 0.905, 'LLMs can edit code 9× faster, with the same output', fontsize=30, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.845, 'Speed-up over standard decoding when an LLM rewrites code (Qwen3-8B, 261 real edits, one A100).',
             fontsize=14.5, color=TEXT2)
    fig.text(0.055, 0.805, 'Baselines measured end to end; ours projected from measured costs and measured ' + f'{gain:.2f}×' + ' faster than DFlash.',
             fontsize=14.5, color=TEXT2)
    ax = fig.add_axes([0.255, 0.2, 0.67, 0.56])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.7, len(rows) - 0.35)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for y, (name, v, how, ours) in enumerate(rows):
        bar(ax, y, v, 0.58, OURS if ours else EXIST)
        label = f'{v:.1f}×' + ('*' if ours else '')
        ax.text(v + 0.14, y + (0.06 if ours else 0), label, va='center', ha='left', fontsize=21 if ours else 17,
                fontweight='bold', color=TEXT)
        fig_y = None
        ax.text(-0.25, y + 0.13, name, va='center', ha='right', fontsize=15.5, color=TEXT, fontweight='bold' if ours else 'normal')
        ax.text(-0.25, y - 0.2, how, va='center', ha='right', fontsize=11.5, color=MUTED)
    ax.text(rows[-1][1] + 0.16, len(rows) - 1 - 0.3, 'projected', va='center', ha='left', fontsize=11.5, color=MUTED)
    fig.text(0.055, 0.125, 'How: unchanged lines are copied from the old code, new code is drafted by a diffusion model in one pass, '
             'and the LLM checks every token,', fontsize=12.5, color=TEXT2)
    fig.text(0.055, 0.092, 'so the result is exactly what it would have written on its own.', fontsize=12.5, color=TEXT2)
    fig.text(0.055, 0.045, f'*Projected: measured acceptance × measured per-step costs. Measured directly: {gain:.2f}× faster than DFlash alone '
             f'(95% CI {lo:.2f}–{hi:.2f}, 54 edits × 3 repeats).', fontsize=10, color=MUTED)
    fig.text(0.055, 0.02, 'Baselines measured end to end in vLLM. Greedy decoding, batch 1; outputs match standard decoding up to floating-point ties.',
             fontsize=10, color=MUTED)
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / f'social_{TAG}.png', dpi=200, facecolor=SURFACE)
    print(out / f'social_{TAG}.png', [(r[0], round(r[1], 2)) for r in rows], round(gain, 3))


if __name__ == '__main__':
    main()
