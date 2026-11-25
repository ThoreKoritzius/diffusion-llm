"""Shareable explainer: the idea in one picture (LLM passes per token vs per chunk) + the speedup chart.

Usage: python social_explainer.py [TAG] -> figures/explainer_<tag>.png (2400x2700)
Token counts of the example are real Qwen3 tokens; the 4-pass split assumes every proposal is accepted (labelled).
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from hero_figure import HERE, jl, measured, projected, ratio_ci
from social_figure import bar, font

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
SURFACE, TEXT, TEXT2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'
EXIST, OURS = '#2a78d6', '#eb6834'
COPY, COPY_BG, DRAFT, DRAFT_BG, DRAFT_INK, PASS = '#bdbbb4', '#efeee9', '#1baf7a', '#d6f2e7', '#0f7a54', '#6f6e69'
MONO = 'Menlo'
NEW = [('def average(values):', 'copy', 4),
       ('    values = [v for v in values if v is not None]', 'draft', 15),
       ('    if not values:', 'draft', 5),
       ('        return 0', 'draft', 5),
       ('    total = sum(values)', 'copy', 6),
       ('    return total / len(values)', 'copy', 7)]
CHUNKS = [('copy', 4), ('draft', 16), ('draft', 9), ('copy', 13)]  # one LLM pass each (idealized: all accepted)


def chip(fig, x, y, color, label, ink):
    fig.patches.append(FancyBboxPatch((x, y - 0.006), 0.018, 0.012, boxstyle='round,pad=0,rounding_size=0.003',
                                      transform=fig.transFigure, fc=color, ec='none'))
    fig.text(x + 0.026, y, label, fontsize=13, color=ink, va='center')


def strip(ax, y, segments, total, h, colors, gap=0.0016):
    """Horizontal strip of `total` tokens cut into passes; segments = [(n_tokens, color)]."""
    x = 0.0
    for n, c in segments:
        w = n / total
        ax.add_patch(FancyBboxPatch((x + gap / 2, y), w - gap, h, boxstyle='round,pad=0,rounding_size=0.004',
                                    mutation_aspect=4, fc=c, ec='none'))
        x += w


def main():
    AR = jl('vllm_ar')
    rows = [
        ('Standard decoding', 1.0, 'one token per LLM pass', False),
        ('Small draft model', measured('draft06b7', AR), 'a 0.6B model guesses ahead', False),
        ('EAGLE-3', measured('eagle3_7', AR), 'a light head guesses ahead', False),
        ('Copy from the old code', measured('ngram31', AR), 'reuses unchanged lines', False),
        ('DFlash', measured('dflash15', AR), 'diffusion drafts 16 tokens at once', False),
        ('Copy + DFlash (ours)', projected('Copy + DFlash, router (deployable)'), 'copy old lines, diffuse new code', True),
    ]
    by = {}
    for l in (RES / 'paired_timing.jsonl').open():
        r = json.loads(l)
        by.setdefault(r['id'], {}).setdefault(r['mode'], 0.0)
        by[r['id']][r['mode']] += r['total_ms']
    pids = [i for i, v in by.items() if len(v) == 2]
    gain, lo, hi = ratio_ci([by[i]['dflash'] for i in pids], [by[i]['hybrid_router'] for i in pids])
    H = jl('harness_hybrid_router8')
    tpp = sum(len(h['out']) - 1 for h in H.values()) / sum(len(h['cycles']) for h in H.values())

    plt.rcParams.update({'font.family': font()})
    fig = plt.figure(figsize=(12, 13.5), dpi=200, facecolor=SURFACE)
    fig.text(0.055, 0.955, 'Editing code 9× faster, with the same output', fontsize=30, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.925, 'Most of an edited file stays the same. Copy those lines, let a diffusion model draft only the new ones,',
             fontsize=14.5, color=TEXT2)
    fig.text(0.055, 0.906, 'and have the LLM check whole chunks at once instead of writing one token at a time.', fontsize=14.5, color=TEXT2)

    # --- the idea -------------------------------------------------------------------------------------------
    fig.text(0.055, 0.858, 'How it works', fontsize=17, fontweight='bold', color=TEXT)
    chip(fig, 0.055, 0.83, COPY, 'copied from the old code', TEXT2)
    chip(fig, 0.33, 0.83, DRAFT, 'new code, drafted by diffusion', TEXT2)
    code = fig.add_axes([0.055, 0.61, 0.5, 0.2]); code.axis('off'); code.set_xlim(0, 1); code.set_ylim(0, 1)
    step = 1 / len(NEW)
    for k, (line, src, _) in enumerate(NEW):
        y = 1 - (k + 1) * step
        code.add_patch(Rectangle((0, y + 0.012), 1, step - 0.024, fc=COPY_BG if src == 'copy' else DRAFT_BG, ec='none'))
        if src == 'draft':
            code.add_patch(Rectangle((0, y + 0.012), 0.008, step - 0.024, fc=DRAFT, ec='none'))
        code.text(0.03, y + step / 2, line, family=MONO, fontsize=11.2, color=TEXT, va='center')
    total = sum(n for _, _, n in NEW)
    pas = fig.add_axes([0.6, 0.61, 0.345, 0.2]); pas.axis('off'); pas.set_xlim(0, 1); pas.set_ylim(0, 1)
    pas.text(0, 0.97, 'LLM passes to write this function', fontsize=13.5, fontweight='bold', color=TEXT, va='top')
    pas.text(0, 0.78, 'Standard decoding', fontsize=12.5, color=TEXT2, va='bottom')
    pas.text(1, 0.78, f'{total} passes', fontsize=12.5, color=TEXT, fontweight='bold', va='bottom', ha='right')
    strip(pas, 0.6, [(1, PASS)] * total, total, 0.15, None, gap=0.006)
    pas.text(0, 0.36, 'Copy + DFlash', fontsize=12.5, color=TEXT2, va='bottom')
    pas.text(1, 0.36, f'{len(CHUNKS)} passes', fontsize=12.5, color=TEXT, fontweight='bold', va='bottom', ha='right')
    strip(pas, 0.18, [(n, COPY if s == 'copy' else DRAFT) for s, n in CHUNKS], total, 0.15, None, gap=0.012)
    pas.text(0, 0.02, 'one block = one pass of the LLM', fontsize=11, color=MUTED, va='bottom')
    fig.text(0.055, 0.578, 'The LLM checks each chunk in a single pass and keeps only what it would have written itself, so the output is unchanged.',
             fontsize=12.5, color=TEXT2)
    fig.text(0.055, 0.558, f'Idealised example (every guess accepted). On the 261 real edits the LLM keeps {tpp:.0f} tokens per pass on average.',
             fontsize=11, color=MUTED)

    # --- the result -----------------------------------------------------------------------------------------
    fig.text(0.055, 0.5, 'Speed-up over standard decoding', fontsize=17, fontweight='bold', color=TEXT)
    fig.text(0.055, 0.481, 'Qwen3-8B, 261 real code edits, one A100. Baselines measured end to end; ours projected from measured costs.',
             fontsize=12.5, color=TEXT2)
    ax = fig.add_axes([0.255, 0.1, 0.67, 0.36])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.7, len(rows) - 0.35)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for y, (name, v, how, ours) in enumerate(rows):
        bar(ax, y, v, 0.56, OURS if ours else EXIST)
        ax.text(v + 0.14, y + (0.07 if ours else 0), f'{v:.1f}×' + ('*' if ours else ''), va='center', ha='left',
                fontsize=20 if ours else 16, fontweight='bold', color=TEXT)
        ax.text(-0.25, y + 0.14, name, va='center', ha='right', fontsize=15, color=TEXT, fontweight='bold' if ours else 'normal')
        ax.text(-0.25, y - 0.2, how, va='center', ha='right', fontsize=11, color=MUTED)
    ax.text(rows[-1][1] + 0.16, len(rows) - 1 - 0.33, 'projected', va='center', ha='left', fontsize=11, color=MUTED)
    fig.text(0.055, 0.047, f'*Projected: measured acceptance × measured per-pass costs. Measured directly: {gain:.2f}× faster than DFlash alone '
             f'(95% CI {lo:.2f}–{hi:.2f}, 54 edits × 3 repeats).', fontsize=10, color=MUTED)
    fig.text(0.055, 0.029, 'Baselines measured end to end in vLLM. Greedy decoding, batch 1; outputs match standard decoding up to '
             'floating-point ties.', fontsize=10, color=MUTED)
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / f'explainer_{TAG}.png', dpi=200, facecolor=SURFACE)
    print(out / f'explainer_{TAG}.png', f'tokens per pass {tpp:.1f}')


if __name__ == '__main__':
    main()
