"""README graphics as small, hand-built SVGs in the playground's style (boxy, Inter, blue accent).

  docs/assets/hero.svg           animated: a SQL query denoising from masks in confidence order (replaces the 25 MB GIF)
  docs/assets/arch_model.svg     diffusion as the model: masked SQL window filled in a few parallel steps
  docs/assets/arch_drafter.svg   diffusion as the drafter: copy + one-pass diffusion draft, verified by the LLM

Usage: python docs/assets/make_readme_svgs.py [--final OUT_DIR]   (--final also writes un-animated end states for review)
The animation is plain CSS inside the SVG, so it plays in GitHub's README image view without scripts.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
SANS = "Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
PAGE, CARD, PANEL, LINE = '#f4f7fc', '#ffffff', '#eef3fb', '#dbe2ee'
INK, INK2, MUTED = '#0f172a', '#334155', '#64748b'
BLUE, BLUE_BG, PURPLE = '#2f6ee0', '#e6eefc', '#7c3aed'
GREEN, GREEN_BG, GREEN_LINE = '#1a7f4b', '#e7f7ee', '#b9e4cb'
AMBER, MASK = '#b7791f', '#d6deeb'
COPY, COPY_BG, DRAFT, DRAFT_BG = '#94a3b8', '#eef1f6', '#1baf7a', '#e3f6ee'


def t(x, y, s, size=14, fill=INK, weight=400, family=SANS, anchor='start', cls='', extra=''):
    c = f' class="{cls}"' if cls else ''
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{c}{extra}>{escape(s)}</text>')


def r(x, y, w, h, fill, stroke='none', sw=1, cls='', extra=''):
    c = f' class="{cls}"' if cls else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{c}{extra}/>'


def svg(w, h, body, style=''):
    st = f'<style>{style}</style>' if style else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">'
            f'{st}{r(0, 0, w, h, PAGE)}{body}</svg>\n')


# ---------------------------------------------------------------------------------------------------------------
# hero: the playground, denoising "What is the highest car price for each model?" in 5 steps (early stop of 24)
# ---------------------------------------------------------------------------------------------------------------
STEPS = 5
CAP = 24
T = 9.0                                   # seconds per loop
AT = [0.12 + 0.1 * k for k in range(STEPS)]   # fraction of the loop where step k+1 commits
HOLD_END, RESET = 0.88, 0.94
# (line, column, text, kind, commit step) - confidence order: structure first, then names, then details
TOKENS = [(0, 0, 'SELECT', 'kw', 1), (0, 7, 'model', 'id', 2), (0, 12, ',', 'p', 5), (0, 14, 'MAX', 'fn', 3),
          (0, 17, '(', 'p', 3), (0, 18, 'price', 'id', 4), (0, 23, ')', 'p', 3),
          (1, 0, 'FROM', 'kw', 1), (1, 5, 'cars', 'id', 2),
          (2, 0, 'GROUP', 'kw', 1), (2, 6, 'BY', 'kw', 1), (2, 9, 'model', 'id', 4), (2, 14, ';', 'p', 2)]
GATE = [0.18, 0.42, 0.63, 0.81, 0.98]     # lowest confidence among still-masked tokens, per step
MEAN = [0.55, 0.86, 0.80, 0.90, 1.00]
AR_TOKENS = 17


def pct(f):
    return f'{100 * f:.2f}%'


def window_css(name, start, end):
    """opacity 1 from start to end (fractions of the loop), else 0."""
    e = 0.0001
    return (f'@keyframes {name}{{0%,{pct(max(start - e, 0))}{{opacity:0}}{pct(start)},{pct(end - e)}{{opacity:1}}'
            f'{pct(end)},100%{{opacity:0}}}}.{name}{{opacity:0;animation:{name} {T}s linear infinite}}')


def hero(final=False):
    css = [f'text{{dominant-baseline:alphabetic}}']
    for k in range(1, STEPS + 1):
        css.append(window_css(f'on{k}', AT[k - 1], RESET))                 # committed tokens of step k
        end = AT[k] if k < STEPS else HOLD_END
        css.append(window_css(f'st{k}', AT[k - 1], end if k < STEPS else RESET))  # "step k" readouts
        css.append(window_css(f'fl{k}', AT[k - 1], min(AT[k - 1] + 0.05, end)))  # flash on commit
        css.append(f'@keyframes off{k}{{0%,{pct(AT[k - 1] - 0.0001)}{{opacity:1}}{pct(AT[k - 1])},{pct(RESET - 0.0001)}'
                   f'{{opacity:0}}{pct(RESET)},100%{{opacity:1}}}}.off{k}{{animation:off{k} {T}s linear infinite}}')
    css.append(window_css('st0', 0, AT[0]))
    css.append(window_css('done', AT[-1], RESET))
    css.append(window_css('busy', 0, AT[-1]))
    vis = (lambda cls: '') if final else (lambda cls: cls)
    on = lambda cls: f' class="{cls}"' if not final else ''
    # in the final (review) render, show the end state: step 5, everything committed
    show = lambda k: final and k == STEPS

    W, H = 1200, 600
    b = []
    # header
    b.append(r(40, 30, 20, 20, BLUE))
    b.append(t(70, 47, 'Diffusion LLM', 21, INK, 700))
    b.append(r(930, 24, 230, 34, CARD, LINE))
    b.append(t(1045, 46, 'diffusion.lethonium.com  ↗', 14, BLUE, 600, anchor='middle'))
    # left card
    L = (40, 76, 760, 494)
    b.append(r(*L, CARD, LINE))
    b.append(t(66, 118, 'Text Diffusion Playground', 18, INK, 700))
    b.append(r(470, 96, 196, 32, GREEN_BG, GREEN_LINE))
    b.append(t(568, 117, f'{AR_TOKENS / STEPS:.1f}× fewer steps vs AR', 14, GREEN, 700, anchor='middle'))
    b.append(r(676, 96, 100, 32, BLUE_BG, '#c9d8f5'))
    for k in range(0, STEPS + 1):
        if final and k != STEPS:
            continue
        b.append(t(726, 117, f'Step {k} / {CAP}', 14, '#1d4ed8', 700, anchor='middle', cls='' if final else f'st{k}'))
    # code area with tokens
    cx, cy, cw_, ch_ = 66, 142, 710, 248
    b.append(r(cx, cy, cw_, ch_, PANEL, LINE))
    fs, cw = 36, 0.6 * 36
    x0, ys = 100, [212, 278, 344]
    color = {'kw': BLUE, 'fn': PURPLE, 'id': '#1e293b', 'p': MUTED}
    for line, col, s, kind, k in TOKENS:
        x, y = x0 + col * cw, ys[line]
        w = len(s) * cw
        if not final:
            b.append(f'<g class="off{k}">{r(x + 1, y - 29, w - 2, 36, MASK)}</g>')
            b.append(f'<g class="fl{k}">{r(x - 3, y - 33, w + 6, 44, "#cfe0ff")}</g>')
        weight = 700 if kind in ('kw', 'fn') else 500
        b.append(t(x, y, s, fs, color[kind], weight, MONO, cls='' if final else f'on{k}'))
    # confidence panel
    px, py, pw, ph = 66, 404, 710, 150
    b.append(r(px, py, pw, ph, PAGE, LINE))
    b.append(t(px + 18, py + 28, 'Token confidence', 14, INK2, 600))
    for k in range(1, STEPS + 1):
        if final and k != STEPS:
            continue
        b.append(t(px + pw - 18, py + 28, f'Step {k} / {CAP}  ·  gate {GATE[k - 1]:.0%}  ·  mean {MEAN[k - 1]:.0%}',
                   13, INK, 600, MONO, anchor='end', cls='' if final else f'st{k}'))
    gx0, gx1, gy0, gy1 = px + 18, px + pw - 18, py + 44, py + 118
    b.append(r(gx0, gy0, gx1 - gx0, gy1 - gy0, CARD, '#e5eaf3'))
    thr = gy1 - 0.9 * (gy1 - gy0)
    b.append(f'<line x1="{gx0}" y1="{thr:.1f}" x2="{gx1}" y2="{thr:.1f}" stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    X = lambda i: gx0 + i * (gx1 - gx0) / (CAP - 1)
    Y = lambda v: gy1 - v * (gy1 - gy0)
    for k in range(1, STEPS + 1):
        x1, x2 = X(k - 2) if k > 1 else X(0), X(k - 1)
        g1 = GATE[k - 2] if k > 1 else 0.0
        m1 = MEAN[k - 2] if k > 1 else 0.0
        seg = (f'<polygon points="{x1:.1f},{gy1} {x1:.1f},{Y(g1):.1f} {x2:.1f},{Y(GATE[k - 1]):.1f} {x2:.1f},{gy1}" '
               f'fill="{BLUE}" fill-opacity="0.09"/>'
               f'<line x1="{x1:.1f}" y1="{Y(m1):.1f}" x2="{x2:.1f}" y2="{Y(MEAN[k - 1]):.1f}" stroke="{COPY}" '
               f'stroke-width="1.6" stroke-dasharray="4 3"/>'
               f'<line x1="{x1:.1f}" y1="{Y(g1):.1f}" x2="{x2:.1f}" y2="{Y(GATE[k - 1]):.1f}" stroke="{BLUE}" stroke-width="2.4"/>')
        b.append(f'<g{on(f"on{k}")}>{seg}</g>')
        if not final or k == STEPS:
            mk = (f'<line x1="{x2:.1f}" y1="{gy0}" x2="{x2:.1f}" y2="{gy1}" stroke="{BLUE}" stroke-width="1.5"/>'
                  f'<rect x="{x2 - 4:.1f}" y="{Y(GATE[k - 1]) - 4:.1f}" width="8" height="8" fill="{BLUE}"/>')
            b.append(f'<g{on(f"st{k}")}>{mk}</g>')
    ly = py + 140
    b.append(f'<line x1="{px + 18}" y1="{ly - 4}" x2="{px + 36}" y2="{ly - 4}" stroke="{BLUE}" stroke-width="2.4"/>')
    b.append(t(px + 42, ly, 'gate', 12.5, BLUE, 600))
    b.append(f'<line x1="{px + 86}" y1="{ly - 4}" x2="{px + 104}" y2="{ly - 4}" stroke="{COPY}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    b.append(t(px + 110, ly, 'mean', 12.5, MUTED, 600))
    b.append(f'<line x1="{px + 158}" y1="{ly - 4}" x2="{px + 176}" y2="{ly - 4}" stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.append(t(px + 182, ly, 'early-stop at 90%', 12.5, AMBER, 600))
    b.append(t(px + pw - 18, ly, f'stopped at step {STEPS} of {CAP}  ·  {CAP - STEPS} saved', 12.5, AMBER, 700, anchor='end',
               cls='' if final else 'done'))
    # right card: prompt, context, status, button
    R = (816, 76, 344, 494)
    b.append(r(*R, CARD, LINE))
    b.append(t(840, 112, 'Prompt', 14, INK2, 700))
    b.append(r(840, 124, 296, 74, CARD, '#cdd6e4'))
    b.append(t(856, 152, 'What is the highest car price', 15, INK))
    b.append(t(856, 174, 'for each model?', 15, INK))
    b.append(t(840, 228, 'Context', 14, INK2, 700))
    b.append(r(840, 240, 296, 104, CARD, '#cdd6e4'))
    for i, s in enumerate(['CREATE TABLE cars (id INT,', 'make VARCHAR(50), model', 'VARCHAR(50), year INT, price',
                           'DECIMAL(10,2));']):
        b.append(t(856, 266 + 22 * i, s, 13.5, INK2, 400, MONO))
    b.append(r(840, 364, 296, 74, PANEL))
    if final:
        b.append(t(856, 394, 'Completed.', 14, INK, 400, MONO))
        b.append(t(856, 418, f'{STEPS} passes (AR: {AR_TOKENS})', 14, GREEN, 700, MONO))
    else:
        b.append(t(856, 394, 'Denoising …', 14, MUTED, 400, MONO, cls='busy'))
        b.append(t(856, 394, 'Completed.', 14, INK, 400, MONO, cls='done'))
        b.append(t(856, 418, f'{STEPS} passes (AR: {AR_TOKENS})', 14, GREEN, 700, MONO, cls='done'))
    b.append(r(840, 474, 296, 72, BLUE))
    b.append(t(988, 517, 'Run Generation', 18, '#ffffff', 700, anchor='middle'))
    return svg(W, H, ''.join(b), '' if final else ''.join(css))


# ---------------------------------------------------------------------------------------------------------------
# architecture diagrams (static)
# ---------------------------------------------------------------------------------------------------------------
def box(x, y, w, h, title, sub=(), fill=CARD, stroke=LINE, tcolor=INK, size=16):
    out = [r(x, y, w, h, fill, stroke, 1.2)]
    ty = y + 30 if sub else y + h / 2 + 6
    out.append(t(x + w / 2, ty, title, size, tcolor, 700, anchor='middle'))
    for i, s in enumerate(sub):
        out.append(t(x + w / 2, ty + 22 + 19 * i, s, 13, MUTED, 400, anchor='middle'))
    return ''.join(out)


def arrow(x1, y1, x2, y2, color=MUTED, w=1.8, dash=''):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"{d} '
            f'marker-end="url(#ah{color[1:]})"/>')


def markers(*colors):
    return '<defs>' + ''.join(
        f'<marker id="ah{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors) + '</defs>'


def slots(x, y, states, cw=34, h=26, gap=4, labels=None, color='#9db8ea'):
    """A row of token slots: state 'm' masked, 'n' newly committed, 'c' committed."""
    out = []
    for i, s in enumerate(states):
        fill = {'m': MASK, 'n': BLUE, 'c': color}[s]
        out.append(r(x + i * (cw + gap), y, cw, h, fill))
        if labels and s != 'm':
            out.append(t(x + i * (cw + gap) + cw / 2, y + 17, labels[i], 10.5, '#ffffff', 700, MONO, anchor='middle'))
    return ''.join(out)


def arch_model():
    W, H = 1200, 470
    b = [markers(MUTED, BLUE)]
    b.append(t(40, 52, 'Diffusion as the model', 24, INK, 700))
    b.append(t(40, 80, 'A bidirectional encoder fills in a whole SQL window in a few parallel steps instead of one token per step.',
               15, MUTED))
    b.append(box(40, 130, 220, 120, 'Question + schema', ('“highest car price', 'for each model?”', 'CREATE TABLE cars …')))
    b.append(arrow(262, 190, 318, 190))
    b.append(box(320, 130, 240, 120, 'ModernBERT encoder', ('150M, bidirectional', 'predicts every masked', 'position at once'),
                 fill=BLUE_BG, stroke='#c9d8f5', tcolor='#1d4ed8'))
    b.append(arrow(562, 190, 618, 190))
    lab = ['SEL', 'mod', ',', 'MAX', '(', 'pri', ')', 'FRM', 'cars', 'GRP', 'BY', 'mod', ';']
    rows = [('step 1', 'nmmmmmmnmnnmm'), ('step 2', 'cnmmmmmcncccn'), ('step 3', 'ccmnnmncccccc'),
            ('step 5', 'ccnccncccccnc')]
    b.append(r(620, 112, 540, 168, CARD, LINE, 1.2))
    b.append(t(640, 136, 'SQL window: commit the most confident tokens each step', 13.5, INK2, 600))
    for i, (name, st) in enumerate(rows):
        y = 150 + i * 31
        b.append(t(640, y + 17, name, 12.5, MUTED, 600))
        b.append(slots(700, y, st, cw=30, gap=4, labels=lab))
    # loop arrow back to encoder
    b.append(f'<path d="M890,282 L890,312 L440,312 L440,254" fill="none" stroke="{BLUE}" stroke-width="1.8" '
             f'stroke-dasharray="5 4" marker-end="url(#ah{BLUE[1:]})"/>')
    b.append(t(665, 330, 'repeat until every slot is confident (early stop)', 13, BLUE, 600, anchor='middle'))
    # comparison strip
    b.append(r(40, 360, 1120, 80, CARD, LINE, 1.2))
    b.append(t(64, 392, 'Diffusion', 15, INK, 700))
    b.append(slots(160, 376, 'c' * 5, cw=40, h=24, gap=5))
    b.append(t(400, 393, '5 passes', 15, BLUE, 700))
    b.append(t(64, 426, 'Autoregressive', 15, INK, 700))
    b.append(slots(190, 410, 'c' * 17, cw=18, h=24, gap=4))
    b.append(t(570, 427, '17 passes, one token each', 15, MUTED, 700))
    b.append(t(1136, 393, 'matched 150M pair on an H100:', 13, MUTED, anchor='end'))
    b.append(t(1136, 418, '2.6× faster at tied accuracy', 16, GREEN, 700, anchor='end'))
    return svg(W, H, ''.join(b))


def arch_drafter():
    W, H = 1200, 520
    b = [markers(MUTED, INK, GREEN)]
    b.append(t(40, 52, 'Diffusion as the drafter (lossless speculative decoding)', 24, INK, 700))
    b.append(t(40, 80, 'A cheap proposal is checked by the full LLM in one forward pass. It keeps exactly the tokens it would have '
               'written itself.', 15, MUTED))
    # proposals
    b.append(r(40, 120, 470, 290, CARD, LINE, 1.2))
    b.append(t(64, 148, 'Propose: copy when the old code matches, else draft', 15, INK, 700))
    b.append(r(64, 166, 422, 96, COPY_BG, '#d5dbe5'))
    b.append(t(84, 192, 'Copy from the input', 15, INK, 700))
    b.append(t(84, 214, 'the current text also appears in the input', 13, MUTED))
    b.append(t(84, 233, '(≥ 8-token match) → copy up to 63 tokens', 13, MUTED))
    b.append(slots(84, 242, 'c' * 11, cw=22, h=12, gap=3, color=COPY))
    b.append(r(64, 278, 422, 112, DRAFT_BG, '#bfe6d4'))
    b.append(t(84, 304, 'DFlash diffusion drafter', 15, '#0f7a54', 700))
    b.append(t(84, 326, 'small block-diffusion model reads the LLM’s', 13, MUTED))
    b.append(t(84, 345, 'hidden states, drafts 16 tokens in ONE pass', 13, MUTED))
    b.append(slots(84, 360, 'c' * 15, cw=22, h=12, gap=3, color=DRAFT))
    # verify
    b.append(arrow(512, 265, 586, 265, INK))
    b.append(box(590, 190, 250, 150, 'Target LLM', ('verifies the whole block', 'in a single forward pass', '(Qwen3-8B, Qwen2.5-0.5B …)'),
                 fill=BLUE_BG, stroke='#c9d8f5', tcolor='#1d4ed8', size=17))
    b.append(arrow(842, 265, 906, 265, INK))
    b.append(r(910, 170, 250, 190, CARD, LINE, 1.2))
    b.append(t(1035, 200, 'Keep the agreeing prefix', 15, INK, 700, anchor='middle'))
    acc = [(GREEN, 9), ('#e11d48', 1), ('#cbd5e1', 6)]
    x = 934
    for c, n in acc:
        for _ in range(n):
            b.append(r(x, 218, 11, 22, c)); x += 13
    b.append(t(1035, 266, '+ 1 token from the LLM itself', 13, MUTED, anchor='middle'))
    b.append(t(1035, 292, 'output = the LLM’s own', 13.5, INK2, 600, anchor='middle'))
    b.append(t(1035, 312, 'greedy output (lossless)', 13.5, INK2, 600, anchor='middle'))
    b.append(f'<path d="M1035,362 L1035,440 L275,440 L275,412" fill="none" stroke="{GREEN}" stroke-width="1.8" '
             f'stroke-dasharray="5 4" marker-end="url(#ah{GREEN[1:]})"/>')
    b.append(t(655, 432, 'next step starts after the accepted tokens', 13, GREEN, 600, anchor='middle'))
    # results strip
    b.append(r(40, 458, 1120, 46, CARD, LINE, 1.2))
    b.append(t(64, 487, 'Text-to-SQL, Qwen2.5-0.5B + DFlash:', 14, INK2))
    b.append(t(318, 487, '2.0× faster', 15, GREEN, 700))
    b.append(t(460, 487, 'Code edits, Qwen3-8B, copy + DFlash:', 14, INK2))
    b.append(t(712, 487, '9.0× faster', 15, GREEN, 700))
    b.append(t(1136, 487, 'measured end to end in vLLM, batch 1', 13, MUTED, anchor='end'))
    return svg(W, H, ''.join(b))


def sql_chart():
    """Accuracy vs. latency on one H100 (GPU session 2): diffusion as the model and as the drafter. Data: bench summaries."""
    import json
    import math
    bench = HERE.parent.parent / 'experiments' / '2026-10-14_gpu-session-2' / 'data' / 'bench'
    pt = lambda n: (lambda r: (r['lat_mean_ms'], 100 * r['exec']))(json.loads((bench / f'{n}.summary.json').read_text()))
    AR_C, DIFF_C = '#e8743b', BLUE
    W, H = 1200, 700
    x0, x1, y0, y1 = 120, 1150, 150, 600           # plot box
    lo, hi = math.log(18), math.log(100)           # reversed log axis: faster to the right
    X = lambda ms: x0 + (hi - math.log(ms)) / (hi - lo) * (x1 - x0)
    Y = lambda acc: y1 - (acc - 49) / (82 - 49) * (y1 - y0)
    b = [markers(INK)]
    b.append(t(40, 52, 'Diffusion makes text-to-SQL faster, two ways', 26, INK, 700))
    b.append(t(40, 84, 'Accuracy vs. latency on one H100, batch 1, 1,000 test queries  ·  up and to the right is better', 15, MUTED))
    b.append(r(x0, y0, x1 - x0, y1 - y0, CARD, LINE))
    for acc in range(50, 85, 5):
        if acc <= 80:
            b.append(f'<line x1="{x0}" y1="{Y(acc):.1f}" x2="{x1}" y2="{Y(acc):.1f}" stroke="#e8edf5" stroke-width="1"/>')
            b.append(t(x0 - 12, Y(acc) + 5, f'{acc}%', 13, MUTED, anchor='end'))
    for ms in (100, 70, 50, 30, 20):
        b.append(f'<line x1="{X(ms):.1f}" y1="{y0}" x2="{X(ms):.1f}" y2="{y1}" stroke="#e8edf5" stroke-width="1"/>')
        b.append(t(X(ms), y1 + 24, f'{ms} ms', 13, MUTED, anchor='middle'))
    b.append(t((x0 + x1) / 2, y1 + 52, 'latency per query  →  faster', 14, INK2, 600, anchor='middle'))
    b.append(f'<text x="52" y="{(y0 + y1) / 2}" font-family="{SANS}" font-size="14" font-weight="600" fill="{INK2}" '
             f'text-anchor="middle" transform="rotate(-90 52 {(y0 + y1) / 2})">execution accuracy</text>')

    def dot(name, color, rad, ring=None):
        x, y = pt(name)
        if ring:
            b.append(f'<rect x="{X(x) - rad - 5:.1f}" y="{Y(y) - rad - 5:.1f}" width="{2 * rad + 10}" height="{2 * rad + 10}" '
                     f'fill="{CARD}" stroke="{ring}" stroke-width="3"/>')
        b.append(f'<rect x="{X(x) - rad:.1f}" y="{Y(y) - rad:.1f}" width="{2 * rad}" height="{2 * rad}" fill="{color}"/>')
        return X(x), Y(y), x, y

    def label(px, py, name, ms, acc, dx, dy, anchor='start', big=False):
        b.append(t(px + dx, py + dy, name, 17 if big else 14, INK, 700, anchor=anchor))
        b.append(t(px + dx, py + dy + (21 if big else 18), f'{ms:.0f} ms  ·  {acc:.0f}%', 14 if big else 13, MUTED, anchor=anchor))

    qx, qy, qms, qa = dot('sql-ar-qwen25-coder-0.5b-ar-vllm', AR_C, 9)
    ex, ey, ems, ea = dot('sql-ar-qwen25-coder-0.5b-ar-vllm-eagle3', AR_C, 7)
    fx, fy, fms, fa = dot('sql-ar-qwen25-coder-0.5b-ar-vllm-dflash', AR_C, 8, ring=DIFF_C)
    ax_, ay, ams, aa = dot('ettin150-ar-10ep-vllm-pr61276', AR_C, 13)
    dx_, dy_, dms, da = dot('ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs', DIFF_C, 13)
    fx4, fy4, f4ms, f4a = dot('ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs', DIFF_C, 9)
    label(qx, qy, 'Qwen2.5-0.5B AR', qms, qa, -16, -8, 'end')
    label(ex, ey, '+ AR drafter (EAGLE-3)', ems, ea, 0, 32, 'middle')
    label(fx, fy, '+ diffusion drafter', fms, fa, 20, -8)
    label(ax_, ay, 'Ettin-150M AR', ams, aa, 0, 44, 'middle', big=True)
    label(dx_, dy_, 'Ettin-150M diffusion', dms, da, 0, 44, 'middle', big=True)
    label(fx4, fy4, 'Ettin-400M diffusion', f4ms, f4a, 18, -6)
    # arrows: same model with a diffusion drafter; same size as a diffusion model
    b.append(f'<path d="M{qx + 16:.1f},{qy - 10:.1f} Q{(qx + fx) / 2:.1f},{qy - 90:.1f} {fx - 18:.1f},{fy - 12:.1f}" fill="none" '
             f'stroke="{INK}" stroke-width="2.2" marker-end="url(#ah{INK[1:]})"/>')
    b.append(t((qx + fx) / 2, qy - 84, f'{qms / fms:.1f}× faster', 22, INK, 700, anchor='middle'))
    b.append(t((qx + fx) / 2, qy - 62, 'same model, same outputs', 13.5, MUTED, anchor='middle'))
    b.append(arrow(ax_ + 22, ay + 2, dx_ - 24, dy_ + 2, INK, 2.6))
    b.append(t((ax_ + dx_) / 2, (ay + dy_) / 2 - 34, f'{ams / dms:.1f}× faster', 30, INK, 700, anchor='middle'))
    b.append(t((ax_ + dx_) / 2, (ay + dy_) / 2 - 12, 'same size, tied accuracy', 14, MUTED, anchor='middle'))
    # legend
    lx, ly = x0 + 20, y0 + 22
    b.append(r(lx, ly - 10, 12, 12, DIFF_C)); b.append(t(lx + 22, ly + 1, 'diffusion', 13.5, INK2))
    b.append(r(lx, ly + 14, 12, 12, AR_C)); b.append(t(lx + 22, ly + 25, 'autoregressive (AR)', 13.5, INK2))
    b.append(r(lx - 2, ly + 36, 16, 16, CARD, DIFF_C, 2.5)); b.append(r(lx + 2, ly + 40, 8, 8, AR_C))
    b.append(t(lx + 22, ly + 49, 'AR + diffusion drafter', 13.5, INK2))
    b.append(t(40, H - 18, 'H100 PCIe, mean latency. AR on vLLM, diffusion on CUDA graphs; drafters (EAGLE-3, DFlash) trained with '
               'Speculators. Data: experiments/2026-10-14_gpu-session-2.', 12.5, MUTED))
    return svg(W, H, ''.join(b))


def main():
    out = {'hero.svg': hero(), 'arch_model.svg': arch_model(), 'arch_drafter.svg': arch_drafter(), 'sql_results.svg': sql_chart()}
    for name, s in out.items():
        (HERE / name).write_text(s)
        print(name, f'{len(s.encode()) / 1024:.1f} KB')
    if '--final' in sys.argv:
        d = Path(sys.argv[sys.argv.index('--final') + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / 'hero_final.svg').write_text(hero(final=True))


if __name__ == '__main__':
    main()
