"""Pre-registered predictions for the generalization run, made BEFORE any new decoding.

Law (fitted on the A100 Qwen3-8B edit261 run, ../2026-11-22_h100-bench/results/Qwen3-8B):
  cycles = C/tau_copy + N/tau_new + delta*R, structure (C copied, N new, R copy runs) taken from the *reference*
  answer only (old -> reference code; for diff104 the reference rendered as search/replace blocks).
Predicted per cohort and method: tokens per target call (hardware independent) and end-to-end speedup with A100
cost constants. The copy + DFlash composition is NOT pre-registered: built from reference structure its upper bound
collapses to DFlash alone, while the A100 router measured well above DFlash; composition stays a post-hoc analysis. Cross-target rows reuse the
Qwen3-8B constants unchanged (exploratory transfer; expected to be less accurate).
Writes predictions.json with a timestamp and the sha256 of its content.
"""
import difflib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import DIFF_HEAD, HERE, cohort, prompt_text, targets

A100 = HERE.parent / '2026-11-22_h100-bench' / 'results' / 'Qwen3-8B'
MAP = {'ar': 'ar', 'ngram7': 'ngram7', 'ngram15': 'ngram15', 'ngram31': 'ngram31', 'draft7': 'draft06b7',
       'eagle7': 'eagle3_7', 'dflash': 'dflash15'}


def labels(P, Y, n=4):
    """Copy label per output token: 1 if it lies in an n-gram that already occurred in the prompt or earlier output
    (what a lookup-based proposer can copy, in any order and as often as needed), else -1 (new code)."""
    seen = {tuple(P[i:i + n]) for i in range(len(P) - n + 1)}
    lab = [-1] * len(Y)
    for j in range(len(Y) - n + 1):
        g = tuple(Y[j:j + n])
        if g in seen:
            for t in range(j, j + n):
                lab[t] = 1
        if j >= n:
            seen.add(tuple(Y[j - n:j]))
    return lab


def structure(lab):
    C = sum(1 for x in lab if x >= 0)
    R = sum(1 for t in range(len(lab)) if lab[t] >= 0 and (t == 0 or lab[t - 1] < 0))
    return np.array([C, len(lab) - C, R], float)


def diff_answer(old, ref):
    """Reference edit rendered as the search/replace answer (without the prefilled first header)."""
    a, b = old.splitlines(keepends=True), ref.splitlines(keepends=True)
    blocks = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if i1 == i2:  # pure insertion: anchor on the previous line
            i1 = max(0, i1 - 1); j1 = max(0, j1 - 1)
        blocks.append(f"{DIFF_HEAD}{''.join(a[i1:i2])}=======\n{''.join(b[j1:j2])}>>>>>>> REPLACE\n")
    text = '\n'.join(blocks)
    return text[len(DIFF_HEAD):] if text.startswith(DIFF_HEAD) else text


def nnls(X, y):
    keep = list(range(X.shape[1]))
    while True:
        w = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
        if (w >= 0).all() or len(keep) == 1:
            out = np.zeros(X.shape[1]); out[keep] = np.clip(w, 0, None)
            return out
        keep.pop(int(np.argmin(w)))


def ref_features(tok, it):
    P = tok.encode(prompt_text(tok, it), add_special_tokens=False)
    ans = diff_answer(it['old'], it['ref']) if it['format'] == 'diff' else it['ref']
    Y = tok.encode(ans, add_special_tokens=False)
    return P, Y, structure(labels(P, Y))


def main():
    tok8 = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    jl = lambda n: {json.loads(l)['id']: json.loads(l) for l in (A100 / f'vllm_{n}.jsonl').open()}
    V = {k: jl(v) for k, v in MAP.items()}
    AR = V['ar']
    base = {c['id']: c for c in cohort('edit261')}
    ids = [i for i in AR if AR[i]['out_tokens'] > 8 and i in base]
    feats = {}
    for i in ids:
        old = dict(base[i]); _, _, x = ref_features(tok8, old); feats[i] = x
    step = {i: (AR[i]['latency_ms'] - AR[i]['ttft_ms']) / (AR[i]['out_tokens'] - 1) for i in ids}
    law = {}
    for m in MAP:
        if m == 'ar':
            continue
        X = np.array([feats[i] for i in ids]); y = np.array([V[m][i]['drafts'] for i in ids], float)
        cost = float(np.median([(V[m][i]['latency_ms'] - V[m][i]['ttft_ms']) / max(1, V[m][i]['drafts']) / step[i] for i in ids]))
        law[m] = dict(w=nnls(X, y).tolist(), cost=cost)
    pre = np.polyfit([AR[i]['prompt_tokens'] for i in ids], [AR[i]['ttft_ms'] for i in ids], 1).tolist()
    s = float(np.median(list(step.values())))
    out = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               law_source='A100 Qwen3-8B edit261, reference-diff structure', law=law, prefill_ms=pre, ar_step_ms=s,
               predictions={})
    for key in targets():
        tok = AutoTokenizer.from_pretrained(targets()[key]['model'])
        for c in ('edit261', 'ceb150', 'diff104'):
            rows = [ref_features(tok, it) for it in cohort(c)]
            L = sum(len(Y) for _, Y, _ in rows)
            Ptot = sum(len(P) for P, _, _ in rows)
            pred = {}
            for m, d in law.items():
                w = np.array(d['w'])
                cyc = sum(max(1.0, float(x @ w)) for _, _, x in rows)
                ar_t = np.polyval(pre, Ptot / len(rows)) * len(rows) + s * L
                pred[m] = dict(tokens_per_call=round(L / cyc, 2),
                               speedup_a100_costs=round(ar_t / (np.polyval(pre, Ptot / len(rows)) * len(rows) + s * d['cost'] * cyc), 2))
            pred['new_code_share'] = round(float(sum(x[1] for _, _, x in rows) / max(1, sum(x[:2].sum() for _, _, x in rows))), 3)
            out['predictions'][f'{key}/{c}'] = dict(kind='same target' if key == 'qwen3-8b' else 'cross-target (exploratory)', **pred)
    body = json.dumps(out, indent=1, sort_keys=True)
    out['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    (HERE / 'predictions.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    for k, v in out['predictions'].items():
        if k.startswith('qwen3-8b'):
            print(k, 'new-code share', v['new_code_share'], '| dflash', v['dflash'], '| ngram31', v['ngram31'])
    print('sha256', out['sha256'], out['created_utc'])


if __name__ == '__main__':
    main()
