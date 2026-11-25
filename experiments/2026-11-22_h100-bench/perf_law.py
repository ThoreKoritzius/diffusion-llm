"""Performance law for speculative code editing, fitted and tested on held-out data (CPU only).

Structure of an edit (from aligning the output with the prompt): C copied tokens, N new tokens, R copy runs.
Law, per method m:   cycles = C / tau_copy + N / tau_new + delta * R          (linear in 1/tau, delta)
                     latency = prefill(P) + s * c_m * cycles,   prefill(P) = a + b * prompt_tokens
c_m (cycle cost in AR steps) and s (AR step) come from the calibration split only.

Tests
 1. held-out edits: random 50/50 splits (20 seeds) and group transfer (functions <-> full files); targets are the
    *measured* vLLM per-edit latency and speedup of every method.
 2. composition: hybrid (copy + DFlash router, harness) cycles predicted from components alone:
    tau_copy from prompt lookup (ngram31) and tau_new from DFlash (dflash15), both fitted in vLLM.
 3. before decoding: the same law with structure from the benchmark's reference diff (old -> reference code)
    instead of the model output.
Usage: python perf_law.py [TAG]  -> results/<tag>/perf_law.md
"""
import difflib
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from common import HERE, cohort, group, prompt_text
from sim import align

TAG = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-8B'
RES = HERE / 'results' / TAG
METHODS = ['ngram7', 'ngram15', 'ngram31', 'eagle3_3', 'eagle3_7', 'draft06b7', 'draft17b7', 'dflash15']


def jl(name):
    p = RES / f'{name}.jsonl'
    return {json.loads(l)['id']: json.loads(l) for l in p.open()} if p.exists() else {}


def structure(lab):
    C = sum(1 for x in lab if x >= 0)
    R = sum(1 for t in range(len(lab)) if lab[t] >= 0 and (t == 0 or lab[t - 1] != lab[t] - 1))
    return np.array([C, len(lab) - C, R], float)


def ref_structure(tok, old, ref):
    a, b = tok.encode(old, add_special_tokens=False), tok.encode(ref, add_special_tokens=False)
    lab = [-1] * len(b)
    for i, j, n in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if n >= 4:
            for t in range(n):
                lab[j + t] = i + t
    return structure(lab)


def nnls(X, y):
    """Non-negative least squares by active-set elimination (3 features, so this is enough)."""
    keep = list(range(X.shape[1]))
    while True:
        w = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
        if (w >= 0).all() or len(keep) == 1:
            out = np.zeros(X.shape[1]); out[keep] = np.clip(w, 0, None)
            return out
        keep.pop(int(np.argmin(w)))


def main():
    tok = AutoTokenizer.from_pretrained(f'Qwen/{TAG}')
    items = {c['id']: c for c in cohort()}
    V = {m: jl(f'vllm_{m}') for m in ['ar'] + METHODS}
    AR = V['ar']
    ids = [i for i in AR if AR[i]['out_tokens'] > 8 and all(i in V[m] for m in METHODS)]
    X, Xref, P, L, ttft, step, grp = {}, {}, {}, {}, {}, {}, {}
    for i in ids:
        Pi = tok.encode(prompt_text(tok, items[i]), add_special_tokens=False)
        Y = tok.encode(AR[i]['text'], add_special_tokens=False)
        X[i] = structure(align(Pi, Y))
        Xref[i] = ref_structure(tok, items[i]['old'], items[i]['ref'])
        P[i], L[i] = len(Pi), AR[i]['out_tokens']
        ttft[i] = AR[i]['ttft_ms']
        step[i] = (AR[i]['latency_ms'] - AR[i]['ttft_ms']) / (AR[i]['out_tokens'] - 1)
        grp[i] = 'file' if group(items[i]) == 'full file' else 'function'
    cyc = {m: {i: V[m][i]['drafts'] for i in ids} for m in METHODS}
    cost = {m: {i: (V[m][i]['latency_ms'] - V[m][i]['ttft_ms']) / max(1, V[m][i]['drafts']) / step[i] for i in ids} for m in METHODS}

    def fit(train, m, feats):
        A = np.array([feats[i] for i in train]); y = np.array([cyc[m][i] for i in train], float)
        w = nnls(A, y)
        pa = np.polyfit([P[i] for i in train], [ttft[i] for i in train], 1)
        return dict(w=w, c=float(np.median([cost[m][i] for i in train])), s=float(np.median([step[i] for i in train])), pre=pa)

    def predict(model, i, feats):
        cyc_hat = max(1.0, float(feats[i] @ model['w']))
        pre = float(np.polyval(model['pre'], P[i]))
        return cyc_hat, pre + model['s'] * model['c'] * cyc_hat, pre + model['s'] * (L[i] - 1)

    def evaluate(train, test, feats):
        rows = {}
        for m in METHODS:
            mod = fit(train, m, feats)
            lat_err, cyc_err, pred_t, true_t, pred_ar, true_ar = [], [], 0, 0, 0, 0
            for i in test:
                ch, lh, arh = predict(mod, i, feats)
                cyc_err.append(abs(ch - cyc[m][i]) / cyc[m][i])
                lat_err.append(abs(lh - V[m][i]['latency_ms']) / V[m][i]['latency_ms'])
                pred_t += lh; true_t += V[m][i]['latency_ms']; pred_ar += arh; true_ar += AR[i]['latency_ms']
            rows[m] = dict(cyc_mape=np.median(cyc_err), lat_mape=np.median(lat_err),
                           spd_pred=pred_ar / pred_t, spd_true=true_ar / true_t, w=mod['w'])
        return rows

    out = [f'# Performance law: {TAG}, {len(ids)} edits\n',
           'cycles = C/τ_copy + N/τ_new + δ·R (non-negative fit); latency = prefill(P) + s·c_m·cycles. '
           'Errors are on held-out edits only; speedups are aggregate (sum of AR latency / sum of method latency).\n']
    # 1a random splits
    rng = np.random.default_rng(0)
    acc = {m: [] for m in METHODS}
    for seed in range(20):
        perm = rng.permutation(ids); half = len(ids) // 2
        r = evaluate(list(perm[:half]), list(perm[half:]), X)
        for m in METHODS:
            acc[m].append(r[m])
    out += ['## 1a. Random 50/50 splits (20 seeds; medians over seeds)\n',
            '| Method | τ_copy | τ_new | δ per copy run | Per-edit cycle error | Per-edit latency error | Speedup predicted / measured |',
            '|---|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        w = np.median([a['w'] for a in acc[m]], axis=0)
        inv = lambda x: f'{1 / x:.1f}' if x > 1e-9 else '∞'
        out.append(f"| {m} | {inv(w[0])} | {inv(w[1])} | {w[2]:.2f} | {np.median([a['cyc_mape'] for a in acc[m]]):.0%} | "
                   f"{np.median([a['lat_mape'] for a in acc[m]]):.0%} | {np.median([a['spd_pred'] for a in acc[m]]):.2f}× / "
                   f"{np.median([a['spd_true'] for a in acc[m]]):.2f}× |")
    # 1b group transfer
    out += ['\n## 1b. Group transfer (fit on one edit type, predict the other)\n',
            '| Method | functions → files: speedup pred / true (latency err) | files → functions: speedup pred / true (latency err) |',
            '|---|---|---|']
    fn = [i for i in ids if grp[i] == 'function']; fl = [i for i in ids if grp[i] == 'file']
    r1, r2 = evaluate(fn, fl, X), evaluate(fl, fn, X)
    for m in METHODS:
        out.append(f"| {m} | {r1[m]['spd_pred']:.2f}× / {r1[m]['spd_true']:.2f}× ({r1[m]['lat_mape']:.0%}) | "
                   f"{r2[m]['spd_pred']:.2f}× / {r2[m]['spd_true']:.2f}× ({r2[m]['lat_mape']:.0%}) |")
    # 2 composition
    H = {i: v for i, v in jl('harness_hybrid_router8').items() if i in X}
    D = {i: v for i, v in jl('harness_dflash').items() if i in X}
    if H:
        wc, wn = fit(ids, 'ngram31', X)['w'], fit(ids, 'dflash15', X)['w']
        out += ['\n## 2. Composition: predict copy + DFlash (router-8) from its components\n',
                'τ_copy and δ from prompt lookup (ngram31), τ_new from DFlash (dflash15), both fitted in vLLM on all edits; '
                'compared with the hybrid\'s actual cycles in the harness (no hybrid data used in the fit).\n',
                '| Scope | n | Tokens per call predicted | Actual | Per-edit cycle error | DFlash-alone actual tokens per call |',
                '|---|---:|---:|---:|---:|---:|']
        for scope, sel in (('all', lambda i: True), ('functions', lambda i: grp[i] == 'function'), ('full files', lambda i: grp[i] == 'file')):
            hs = [i for i in H if sel(i)]
            pred = {i: max(1.0, X[i][0] * wc[0] + X[i][1] * wn[1] + X[i][2] * wc[2]) for i in hs}
            act = {i: len(H[i]['cycles']) for i in hs}
            toks = sum(len(H[i]['out']) - 1 for i in hs)
            err = np.median([abs(pred[i] - act[i]) / act[i] for i in hs])
            dfl = sum(len(D[i]['out']) - 1 for i in hs if i in D) / sum(len(D[i]['cycles']) for i in hs if i in D)
            out.append(f'| {scope} | {len(hs)} | {toks / sum(pred.values()):.1f} | {toks / sum(act.values()):.1f} | {err:.0%} | {dfl:.1f} |')
    # 3 before decoding
    out += ['\n## 3. Before decoding: structure from the reference diff instead of the model output (random splits)\n',
            '| Method | Per-edit latency error (output structure) | Per-edit latency error (reference structure) | Speedup pred / true (reference structure) |',
            '|---|---:|---:|---|']
    rng = np.random.default_rng(1)
    accr = {m: [] for m in METHODS}
    for seed in range(20):
        perm = rng.permutation(ids); half = len(ids) // 2
        r = evaluate(list(perm[:half]), list(perm[half:]), Xref)
        for m in METHODS:
            accr[m].append(r[m])
    for m in METHODS:
        out.append(f"| {m} | {np.median([a['lat_mape'] for a in acc[m]]):.0%} | {np.median([a['lat_mape'] for a in accr[m]]):.0%} | "
                   f"{np.median([a['spd_pred'] for a in accr[m]]):.2f}× / {np.median([a['spd_true'] for a in accr[m]]):.2f}× |")
    corr = np.corrcoef([X[i][1] / max(1, X[i].sum() - 0) for i in ids], [Xref[i][1] / max(1, Xref[i].sum()) for i in ids])[0, 1]
    out.append(f'\nCorrelation between new-code share in the model output and in the reference diff: {corr:.2f}.')
    (RES / 'perf_law.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
