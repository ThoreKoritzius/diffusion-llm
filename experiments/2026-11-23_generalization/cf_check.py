"""Compare vLLM methods on the same requests: speedup over AR, tokens per target call, output identical to AR.

Usage: python cf_check.py KEY [COHORT ...] [--methods dflash dflash_pad31 copyflash] [--boot 2000]
Speedup = sum of AR latency / sum of method latency over the ids every listed method has (paired), with a
bootstrap 95% CI over requests. Tokens per call = (accepted + drafts) / drafts from the server's counters.
"""
import argparse
import json
import random

from common import HERE


def load(key, cohort, name):
    p = HERE / 'results' / key / cohort / f'vllm_{name}.jsonl'
    if not p.exists():
        return {}
    return {r['id']: r for r in map(json.loads, p.open())}


def ratio_ci(num, den, boot, seed=0):
    rng = random.Random(seed)
    n = len(num)
    est = sum(num) / sum(den)
    bs = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs.append(sum(num[i] for i in idx) / sum(den[i] for i in idx))
    bs.sort()
    return est, bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('cohorts', nargs='*', default=['edit261', 'ceb150', 'diff104'])
    ap.add_argument('--methods', nargs='+', default=['ngram31', 'dflash', 'dflash_pad31', 'copyflash'])
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    for c in a.cohorts:
        ar = load(a.key, c, 'ar')
        ms = {m: load(a.key, c, m) for m in a.methods}
        ms = {m: v for m, v in ms.items() if v}
        if not ar or not ms:
            continue
        ids = sorted(set(ar).intersection(*ms.values()))
        print(f'\n{a.key} {c}: {len(ids)} paired requests')
        print(f'{"method":>14} {"vs AR":>7} {"95% CI":>13} {"tok/call":>9} {"same as AR":>11}')
        base = None
        for m, R in ms.items():
            lat = [R[i]['latency_ms'] for i in ids]
            s, lo, hi = ratio_ci([ar[i]['latency_ms'] for i in ids], lat, a.boot)
            d = sum(R[i].get('drafts', 0) for i in ids)
            tpc = (sum(R[i].get('accepted', 0) for i in ids) + d) / d if d else float('nan')
            same = sum(R[i]['text'] == ar[i]['text'] for i in ids) / len(ids)
            print(f'{m:>14} {s:6.2f}x {lo:5.2f}-{hi:5.2f}x {tpc:9.2f} {same:10.0%}')
            if m == 'dflash':
                base = lat
        for m in ms:
            if base and m.startswith('copyflash'):
                g, lo, hi = ratio_ci(base, [ms[m][i]['latency_ms'] for i in ids], a.boot)
                print(f'  {m} vs dflash (paired): {g:.3f}x (95% CI {lo:.3f}-{hi:.3f})')


if __name__ == '__main__':
    main()
