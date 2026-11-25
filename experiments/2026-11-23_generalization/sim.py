"""Exact replay of lossless greedy speculative decoding against recorded target outputs.

Each cycle: a proposer drafts tokens, the target verifies them in one call, the longest
matching prefix is accepted plus one bonus token from the target. Output is identical to
greedy AR by construction, so only target calls (and widths) differ between policies.
"""
import difflib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent / 'data'  # load() unused here
MIN_BLOCK = 4  # source-aligned blocks shorter than this count as novel tokens


def load():
    cohort = {c['id']: c for c in json.loads((HERE / 'cohort.json').read_text())}
    rows = [json.loads(l) for l in (HERE / 'targets.jsonl').open()]
    return cohort, rows


def align(P, Y):
    """Label each output token: source position it copies (>=0) or -1 for novel."""
    lab = [-1] * len(Y)
    for a, b, size in difflib.SequenceMatcher(None, P, Y, autojunk=False).get_matching_blocks():
        if size >= MIN_BLOCK:
            for t in range(size):
                lab[b + t] = a + t
    return lab


def lcp(prop, Y, i):
    n = 0
    while n < len(prop) and i + n < len(Y) and prop[n] == Y[i + n]:
        n += 1
    return n


class SuffixIndex:
    """Longest-suffix-match copy proposer over prompt + generated tokens."""
    def __init__(self, P, max_suffix=32):
        self.ctx = list(P)
        self.pos = defaultdict(list)
        for j, t in enumerate(self.ctx):
            self.pos[t].append(j)
        self.max_suffix = max_suffix
        self.ptr = None  # position after last proposal source, for continuity ties

    def extend(self, toks):
        for t in toks:
            self.pos[t].append(len(self.ctx))
            self.ctx.append(t)

    def propose(self, K):
        ctx, n = self.ctx, len(self.ctx)
        best_len, best = 0, []
        for j in self.pos.get(ctx[-1], []):
            if j >= n - 1:
                continue
            L = 1
            while L < self.max_suffix and j - L >= 0 and ctx[j - L] == ctx[n - 1 - L]:
                L += 1
            if L > best_len:
                best_len, best = L, [j]
            elif L == best_len:
                best.append(j)
        self.last_len = best_len  # matched suffix length: the deployable router's copy-confidence signal
        if not best:
            return [], None
        # prefer continuing right after the previous copy pointer, else latest occurrence
        j = min(best, key=lambda x: (0 if self.ptr is not None and x + 1 == self.ptr else 1, -x))
        return ctx[j + 1:j + 1 + K], j + 1


def ngram_propose(ctx, K, nmax=3):
    """HF-style prompt lookup: earliest match of the last n-gram, n from nmax down to 1."""
    n = len(ctx)
    for g in range(nmax, 0, -1):
        if n < g + 1:
            continue
        pat = ctx[n - g:]
        for j in range(0, n - g):
            if ctx[j:j + g] == pat:
                return ctx[j + g:j + g + K]
    return []


def replay(P, Y, lab, policy, K):
    """Return list of cycles (start, proposal_len, accepted, region)."""
    i, cycles = 0, []
    sx = SuffixIndex(P) if policy != 'ar' else None
    # the first output token comes from prefill (a target call already paid in prefill)
    i = 1
    if sx:
        sx.extend(Y[:1])
    while i < len(Y):
        region = 'copy' if lab[i] >= 0 else 'novel'
        src = None
        if policy == 'ar':
            prop = []
        elif policy == 'ngram':
            prop = ngram_propose(P + Y[:i], K)
        elif policy == 'suffix':
            prop, src = sx.propose(K)
        else:
            # oracle-assisted bounds: best of realistic suffix match and alignment-known copy
            cands = [sx.propose(K)[0], P[lab[i]:lab[i] + K] if lab[i] >= 0 else []]
            if policy.startswith('oracle_fill') and lab[i] < 0:
                cands.append(Y[i:i + int(policy.split('_')[-1])])
            if policy == 'perfect':
                cands = [Y[i:i + K]]
            prop = max(cands, key=lambda c: lcp(c, Y, i))
            src = None
        a = lcp(prop, Y, i)
        cycles.append((i, len(prop), a, region))
        step = min(a + 1, len(Y) - i)
        if sx:
            sx.extend(Y[i:i + step])
            sx.ptr = (src + a) if (src is not None and prop and a) else None
        i += step
    return cycles


def summarize(rows, cohort, policies):
    out = []
    for r in rows:
        P, Y = r['prompt_ids'], r['out_ids']
        if len(Y) < 8:
            continue
        lab = align(P, Y)
        rec = dict(id=r['id'], regime=cohort[r['id']]['regime'], source=cohort[r['id']]['source'],
                   L=len(Y), novel=sum(1 for x in lab if x < 0),
                   copy_runs=sum(1 for t in range(len(lab)) if lab[t] >= 0 and (t == 0 or lab[t - 1] != lab[t] - 1)),
                   ar_ms=r['total_ms'], prefill_ms=r['prefill_ms'], stop=r['stop'])
        for name, pol, K in policies:
            cyc = replay(P, Y, lab, pol, K)
            rec[name] = dict(calls=len(cyc), widths=[c[1] + 1 for c in cyc],
                             novel_calls=sum(1 for c in cyc if c[3] == 'novel'),
                             acc=[c[2] for c in cyc])
        out.append(rec)
    return out


POLICIES = [('ar', 'ar', 0)] + [(f'ngram{K}', 'ngram', K) for K in (16, 32, 64)] \
    + [(f'suffix{K}', 'suffix', K) for K in (16, 32, 64)] \
    + [(f'copy_best{K}', 'copy_best', K) for K in (16, 32, 64)] \
    + [(f'oracle_fill{k}_{K}', f'oracle_fill_{k}', K) for K in (32,) for k in (4, 8, 16, 32)] \
    + [(f'perfect{K}', 'perfect', K) for K in (16, 32, 64)]

if __name__ == '__main__':
    cohort, rows = load()
    res = summarize(rows, cohort, POLICIES)
    (HERE / 'sim.json').write_text(json.dumps(res))
    print(len(res), 'items simulated')
