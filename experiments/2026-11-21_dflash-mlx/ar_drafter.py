"""Matched comparison: copy + small AR drafter (EfficientEdit-style) vs copy + DFlash, same Qwen3-4B outputs.

An AR drafter needs no target hidden states, so the target's recorded greedy output Y (dflash_Qwen3-4B.jsonl) is
replayed exactly. Per cycle: suffix-match copy proposal (up to 32) and, when copy cannot cover gamma tokens, a
greedy AR draft of gamma tokens; the longer match with Y is accepted plus one target token. This equals two-branch
tree verification on the target (up to bf16 near-ties). Drafting stops at the first mismatch, which does not change
acceptance; cost is charged for gamma sequential drafter steps in the projection.

Usage: python ar_drafter.py DRAFTER [INPUT_JSONL OUTPUT_JSONL]  (e.g. Qwen/Qwen3-0.6B)
Default input: dflash_Qwen3-4B.jsonl here. Any file with prompt/out token ids works (e.g. Qwen3-8B harness_dflash.jsonl).
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-21_copy-audit'))
from sim import SuffixIndex, align, lcp  # noqa: E402


class Drafter:
    def __init__(self, model):
        self.m = model

    def reset(self):
        self.cache, self.n = make_prompt_cache(self.m), 0

    def draft(self, ctx, Y, i, gamma):
        logits = self.m(mx.array([ctx[self.n:]]), self.cache)
        self.n = len(ctx)
        out = []
        for _ in range(gamma):
            t = mx.argmax(logits[0, -1]).item()
            out.append(t)
            if i + len(out) - 1 >= len(Y) or t != Y[i + len(out) - 1] or len(out) == gamma:
                break
            logits = self.m(mx.array([[t]]), self.cache)
        extra = self.cache[0].offset - self.n
        if extra > 0:
            trim_prompt_cache(self.cache, extra)
        return out


def replay(P, Y, lab, dr, policy, gamma):
    dr.reset()
    sx = SuffixIndex(P); sx.extend(Y[:1])
    i, cyc = 1, []
    while i < len(Y):
        cp, src = sx.propose(32) if policy == 'hybrid' else ([], None)
        a_copy = lcp(cp, Y, i)
        a_d, drafted = -1, False
        if policy == 'alone' or a_copy < gamma:
            d = dr.draft(P + Y[:i], Y, i, gamma)
            a_d, drafted = lcp(d, Y, i), True
        a = max(a_copy, a_d)
        cyc.append(dict(n=i, acc=a, drafted=drafted, a_draft=a_d, novel=lab[i] < 0))
        step = min(a + 1, len(Y) - i)
        sx.extend(Y[i:i + step]); sx.ptr = (src + a) if (cp and a == a_copy and a) else None
        i += step
    return cyc


def main():
    name = sys.argv[1]
    # bound MLX memory: GPU buffers are wired on Apple Silicon and cannot be swapped or compressed
    mx.set_memory_limit(8 * 2**30)
    mx.set_cache_limit(1 * 2**30)
    model, _ = load(name)
    dr = Drafter(model)
    tag = name.split('/')[-1]
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'dflash_Qwen3-4B.jsonl'
    rows = [json.loads(l) for l in src.open()]
    path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / f'ardraft_{tag}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    with path.open('a') as f:
        for k, r in enumerate(rows):
            if r['id'] in done or not 8 <= len(r['out']) < 1400:
                continue
            P, Y = r['prompt'], r['out']
            lab = align(P, Y)
            t0 = time.perf_counter()
            rec = dict(id=r['id'], L=len(Y), P=len(P))
            for policy, gamma in (('alone', 7), ('hybrid', 7), ('hybrid', 15)):
                rec[f'{policy}{gamma}'] = replay(P, Y, lab, dr, policy, gamma)
            f.write(json.dumps(rec) + '\n'); f.flush()
            mx.clear_cache()
            print(tag, k, r['id'], len(Y), {p: round((len(Y) - 1) / len(rec[p]), 1) for p in ('alone7', 'hybrid7', 'hybrid15')},
                  f'{time.perf_counter() - t0:.1f}s', f'mlx {mx.get_active_memory() / 2**30:.1f}G peak {mx.get_peak_memory() / 2**30:.1f}G', flush=True)


if __name__ == '__main__':
    main()
