"""Real drafters on the recorded target outputs (acceptance is exact; costs are measured).

A. Lossless policies with a learned drafter (Qwen2.5-Coder-0.5B-Instruct, same tokenizer):
   ar_draft   : drafter proposes k tokens every cycle (classical speculative decoding)
   hybrid     : copy proposer (suffix match) and drafter, verified as a 2-branch tree;
                the drafter only runs when the copy proposal cannot reach k tokens
B. Right-context ablation on edited spans, function regime, Qwen2.5-Coder-0.5B base in FIM mode:
   fim_none   : no suffix
   fim_real   : suffix = old source after the last copied position (deployable)
   fim_oracle : suffix = old source from the true next copy anchor (upper bound)

Drafting stops at the first token that disagrees with the target: tokens after a mismatch can't be
accepted, so acceptance is unaffected. Drafting cost is reported from measured per-token times, charged
for the full k tokens.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from sim import SuffixIndex, align, lcp, load

HERE = Path(__file__).resolve().parent
K_COPY, K_DRAFT = 32, 16
FILE_LIMIT = 50


def sync():
    torch.mps.synchronize()


class CachedDrafter:
    """Greedy AR drafter over prompt + accepted output, KV cache kept across cycles."""
    def __init__(self, model):
        self.m = model
        self.cache, self.n, self.last = None, 0, None

    def reset(self):
        self.cache, self.n, self.last = DynamicCache(), 0, None

    @torch.inference_mode()
    def draft(self, ctx, Y, i, k):
        # extend cache to cover ctx[:-1]; the last ctx token is fed as the first decode input
        new = ctx[self.n:len(ctx) - 1]
        if new:
            self.m(input_ids=torch.tensor([new], device='mps'), past_key_values=self.cache, use_cache=True)
            self.n += len(new)
        base = self.n
        cur, out = ctx[-1], []
        for _ in range(k):
            o = self.m(input_ids=torch.tensor([[cur]], device='mps'), past_key_values=self.cache, use_cache=True)
            cur = int(o.logits[0, -1].argmax())
            out.append(cur)
            if i + len(out) - 1 >= len(Y) or cur != Y[i + len(out) - 1]:
                break
        self.cache.crop(base)
        return out


def run_A(rows, cohort, drafter):
    res = []
    files = 0
    for r in rows:
        c = cohort[r['id']]
        if c['regime'] == 'file':
            files += 1
            if files > FILE_LIMIT:
                continue
        P, Y = r['prompt_ids'], r['out_ids']
        if len(Y) < 8:
            continue
        lab = align(P, Y)
        rec = dict(id=r['id'], regime=c['regime'], L=len(Y))
        for policy in ('ar_draft', 'hybrid'):
            drafter.reset()
            sx = SuffixIndex(P)
            sx.extend(Y[:1])
            i, cyc = 1, []
            while i < len(Y):
                cp, src = sx.propose(K_COPY) if policy == 'hybrid' else ([], None)
                a_copy = lcp(cp, Y, i)
                a_model, drafted = -1, False
                if policy == 'ar_draft' or a_copy < K_DRAFT:
                    d = drafter.draft(P + Y[:i], Y, i, K_DRAFT)
                    a_model, drafted = lcp(d, Y, i), True
                a = max(a_copy, a_model)
                width = 1 + len(cp) + (K_DRAFT if drafted else 0)
                cyc.append(dict(i=i, a=a, a_copy=a_copy, a_model=a_model, width=width, drafted=drafted,
                                region='copy' if lab[i] >= 0 else 'novel'))
                step = min(a + 1, len(Y) - i)
                sx.extend(Y[i:i + step])
                sx.ptr = (src + a) if (cp and a == a_copy and a) else None
                i += step
            rec[policy] = cyc
        res.append(rec)
        print('A', r['id'], len(Y), len(rec['ar_draft']), len(rec['hybrid']), file=sys.stderr, flush=True)
    return res


def fim_texts(tok, item, P, Y, lab, i, old_start):
    old = item['old'].rstrip()
    commented = '\n'.join('# ' + l for l in old.splitlines())
    prefix = (f"# Change request: {item['instruction']}\n# Original version:\n{commented}\n"
              f"# Revised version:\n") + tok.decode(Y[:i])
    def src_from(p):
        txt = tok.decode(P[p:])
        return txt[:txt.find('```')] if '```' in txt else txt
    prev = [lab[t] for t in range(i) if lab[t] >= 0]
    real = src_from(prev[-1] + 1 if prev else old_start)
    nxt = [lab[t] for t in range(i, len(Y)) if lab[t] >= 0]
    oracle = src_from(nxt[0]) if nxt else ''
    return prefix, {'fim_none': '', 'fim_real': real, 'fim_oracle': oracle}


@torch.inference_mode()
def fim_draft(model, tok, prefix, suffix, Y, i, k):
    text = f'<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>'
    ids = tok(text, return_tensors='pt').input_ids.to('mps')
    target = tok.decode(Y[i:i + k + 4])
    cache = DynamicCache()
    o = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    gen, ended = [], False
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids('<|fim_pad|>'), tok.convert_tokens_to_ids('<|file_sep|>'),
             tok.convert_tokens_to_ids('<|fim_prefix|>'), tok.convert_tokens_to_ids('<|fim_suffix|>'),
             tok.convert_tokens_to_ids('<|fim_middle|>'), tok.convert_tokens_to_ids('<|im_end|>')}
    for _ in range(k + 4):
        t = int(o.logits[0, -1].argmax())
        if t in stops:
            ended = True
            break
        gen.append(t)
        if not target.startswith(tok.decode(gen)):
            break
        o = model(input_ids=torch.tensor([[t]], device='mps'), past_key_values=cache, use_cache=True)
    s = tok.decode(gen)
    if ended:
        # an infill drafter emits middle + right anchor: resume copying the given suffix
        s = s + suffix
    # count target tokens fully covered by the drafted text (character-level agreement)
    a, acc = 0, ''
    while a < k and i + a < len(Y):
        nxt = acc + tok.decode([Y[i + a]])
        if not s.startswith(nxt):
            break
        acc, a = nxt, a + 1
    return a


def run_B(rows, cohort, base, tok):
    res = []
    for r in rows:
        c = cohort[r['id']]
        if c['regime'] != 'function':
            continue
        P, Y = r['prompt_ids'], r['out_ids']
        if len(Y) < 8:
            continue
        lab = align(P, Y)
        p_txt = tok.decode(P)
        head = p_txt[:p_txt.index('```python\n') + len('```python\n')]
        old_start = len(tok(head).input_ids)
        # positions: every edited-span cycle start of the suffix-copy replay
        sx = SuffixIndex(P); sx.extend(Y[:1])
        i, pts = 1, []
        while i < len(Y):
            cp, src = sx.propose(K_COPY)
            a = lcp(cp, Y, i)
            if lab[i] < 0:
                pts.append(i)
            step = min(a + 1, len(Y) - i)
            sx.extend(Y[i:i + step]); sx.ptr = (src + a) if (cp and a) else None
            i += step
        rec = dict(id=r['id'], points=[])
        for i in pts:
            prefix, suffixes = fim_texts(tok, c, P, Y, lab, i, old_start)
            rec['points'].append({name: fim_draft(base, tok, prefix, suf, Y, i, K_DRAFT)
                                  for name, suf in suffixes.items()} | {'i': i})
        res.append(rec)
        print('B', r['id'], len(pts), file=sys.stderr, flush=True)
    return res


@torch.inference_mode()
def timing(model, n_ctx=600, reps=20):
    """Per-token decode and per-k-token prefill cost of a drafter at a typical context."""
    ids = torch.randint(100, 30000, (1, n_ctx), device='mps')
    out = {}
    for k in (1, 16):
        ts = []
        for _ in range(reps):
            cache = DynamicCache()
            model(input_ids=ids, past_key_values=cache, use_cache=True)
            sync(); t0 = time.perf_counter()
            if k == 1:
                model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
            else:
                model(input_ids=ids[:, :k], past_key_values=cache, use_cache=True)
            sync(); ts.append((time.perf_counter() - t0) * 1e3)
        out[f'forward_{k}_ms'] = sorted(ts)[len(ts) // 2]
    return out


def main():
    part = sys.argv[1]
    cohort, rows = load()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct')
    if part == 'A':
        m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct', torch_dtype=torch.float16,
                                                 attn_implementation='sdpa').eval().to('mps')
        res = dict(timing=timing(m), items=run_A(rows, cohort, CachedDrafter(m)))
        (HERE / 'draft_A.json').write_text(json.dumps(res))
    else:
        m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.float16,
                                                 attn_implementation='sdpa').eval().to('mps')
        (HERE / 'draft_B.json').write_text(json.dumps(run_B(rows, cohort, m, tok)))


if __name__ == '__main__':
    main()
