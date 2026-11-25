"""Instrumented DFlash (MLX) on the copy-audit edit cohort.

Modes
  dflash : stock DFlash greedy speculative decoding (block 16). Logs every cycle; its output is the target's
           greedy output and serves as ground truth Y for the other modes.
  hybrid : two branches per cycle, suffix-match copy (up to 32 tokens) and the DFlash draft. The branch that the
           target would accept furthest (checked against Y) is verified, which is what two-branch tree verification
           accepts. The drafter runs only when copy cannot cover the whole block; target hidden states of skipped
           cycles are queued and handed to the drafter at its next call.
  ar     : plain greedy decoding with the same target, for Mac timing.

Usage: python run.py MODE [TARGET] [DRAFT] [LIMIT]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from dflash.model_mlx import _patch_model, _prefill_target, _trim_recent_cache, load_draft

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / '2026-11-21_copy-audit'
sys.path.insert(0, str(AUDIT))
from sim import SuffixIndex, lcp  # noqa: E402

FENCE = '```python\n'


def build_prompt(tok, item):
    what = 'function' if item['regime'] == 'function' else 'file'
    user = (f'Revise the following Python {what} according to the change request. '
            f'Reply with only the complete revised {what} in a single Python code block.\n\n'
            f"Change request: {item['instruction']}\n\n```python\n{item['old'].rstrip()}\n```")
    text = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return text + FENCE


def stopped(tok, toks):
    return any(t in tok.eos_token_ids for t in toks[-1:]) or '```' in tok.decode(toks[-6:])


def run_ar(model, tok, prompt, limit):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array(prompt)[None], cache)
    tok_ = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok_)
    pre = time.perf_counter() - t0
    out = [tok_.item()]
    while len(out) < limit and not stopped(tok, out):
        logits = model(mx.array([[out[-1]]]), cache)
        out.append(mx.argmax(logits[:, -1], axis=-1).item())
    return out, dict(prefill_ms=pre * 1e3, total_ms=(time.perf_counter() - t0) * 1e3)


def run_spec(model, draft, tok, prompt, limit, mode, Y=None, K_COPY=32):
    _patch_model(model, draft.config.target_layer_ids)
    draft.bind(model)
    bs = int(draft.config.block_size)
    mask_id = int(draft.config.mask_token_id)
    tcache, dcache = make_prompt_cache(model), make_prompt_cache(draft)
    P = list(prompt)
    t0 = time.perf_counter()
    logits, hidden, off = _prefill_target(model, mx.array(prompt), tcache, None, 2048)
    for c in dcache:
        c.offset = off
    first = mx.argmax(logits[:, -1], axis=-1); mx.eval(first, hidden)
    pre = time.perf_counter() - t0
    out = [first.item()]
    sx = SuffixIndex(P); sx.extend(out)
    pending = hidden  # target hidden states not yet given to the drafter
    cycles, diverged = [], False
    while len(out) < limit and not stopped(tok, out):
        n = len(out)
        if Y is not None and not diverged and out != Y[:n]:
            diverged = True
        cp, src = sx.propose(K_COPY) if mode == 'hybrid' else ([], None)
        a_copy = lcp(cp, Y, n) if (cp and Y is not None and not diverged) else 0
        need_draft = mode == 'dflash' or a_copy < bs - 1 or diverged
        c0 = time.perf_counter()
        d_list, a_draft = [], -1
        if need_draft:
            block = mx.array([[out[-1]] + [mask_id] * (bs - 1)])
            dl = draft(block, pending, dcache, logits_start=1)
            dt = mx.argmax(dl, axis=-1)
            if (trim := dcache[0].offset - (len(P) + n - 1)) > 0:
                _trim_recent_cache(dcache, trim)
            mx.eval(dt)
            d_list = dt[0].tolist()
            pending = None
            if Y is not None and not diverged:
                a_draft = lcp(d_list, Y, n)
        c1 = time.perf_counter()
        use_copy = mode == 'hybrid' and not diverged and a_copy > max(a_draft, 0)
        prop = cp if use_copy else d_list
        verify = mx.array([[out[-1]] + prop])
        lg = model(verify, tcache)
        h = mx.concatenate(model._hidden_states, axis=-1)
        tt = mx.argmax(lg, axis=-1); mx.eval(tt, h)
        t_list = tt[0].tolist()
        acc = next((i for i in range(len(prop)) if prop[i] != t_list[i]), len(prop))
        new = prop[:acc] + [t_list[acc]]
        trim = len(prop) - acc
        if trim > 0:
            _trim_recent_cache(tcache, trim)
        h = h[:, :acc + 1]
        pending = h if pending is None else mx.concatenate([pending, h], axis=1)
        c2 = time.perf_counter()
        cycles.append(dict(n=n, acc=acc, branch='copy' if use_copy else 'draft', drafted=need_draft,
                           a_copy=a_copy, a_draft=a_draft, width=len(prop) + 1,
                           draft_ms=(c1 - c0) * 1e3, verify_ms=(c2 - c1) * 1e3))
        out.extend(new)
        sx.extend(new)
        sx.ptr = (src + acc) if (use_copy and acc) else None
    return out, cycles, dict(prefill_ms=pre * 1e3, total_ms=(time.perf_counter() - t0) * 1e3, diverged=diverged)


def main():
    mode = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'Qwen/Qwen3-4B'
    draft_id = sys.argv[3] if len(sys.argv) > 3 else 'z-lab/Qwen3-4B-DFlash-b16'
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    tag = target.split('/')[-1]
    model, tok = load(target)
    draft = load_draft(draft_id) if mode != 'ar' else None
    items = json.loads((AUDIT / 'cohort.json').read_text())[:lim]
    if mode == 'ar':  # timing subset: every 8th item
        items = items[::8]
    ref = {}
    if mode == 'hybrid':
        ref = {json.loads(l)['id']: json.loads(l)['out'] for l in (HERE / f'dflash_{tag}.jsonl').open()}
    path = HERE / f'{mode}_{tag}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    with path.open('a') as f:
        for k, it in enumerate(items):
            if it['id'] in done or (mode == 'hybrid' and it['id'] not in ref):
                continue
            prompt = tok.encode(build_prompt(tok, it), add_special_tokens=False)
            limit = min(1400, int(1.6 * len(tok.encode(it['old'], add_special_tokens=False)) + 96))
            if mode == 'ar':
                out, meta = run_ar(model, tok, prompt, limit); cyc = []
            else:
                out, cyc, meta = run_spec(model, draft, tok, prompt, limit, mode, ref.get(it['id']))
            f.write(json.dumps(dict(id=it['id'], prompt=prompt, out=out, cycles=cyc, **meta)) + '\n'); f.flush()
            tpc = (len(out) - 1) / max(1, len(cyc))
            print(mode, k, it['id'], len(out), f'{tpc:.1f} tok/call', f"{meta['total_ms'] / 1e3:.1f}s", flush=True)


if __name__ == '__main__':
    main()
