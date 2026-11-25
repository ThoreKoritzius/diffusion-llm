"""PyTorch harness for what no serving engine implements: copy + DFlash hybrids and AR-drafter replays.

Built on dflash.model (the reference transformers implementation); the `dflash` mode reproduces dflash_generate.
Modes (all greedy, lossless):
  dflash         stock DFlash. Its output is the reference Y replayed by the AR drafters.
  hybrid_oracle  copy branch (suffix match, up to 32) + DFlash with two-branch tree semantics: both branches are
                 verified by the target itself (cache rolled back between them) and the one accepted further is kept.
                 Extra target calls are logged in `calls`; the cost model charges one tree-verify per cycle.
  hybrid_router  deployable, no oracle: copy if the suffix match is at least --router-min tokens long, else DFlash.
  ardraft        replays Y with a small AR drafter (alone, and copy + drafter), gamma 7 and 15. No target needed.
Acceptance counts are hardware independent; per-cycle timings here are eager PyTorch and used only as ratios.

Usage: python harness.py MODE --key qwen3-8b --cohort edit261 [--router-min 8] [--ar-drafter ID] [--limit N] [--device cuda]
Writes results/<key>/<cohort>/harness_<mode>.jsonl (resumes).
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from common import HERE, cohort, max_new_tokens, prompt_text, stopped, targets
from sim import SuffixIndex, align, lcp


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()
    elif dev == 'mps':
        torch.mps.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def run_spec(target, draft, tok, item, P, limit, mode, dev, eos, router_min=8, K_COPY=32):
    from dflash.model import _crop_to, _make_cache, _output_head, _raw_input_embeddings, _draft_value, extract_context_feature
    bs, mask_id = draft.block_size, draft.mask_token_id
    scale = float(_draft_value(draft.config, 'input_embedding_scale', 1.0))
    head = _output_head(target)
    n_in = len(P)
    pos = torch.arange(n_in + limit + 64, device=dev)[None]
    ct, cd = _make_cache(target.config), _make_cache(draft.config)
    t0 = sync(dev)
    o = target(torch.tensor([P], device=dev), position_ids=pos[:, :n_in], past_key_values=ct, use_cache=True,
               logits_to_keep=1, output_hidden_states=True)
    out = [int(o.logits[0, -1].argmax())]
    pending = extract_context_feature(o.hidden_states, draft.target_layer_ids)
    prefill_ms = (sync(dev) - t0) * 1e3
    sx = SuffixIndex(P); sx.extend(out)
    cycles = []

    def verify(prop, start):
        """Target forward on [last] + prop; returns (accepted, new tokens, hidden of accepted positions)."""
        blk = torch.tensor([[out[-1]] + prop], device=dev)
        o = target(blk, position_ids=pos[:, start:start + blk.shape[1]], past_key_values=ct, use_cache=True,
                   output_hidden_states=True)
        post = o.logits[0].argmax(-1).tolist()
        acc = next((i for i in range(len(prop)) if prop[i] != post[i]), len(prop))
        _crop_to(ct, start + acc + 1)
        return acc, prop[:acc] + [post[acc]], extract_context_feature(o.hidden_states, draft.target_layer_ids)[:, :acc + 1]

    def draft_block(start):
        nonlocal pending, ctx_len
        ctx_len = pending.shape[1]  # queued target hidden states the drafter must ingest (grows during copy runs)
        block = torch.tensor([[out[-1]] + [mask_id] * (bs - 1)], device=dev)
        h = draft(target_hidden=pending, noise_embedding=_raw_input_embeddings(target, block, scale),
                  position_ids=pos[:, start - pending.shape[1]:start + bs], past_key_values=cd, use_cache=True)[:, 1 - bs:, :]
        _crop_to(cd, start)
        pending = None
        return draft.compute_logits(h, head).argmax(-1)[0].tolist()

    while len(out) < limit and not stopped(tok, item, out, eos):
        start = n_in + len(out) - 1  # position of the newest accepted token (first element of the block)
        n = len(out)
        cp, src = sx.propose(K_COPY) if mode.startswith('hybrid') else ([], None)
        m_len = getattr(sx, 'last_len', 0) if cp else 0
        c0 = sync(dev)
        a_copy = a_draft = -1
        drafted, use_copy, calls, ctx_len = False, False, 0, 0
        if mode == 'dflash' or (mode == 'hybrid_router' and not (cp and m_len >= router_min)):
            d = draft_block(start); drafted = True
            acc, new, hid = verify(d, start); calls = 1
            a_draft = acc
        elif mode == 'hybrid_router':
            acc, new, hid = verify(cp, start); calls = 1
            use_copy, a_copy = True, acc
        else:
            # hybrid_oracle = two-branch tree semantics, measured with the target itself (no reference output):
            # verify copy; if it cannot cover a full block, also draft + verify, keep the branch accepted further.
            if cp:
                a_copy, new_c, hid_c = verify(cp, start); calls += 1
            if a_copy >= bs - 1:
                acc, new, hid, use_copy = a_copy, new_c, hid_c, True
            else:
                if cp:
                    _crop_to(ct, start)
                d = draft_block(start); drafted = True
                a_draft, new, hid = verify(d, start); calls += 1
                acc = a_draft
                if a_copy > a_draft:
                    _crop_to(ct, start)
                    acc, new, hid = verify(cp, start); calls += 1
                    use_copy = True
        pending = hid if pending is None else torch.cat([pending, hid], dim=1)
        c1 = sync(dev)
        cycles.append(dict(n=n, acc=acc, branch='copy' if use_copy else 'draft', drafted=drafted, a_copy=a_copy,
                           a_draft=a_draft, match_len=m_len, calls=calls, ctx_len=ctx_len,
                           cycle_ms=(c1 - c0) * 1e3))
        cut = next((j for j, t in enumerate(new) if t in eos), None)
        if cut is not None:  # tokens accepted past end-of-sequence are not output
            new = new[:cut + 1]
        out.extend(new)
        sx.extend(new)
        sx.ptr = (src + acc) if (use_copy and acc) else None
    return out, cycles, dict(prefill_ms=prefill_ms, total_ms=(sync(dev) - t0) * 1e3)


class ARDrafter:
    def __init__(self, model, dev):
        self.m, self.dev = model, dev

    def reset(self):
        self.cache, self.n = DynamicCache(), 0

    @torch.inference_mode()
    def draft(self, ctx, Y, i, gamma):
        o = self.m(torch.tensor([ctx[self.n:]], device=self.dev), past_key_values=self.cache, use_cache=True)
        self.n = len(ctx)
        out = []
        for _ in range(gamma):
            t = int(o.logits[0, -1].argmax())
            out.append(t)
            if i + len(out) - 1 >= len(Y) or t != Y[i + len(out) - 1] or len(out) == gamma:
                break
            o = self.m(torch.tensor([[t]], device=self.dev), past_key_values=self.cache, use_cache=True)
        extra = self.cache.get_seq_length() - self.n
        if extra > 0:
            self.cache.crop(-extra)
        return out


def replay_ar(P, Y, lab, dr, policy, gamma):
    dr.reset()
    sx = SuffixIndex(P); sx.extend(Y[:1])
    i, cyc = 1, []
    while i < len(Y):
        cp, src = sx.propose(32) if policy == 'hybrid' else ([], None)
        a_copy = lcp(cp, Y, i)
        a_d, drafted = -1, False
        if policy == 'alone' or a_copy < gamma:
            a_d, drafted = lcp(dr.draft(P + Y[:i], Y, i, gamma), Y, i), True
        a = max(a_copy, a_d)
        cyc.append(dict(n=i, acc=a, drafted=drafted, a_draft=a_d, novel=lab[i] < 0))
        step = min(a + 1, len(Y) - i)
        sx.extend(Y[i:i + step]); sx.ptr = (src + a) if (cp and a == a_copy and a) else None
        i += step
    return cyc


def eos_ids(tok, model_id):
    from transformers import GenerationConfig
    ids = {tok.eos_token_id}
    try:
        g = GenerationConfig.from_pretrained(model_id).eos_token_id
        ids |= set(g if isinstance(g, list) else [g])
    except OSError:
        pass
    for t in ('<|im_end|>', '<|eot_id|>', '<|endoftext|>'):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return {i for i in ids if i is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dflash', 'hybrid_oracle', 'hybrid_router', 'ardraft'])
    ap.add_argument('--key', required=True)
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--router-min', type=int, default=8)
    ap.add_argument('--ar-drafter', default=None, help='defaults to the target\'s small draft model')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    a = ap.parse_args()
    dev = a.device
    t = targets()[a.key]
    ar_drafter = a.ar_drafter or t['draft']
    out_dir = HERE / 'results' / a.key / a.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(t['model'])
    eos = eos_ids(tok, t['model'])
    items = cohort(a.cohort, a.limit)
    ref_path = out_dir / 'harness_dflash.jsonl'
    ref = {json.loads(l)['id']: json.loads(l)['out'] for l in ref_path.open()} if ref_path.exists() else {}
    name = {'hybrid_router': f'hybrid_router{a.router_min}', 'ardraft': f"ardraft_{ar_drafter.split('/')[-1]}"}.get(a.mode, a.mode)
    path = out_dir / f'harness_{name}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    if a.mode == 'ardraft' and not ref:
        raise SystemExit('run the dflash mode first: its outputs are replayed by the AR drafters')
    if a.mode == 'ardraft':
        dr = ARDrafter(AutoModelForCausalLM.from_pretrained(ar_drafter, dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval(), dev)
    else:
        from dflash.model import DFlashDraftModel
        target = AutoModelForCausalLM.from_pretrained(t['model'], dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
        draft = DFlashDraftModel.from_pretrained(t['dflash'], dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
    with path.open('a') as f:
        for k, it in enumerate(items):
            if it['id'] in done or (a.mode == 'ardraft' and it['id'] not in ref):
                continue
            P = tok.encode(prompt_text(tok, it), add_special_tokens=False)
            limit = max_new_tokens(tok, it)
            if a.mode == 'ardraft':
                Y = ref[it['id']]
                if not 8 <= len(Y) < 1400:
                    continue
                lab = align(P, Y)
                rec = dict(id=it['id'], L=len(Y), P=len(P))
                for policy, gamma in (('alone', 7), ('hybrid', 7), ('hybrid', 15)):
                    rec[f'{policy}{gamma}'] = replay_ar(P, Y, lab, dr, policy, gamma)
                msg = {p: round((len(Y) - 1) / len(rec[p]), 1) for p in ('alone7', 'hybrid7', 'hybrid15')}
            else:
                out, cyc, meta = run_spec(target, draft, tok, it, P, limit, a.mode, dev, eos, a.router_min)
                rec = dict(id=it['id'], prompt=P, out=out, cycles=cyc, **meta)
                msg = f'{(len(out) - 1) / max(1, len(cyc)):.1f} tok/call'
            f.write(json.dumps(rec) + '\n'); f.flush()
            if dev == 'mps':
                torch.mps.empty_cache()
            print(a.key, a.cohort, name, k, it['id'], msg, flush=True)


if __name__ == '__main__':
    main()
