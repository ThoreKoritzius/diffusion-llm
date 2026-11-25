"""Batch-one client for a running vLLM server: latency, prefill time and exact per-request speculative stats.

Per edit, sequentially:
  1. completion with max_tokens=1        -> time to first token (prefill + overhead; prefix caching must be off)
  2. full completion, greedy, stop "```" -> end-to-end latency, output tokens, text
  Prometheus counters are read before/after each full request; with one request in flight the deltas are exact
  per-request draft steps, drafted tokens and accepted tokens.

Usage: python bench_vllm.py --name dflash15 --target Qwen/Qwen3-8B [--base-url http://127.0.0.1:8000] [--limit N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from common import HERE, cohort, group, max_new_tokens, prompt_text

COUNTERS = {'drafts': 'vllm:spec_decode_num_drafts_total',
            'draft_tokens': 'vllm:spec_decode_num_draft_tokens_total',
            'accepted': 'vllm:spec_decode_num_accepted_tokens_total'}


def scrape(base):
    text = requests.get(f'{base}/metrics', timeout=30).text
    out = {}
    for key, name in COUNTERS.items():
        vals = [float(m.group(1)) for m in re.finditer(rf'^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$', text, re.M)]
        out[key] = sum(vals)
    return out


def complete(base, model, prompt, max_tokens):
    body = dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0, stop=['```'], seed=0)
    t0 = time.perf_counter()
    r = requests.post(f'{base}/v1/completions', json=body, timeout=900)
    dt = (time.perf_counter() - t0) * 1e3
    r.raise_for_status()
    j = r.json()
    return dt, j['choices'][0]['text'], j['usage']['completion_tokens']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--target', default='Qwen/Qwen3-8B')
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--ids', default=None, help='json file with a list of {"id": ...} or ids; restricts the requests')
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.target)
    items = cohort(a.limit)
    if a.ids:
        keep = {x['id'] if isinstance(x, dict) else x for x in json.loads(open(a.ids).read())}
        items = [it for it in items if it['id'] in keep]
    out_dir = HERE / 'results' / a.target.split('/')[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'vllm_{a.name}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    for it in items[:a.warmup]:
        complete(a.base_url, a.target, prompt_text(tok, it), 64)
    with path.open('a') as f:
        for k, it in enumerate(items):
            if it['id'] in done:
                continue
            prompt = prompt_text(tok, it)
            ttft, _, _ = complete(a.base_url, a.target, prompt, 1)
            before = scrape(a.base_url)
            lat, text, n_out = complete(a.base_url, a.target, prompt, max_new_tokens(tok, it))
            after = scrape(a.base_url)
            d = {key: after[key] - before[key] for key in COUNTERS}
            rec = dict(id=it['id'], group=group(it), config=a.name, prompt_tokens=len(tok.encode(prompt, add_special_tokens=False)),
                       ttft_ms=ttft, latency_ms=lat, out_tokens=n_out, text=text, **d)
            f.write(json.dumps(rec) + '\n'); f.flush()
            tau = 1 + d['accepted'] / d['drafts'] if d['drafts'] else 1.0
            print(a.name, k, it['id'], n_out, f'{lat:.0f} ms', f'tau {tau:.2f}', flush=True)


if __name__ == '__main__':
    main()
