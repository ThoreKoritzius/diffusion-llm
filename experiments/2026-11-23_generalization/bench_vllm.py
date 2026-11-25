"""Batch-one client for a running vLLM server (see run_vllm_suite.sh). Per request: time to first token (max_tokens=1,
prefix caching off), full greedy completion, and exact speculative stats from Prometheus counter deltas.

Usage: python bench_vllm.py --key qwen3-8b --name dflash --cohort edit261 [--limit N]
Writes results/<key>/<cohort>/vllm_<name>.jsonl (resumes).
"""
import argparse
import json
import re
import time

import requests
from transformers import AutoTokenizer

from common import HERE, cohort, group, max_new_tokens, prompt_text, stop_strings, targets

COUNTERS = {'drafts': 'vllm:spec_decode_num_drafts_total',
            'draft_tokens': 'vllm:spec_decode_num_draft_tokens_total',
            'accepted': 'vllm:spec_decode_num_accepted_tokens_total'}


def scrape(base):
    text = requests.get(f'{base}/metrics', timeout=30).text
    return {k: sum(float(m.group(1)) for m in re.finditer(rf'^{re.escape(n)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$', text, re.M))
            for k, n in COUNTERS.items()}


def complete(base, model, prompt, max_tokens, stop):
    body = dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0, stop=stop, seed=0)
    t0 = time.perf_counter()
    r = requests.post(f'{base}/v1/completions', json=body, timeout=1200)
    dt = (time.perf_counter() - t0) * 1e3
    r.raise_for_status()
    j = r.json()
    return dt, j['choices'][0]['text'], j['usage']['completion_tokens'], j['choices'][0].get('finish_reason')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--warmup', type=int, default=3)
    a = ap.parse_args()
    model = targets()[a.key]['model']
    tok = AutoTokenizer.from_pretrained(model)
    items = cohort(a.cohort, a.limit)
    out_dir = HERE / 'results' / a.key / a.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'vllm_{a.name}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    if len(done) >= len(items):
        return
    for it in items[:a.warmup]:
        complete(a.base_url, model, prompt_text(tok, it), 64, stop_strings(it))
    with path.open('a') as f:
        for k, it in enumerate(items):
            if it['id'] in done:
                continue
            prompt = prompt_text(tok, it)
            ttft = complete(a.base_url, model, prompt, 1, stop_strings(it))[0]
            before = scrape(a.base_url)
            lat, text, n_out, finish = complete(a.base_url, model, prompt, max_new_tokens(tok, it), stop_strings(it))
            after = scrape(a.base_url)
            d = {key: after[key] - before[key] for key in COUNTERS}
            f.write(json.dumps(dict(id=it['id'], group=group(it), config=a.name, prompt_tokens=len(tok.encode(prompt, add_special_tokens=False)),
                                    ttft_ms=ttft, latency_ms=lat, out_tokens=n_out, finish=finish, text=text, **d)) + '\n')
            f.flush()
            tau = 1 + d['accepted'] / d['drafts'] if d['drafts'] else 1.0
            print(a.key, a.cohort, a.name, k, it['id'], n_out, f'{lat:.0f} ms', f'tau {tau:.2f}', flush=True)


if __name__ == '__main__':
    main()
