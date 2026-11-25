"""Throughput under load: N concurrent clients replay the first M edits of a cohort against a running vLLM server.

Speculative decoding usually loses ground as the batch grows (verification competes with other requests for compute).
Records total wall time, output tokens, mean and p90 request latency per concurrency level.
Usage: python bench_concurrency.py --key qwen3-8b --name dflash --levels 1 4 16 [--cohort edit261] [--n 64]
Writes results/<key>/concurrency/<name>.json
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from transformers import AutoTokenizer

from common import HERE, cohort, max_new_tokens, prompt_text, stop_strings, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--cohort', default='edit261')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    a = ap.parse_args()
    model = targets()[a.key]['model']
    tok = AutoTokenizer.from_pretrained(model)
    items = cohort(a.cohort)[:a.n]
    reqs = [dict(model=model, prompt=prompt_text(tok, it), max_tokens=max_new_tokens(tok, it), temperature=0.0,
                 stop=stop_strings(it), seed=0) for it in items]

    def one(body):
        t0 = time.perf_counter()
        r = requests.post(f'{a.base_url}/v1/completions', json=body, timeout=1800)
        r.raise_for_status()
        return time.perf_counter() - t0, r.json()['usage']['completion_tokens']

    for body in reqs[:2]:
        one(dict(body, max_tokens=32))
    out = {}
    for c in a.levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, reqs))
        wall = time.perf_counter() - t0
        lat = [r[0] for r in res]
        toks = sum(r[1] for r in res)
        out[c] = dict(wall_s=wall, out_tokens=toks, tok_per_s=toks / wall, mean_latency_s=float(np.mean(lat)),
                      p90_latency_s=float(np.percentile(lat, 90)))
        print(a.key, a.name, 'concurrency', c, f'{toks / wall:.0f} tok/s', f'mean latency {np.mean(lat):.2f}s', flush=True)
    d = HERE / 'results' / a.key / 'concurrency'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{a.name}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
