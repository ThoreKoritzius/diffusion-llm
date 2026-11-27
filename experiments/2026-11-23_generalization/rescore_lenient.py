"""Strict and lenient execution accuracy for every text-to-SQL run on the same 1,000 test queries (777 checkable).

Lenient (sqlbench.score_lenient): the strict check, or the same answer up to row order (when the gold has no ORDER BY),
extra or missing columns, or numeric rounding. Applied with one rule to every model: the GPU session 2 runs (Ettin
150M/400M, Qwen2.5-0.5B, its drafters) and the Qwen3-8B runs here. Writes results/lenient_scores.json.
"""
import json
from pathlib import Path

import sqlbench as S

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / '2026-10-14_gpu-session-2' / 'data' / 'bench'
RUNS = ['ettin150-ar-10ep-vllm-pr61276', 'ettin150-diffusion-iso-10ep-distilled-bf16-cudagraphs',
        'ettin400-diffusion-iso-10ep-distilled-bf16-cudagraphs', 'sql-ar-qwen25-coder-0.5b-ar-vllm',
        'sql-ar-qwen25-coder-0.5b-ar-vllm-dflash', 'sql-ar-qwen25-coder-0.5b-ar-vllm-eagle3']


def evaluate(items, preds):
    st = le = n = 0
    for it, pred in zip(items, preds):
        s = S.score(it['context'], it['sql'], pred)
        if s['gold_ok']:
            n += 1; st += s['exec']; le += S.score_lenient(it['context'], it['sql'], pred)
    return dict(n_exec=n, exec=st / n, exec_lenient=le / n)


def main():
    items = S.items()
    out = {}
    for name in RUNS:
        rows = [json.loads(l) for l in (S2 / f'{name}.jsonl').open()]
        assert all(r['gold'] == it['sql'] for r, it in zip(rows, items)), name  # same queries, same order
        out[name] = evaluate(items, [r['pred'] for r in rows])
    for p in sorted((HERE / 'results' / 'qwen3-8b' / 'sql1000').glob('vllm_*.jsonl')):
        rows = {r['id']: r for r in map(json.loads, p.open())}
        if len(rows) == len(items):
            out['qwen3-8b ' + p.stem[5:]] = evaluate(items, [rows[it['id']]['pred'] for it in items])
    (HERE / 'results' / 'lenient_scores.json').write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:58s} strict {v['exec']:.3f}  lenient {v['exec_lenient']:.3f}")


if __name__ == '__main__':
    main()
