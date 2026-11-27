"""Text-to-SQL with a large off-the-shelf target (Qwen3-8B, zero-shot), plain vs DFlash vs copy+DFlash in vLLM.

Same 1,000 test queries and the same SQLite execution scoring as the SQL results chart (bench/common.py on main,
GPU session 2: first 1,000 rows of gretelai/synthetic_text_to_sql test with a non-empty prompt; exec accuracy over
the rows whose gold SQL runs in SQLite). Batch 1, greedy, timing per request = full completion over HTTP, as in
bench_vllm.py.

  python sqlbench.py prepare                       -> data/sql1000.json            (Mac or box)
  python sqlbench.py run --key qwen3-8b --name dflash [--limit N]   (vLLM server running, see run_sql_suite.sh)
  python sqlbench.py mlx --model Qwen/Qwen3-4B --limit 200          (Mac preview, accuracy only)
  python sqlbench.py score --key qwen3-8b          -> results/<key>/sql1000/summary.json
"""
import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'sql1000.json'
PREFILL, STOP, MAX_NEW = '```sql\n', ['```'], 256


# ---- scoring: verbatim from bench/common.py (main) ----
def normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _clean_context(context):
    context = re.sub(r"\bCREATE\s+SCHEMA\s+[^;]+;", "", context, flags=re.I)
    return re.sub(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", r"\2", context)


def _db_state(conn):
    out = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        order = ", ".join(f'"{c}"' for c in cols) if cols else "rowid"
        out.append((table, tuple(conn.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall())))
    return tuple(out)


def run_sql(context, sql):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_clean_context(context))
        cur = conn.execute(sql)
        if cur.description is not None:
            return "rows", tuple(cur.fetchall())
        conn.commit()
        return "state", _db_state(conn)
    finally:
        conn.close()


def score(context, gold, pred):
    out = {"exact": normalize_sql(pred) == normalize_sql(gold), "valid": False, "exec": False, "gold_ok": False}
    try:
        gold_res = run_sql(context, gold)
        out["gold_ok"] = True
    except Exception:  # noqa: BLE001
        gold_res = None
    if pred.strip():
        try:
            pred_res = run_sql(context, pred)
            out["valid"] = True
            out["exec"] = gold_res is not None and pred_res == gold_res
        except Exception:  # noqa: BLE001
            pass
    return out
# ---- end of bench/common.py ----


def _num_eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b or abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))
    return a == b


def _rows_eq(p, g, ordered):
    if len(p) != len(g):
        return False
    if not ordered:
        p, g = sorted(p, key=repr), sorted(g, key=repr)
    return all(len(x) == len(y) and all(_num_eq(a, b) for a, b in zip(x, y)) for x, y in zip(p, g))


def _subset_match(small, big, ordered):
    """Columns of `small` each equal (row-aligned) to a distinct column of `big`, after sorting rows if unordered."""
    if len(small) != len(big) or not small:
        return False
    if not ordered:  # align rows by sorting both on the matched columns: try every candidate mapping greedily
        big = sorted(big, key=repr)
    ncs, ncb = len(small[0]), len(big[0])
    used, mapping = set(), []
    for i in range(ncs):
        col = [r[i] for r in small]
        hit = None
        for j in range(ncb):
            if j in used:
                continue
            other = [r[j] for r in big]
            same = (all(_num_eq(a, b) for a, b in zip(col, other)) if ordered else
                    all(_num_eq(a, b) for a, b in zip(sorted(col, key=repr), sorted(other, key=repr))))
            if same:
                hit = j; break
        if hit is None:
            return False
        used.add(hit); mapping.append(hit)
    proj = [tuple(r[j] for j in mapping) for r in big]
    return _rows_eq(list(small), proj, ordered)


def score_lenient(context, gold, pred):
    """Strict exec, or the same answer up to (1) row order when the gold has no ORDER BY, (2) extra or missing
    columns (one result's columns are a subset of the other's, same rows), (3) numeric rounding (rel. 1e-4).
    DML table states, broken golds and erroring predictions stay wrong."""
    s = score(context, gold, pred)
    if s['exec'] or not s['gold_ok'] or not s['valid']:
        return s['exec']
    gk, gv = run_sql(context, gold)
    pk, pv = run_sql(context, pred)
    if gk != 'rows' or pk != 'rows' or not gv or not pv:
        return False
    ordered = 'order by' in gold.lower()
    if len(gv[0]) == len(pv[0]) and _rows_eq(list(pv), list(gv), ordered):
        return True
    small, big = (pv, gv) if len(pv[0]) < len(gv[0]) else (gv, pv)
    return _subset_match(list(small), list(big), ordered)


def prompt_text(tok, it):
    user = ('Write a SQLite query that answers the question, using the database below. Reply with only the SQL query '
            f"in a single ```sql code block.\n\nDatabase:\n```sql\n{it['context'].strip()}\n```\n\nQuestion: {it['prompt'].strip()}")
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + PREFILL


def extract(text):
    return text.split('```')[0].strip()


def prepare():
    from datasets import load_dataset
    ds = load_dataset('gretelai/synthetic_text_to_sql', split='test')
    ds = ds.filter(lambda ex: ex['sql_prompt'].strip() != '').select(range(1000))
    items = [dict(id=f'sql-{i:04d}', prompt=ex['sql_prompt'], context=ex.get('sql_context', '') or '', sql=ex['sql'])
             for i, ex in enumerate(ds)]
    DATA.write_text(json.dumps(items))
    print(len(items), 'items ->', DATA)


def items(limit=None):
    it = json.loads(DATA.read_text())
    return it[:limit] if limit else it


def run(a):
    import requests  # noqa: F401
    from transformers import AutoTokenizer
    from bench_vllm import COUNTERS, complete, scrape
    from common import targets
    model = targets()[a.key]['model']
    tok = AutoTokenizer.from_pretrained(model)
    out = HERE / 'results' / a.key / 'sql1000'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'vllm_{a.name}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    its = items(a.limit)
    for it in its[:3]:
        complete(a.base_url, model, prompt_text(tok, it), 32, STOP)
    with path.open('a') as f:
        for k, it in enumerate(its):
            if it['id'] in done:
                continue
            before = scrape(a.base_url)
            lat, text, n_out, finish = complete(a.base_url, model, prompt_text(tok, it), MAX_NEW, STOP)
            after = scrape(a.base_url)
            d = {c: after[c] - before[c] for c in COUNTERS}
            f.write(json.dumps(dict(id=it['id'], config=a.name, latency_ms=lat, out_tokens=n_out, finish=finish,
                                    text=text, pred=extract(text), **d)) + '\n')
            f.flush()
            if k % 50 == 0:
                print(a.key, a.name, k, f'{lat:.0f} ms', n_out, flush=True)


def mlx(a):
    from mlx_lm import generate, load
    model, tok = load(a.model)
    name = a.model.split('/')[-1].lower()
    path = HERE / 'results' / f'mlx-{name}' / 'sql1000'
    path.mkdir(parents=True, exist_ok=True)
    path = path / 'mlx_ar.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    with path.open('a') as f:
        for k, it in enumerate(items(a.limit)):
            if it['id'] in done:
                continue
            t0 = time.perf_counter()
            text = generate(model, tok, prompt=prompt_text(tok, it), max_tokens=MAX_NEW, verbose=False)
            f.write(json.dumps(dict(id=it['id'], config='mlx_ar', latency_ms=(time.perf_counter() - t0) * 1e3,
                                    text=text, pred=extract(text))) + '\n')
            f.flush()
            if k % 20 == 0:
                print(k, extract(text)[:100], flush=True)


def summarize(a):
    gold = {it['id']: it for it in items()}
    root = HERE / 'results' / a.key / 'sql1000'
    res = {}
    for p in sorted(root.glob('*.jsonl')):
        rows = [json.loads(l) for l in p.open()]
        sc = [score(gold[r['id']]['context'], gold[r['id']]['sql'], r['pred']) for r in rows]
        ok = [s for s in sc if s['gold_ok']]
        len_ok = [score_lenient(gold[r['id']]['context'], gold[r['id']]['sql'], r['pred']) for r, s in zip(rows, sc) if s['gold_ok']]
        lat = sorted(r['latency_ms'] for r in rows)
        res[p.stem] = dict(n=len(rows), n_exec=len(ok), exec=sum(s['exec'] for s in ok) / max(1, len(ok)),
                           exec_lenient=sum(len_ok) / max(1, len(ok)),
                           exact=sum(s['exact'] for s in sc) / len(sc), valid=sum(s['valid'] for s in sc) / len(sc),
                           lat_mean_ms=sum(lat) / len(lat), lat_p50_ms=lat[len(lat) // 2],
                           out_tokens=sum(r.get('out_tokens', 0) for r in rows) / len(rows))
        if rows[0].get('drafts') is not None and sum(r['drafts'] for r in rows):
            res[p.stem]['tokens_per_call'] = 1 + sum(r['accepted'] for r in rows) / sum(r['drafts'] for r in rows)
    if 'vllm_ar' in res:
        base = {r['id']: r for r in map(json.loads, (root / 'vllm_ar.jsonl').open())}
        for name in res:
            rows = [json.loads(l) for l in (root / f'{name}.jsonl').open()]
            res[name]['same_as_ar'] = sum(base[r['id']]['text'] == r['text'] for r in rows if r['id'] in base) / len(rows)
            paired = [r for r in rows if r['id'] in base]  # same queries for both (diagnosis runs cover only 300)
            res[name]['speedup_vs_ar'] = (sum(base[r['id']]['latency_ms'] for r in paired)
                                          / sum(r['latency_ms'] for r in paired))
    (root / 'summary.json').write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:22s} exec {v['exec']:.3f} lenient {v['exec_lenient']:.3f} (n={v['n_exec']}) valid {v['valid']:.3f} exact {v['exact']:.3f} "
              f"mean {v['lat_mean_ms']:.1f} ms p50 {v['lat_p50_ms']:.1f}" +
              (f" | {v['speedup_vs_ar']:.2f}x vs AR, same {v['same_as_ar']:.0%}" if 'speedup_vs_ar' in v else '') +
              (f" | tok/call {v['tokens_per_call']:.2f}" if 'tokens_per_call' in v else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'run', 'mlx', 'score'])
    ap.add_argument('--key', default='qwen3-8b')
    ap.add_argument('--name', default='ar')
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    {'prepare': lambda: prepare(), 'run': lambda: run(a), 'mlx': lambda: mlx(a), 'score': lambda: summarize(a)}[a.cmd]()
