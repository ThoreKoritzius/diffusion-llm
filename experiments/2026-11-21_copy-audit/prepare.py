"""Build the edit cohort: CanItEdit (function + full-file) and an EditPackFT sample.

No dataset Python is executed; selection uses ast parsing only.
"""
import ast
import hashlib
import io
import json
import random
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CIE_REV = '3c07f38b1f9385f3214fcea94d4664c79df0d36a'
EPF_REV = '067b4c882f44eca19329fedd275438fc3c060f3f'
TOK = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'


def functions(source):
    """Map qualified name -> (start_line, end_line) of the full def incl. decorators."""
    out = {}
    def visit(node, path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = path + (child.name,)
                if not isinstance(child, ast.ClassDef):
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list]) - 1
                    out[name] = (start, child.end_lineno)
                visit(child, name)
            else:
                visit(child, path)
    visit(ast.parse(source), ())
    return out


def single_function_edit(before, after):
    """Return (old_func, new_func) if exactly the text of one function differs."""
    try:
        fa, fb = functions(before), functions(after)
    except (SyntaxError, ValueError, RecursionError):
        return None
    la, lb = before.splitlines(keepends=True), after.splitlines(keepends=True)
    hits = []
    for key in fa.keys() & fb.keys():
        (s1, e1), (s2, e2) = fa[key], fb[key]
        old, new = ''.join(la[s1:e1]), ''.join(lb[s2:e2])
        if old == new:
            continue
        if ''.join(la[:s1]) == ''.join(lb[:s2]) and ''.join(la[e1:]) == ''.join(lb[e2:]):
            hits.append((len(key), old, new))
    if not hits:
        return None
    hits.sort(key=lambda h: -h[0])  # innermost function whose outside is unchanged
    return hits[0][1], hits[0][2]


def fetch(url):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pq.read_table(io.BytesIO(r.content))


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    n = lambda s: len(tok.encode(s, add_special_tokens=False))
    items = []
    cie = fetch(f'https://huggingface.co/datasets/nuprl/CanItEdit/resolve/{CIE_REV}/data/test-00000-of-00001.parquet').to_pylist()
    for row in cie:
        instr = row['instruction_descriptive']
        items.append(dict(id=f"cie-file-{row['id']}", regime='file', source='canitedit',
                          instruction=instr, old=row['before'], ref=row['after']))
        pair = single_function_edit(row['before'], row['after'])
        if pair:
            items.append(dict(id=f"cie-func-{row['id']}", regime='function', source='canitedit',
                              instruction=instr, old=pair[0], ref=pair[1]))
    epf = fetch(f'https://huggingface.co/datasets/nuprl/EditPackFT/resolve/{EPF_REV}/data/train-00000-of-00001.parquet')
    epf = epf.select(['commit', 'old_contents', 'new_contents', 'subject']).to_pylist()
    rng = random.Random(1121)
    rng.shuffle(epf)
    seen, picked = set(), 0
    for row in epf:
        if picked >= 120:
            break
        subj = (row['subject'] or '').strip()
        if len(subj.split()) < 4 or not row['old_contents'] or not row['new_contents']:
            continue
        pair = single_function_edit(row['old_contents'], row['new_contents'])
        if not pair or not (30 <= n(pair[0]) <= 400) or n(pair[1]) > 450:
            continue
        h = hashlib.sha256((pair[0] + '\0' + pair[1]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        picked += 1
        items.append(dict(id=f"epf-{row['commit'][:10]}", regime='function', source='editpackft',
                          instruction=subj, old=pair[0], ref=pair[1]))
    items = [it for it in items if n(it['old']) <= 1200 and n(it['ref']) <= 1300]
    (HERE / 'cohort.json').write_text(json.dumps(items))
    from collections import Counter
    print(Counter((i['source'], i['regime']) for i in items))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        main()
