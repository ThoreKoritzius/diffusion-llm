"""Build the three cohorts (run once on the Mac; outputs are shipped with the package, no download on the GPU box).

  edit261  the 261 edits of the A100 run (CanItEdit functions/files + EditPackFT), full rewrite
  ceb150   CodeEditorBench code_debug_primary: 50 per language (python3, cpp, java), stratified by bug type,
           before 40-600 tokens; instruction names the bug category; full rewrite
  diff104  the 104 CanItEdit full files, answered as search/replace blocks instead of a full rewrite
Each item: id, cohort, regime, source, language, format, instruction, old, ref.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
A100 = HERE.parent / '2026-11-22_h100-bench'


def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    n = lambda s: len(tok.encode(s, add_special_tokens=False))
    out = HERE / 'data'
    out.mkdir(exist_ok=True)
    base = json.loads((A100 / 'data' / 'cohort.json').read_text())
    e261 = [dict(it, cohort='edit261', language='python', format='rewrite') for it in base]
    (out / 'edit261.json').write_text(json.dumps(e261))
    d104 = [dict(it, id=it['id'].replace('cie-file', 'diff'), cohort='diff104', language='python', format='diff')
            for it in base if it['regime'] == 'file']
    (out / 'diff104.json').write_text(json.dumps(d104))
    rows = [json.loads(l) for l in (HERE / 'raw' / 'code_debug_primary.jsonl').open()]
    lang_name = {'python3': 'Python', 'cpp': 'C++', 'java': 'Java'}
    rng = random.Random(1123)
    ceb = []
    for lang in ('python3', 'cpp', 'java'):
        by_type = defaultdict(list)
        for r in rows:
            if r['code_language'] != lang:
                continue
            before = re.sub(r'^(python3?|cpp|java)\n', '', r['incorrect_solutions'])
            after = r['solutions']
            if not before.strip() or not after.strip() or before.strip() == after.strip():
                continue
            if not (40 <= n(before) <= 600) or n(after) > 700:
                continue
            by_type[r['type']].append(dict(id=f"ceb-{lang}-{r['idx']}", cohort='ceb150', regime='program', source='codeeditorbench',
                                           language={'python3': 'python'}.get(lang, lang), format='rewrite',
                                           instruction=f"Fix the bug in this {lang_name[lang]} program (bug category: {r['type']}).",
                                           old=before, ref=after))
        for v in by_type.values():
            rng.shuffle(v)
        picked, types = [], sorted(by_type)
        while len(picked) < 50 and any(by_type.values()):  # round-robin over bug types
            for t in types:
                if by_type[t] and len(picked) < 50:
                    picked.append(by_type[t].pop())
        ceb += picked
    (out / 'ceb150.json').write_text(json.dumps(ceb))
    from collections import Counter
    print('edit261', len(e261), '| diff104', len(d104), '| ceb150', len(ceb), Counter(c['language'] for c in ceb),
          '| median old tokens ceb', sorted(n(c['old']) for c in ceb)[len(ceb) // 2])


if __name__ == '__main__':
    main()
