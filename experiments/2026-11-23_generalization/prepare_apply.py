"""Build cohort apply100: realistic "fast apply" edits on large real files (Mac, CPU only; output shipped in data/).

Source: the gold patches of SWE-bench Verified (Python, 12 repositories) and SWE-bench Multilingual (JS/TS, Go,
Java, Rust, C/C++, PHP, Ruby). For every non-test file a gold patch modifies, the original file is fetched at the
task's base commit from raw.githubusercontent.com and the patch hunks are applied to get the reference. Kept:
files of 800-3000 Qwen3 tokens (file + edit + output must fit the 8192-token context), no ``` inside, changed
lines <= 40%. 50 Python files (at most 8 per repository) + 50 multilingual files (at most 3 per repository,
round-robin over languages). (CommitPackFT was tried first: its files are all under ~800 tokens.)

Task = the "apply" step of coding agents (Cursor / Morph / Relace style): the model gets the original file and an
edit snippet holding only the changed regions (new-side lines with 2 lines of context), separated by
"... existing code ..." markers, and must output the complete updated file.
Each item: id, cohort, regime='file', source, language, format='apply', instruction, old, edit, ref (+ stats).

Usage: python prepare_apply.py     (fetched files are cached in raw/swebench/)
"""
import difflib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw' / 'swebench'
EXT = {'.py': 'python', '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.ts': 'typescript',
       '.tsx': 'typescript', '.go': 'go', '.java': 'java', '.rs': 'rust', '.c': 'c', '.h': 'c', '.cc': 'cpp',
       '.cpp': 'cpp', '.hpp': 'cpp', '.php': 'php', '.rb': 'ruby'}
COMMENT = {'python': '#', 'ruby': '#'}
LO, HI = 800, 3000
CONTEXT = 2


def file_patches(patch):
    """Unified diff -> {path: [hunk, ...]} for modified files; hunk = (old_start, [lines with ' ', '-', '+'])."""
    out, cur, hunk = {}, None, None
    for line in patch.splitlines():
        if line.startswith('diff --git'):
            cur, hunk = None, None
        elif line.startswith('--- '):
            src = line[4:].strip()
            cur = None if src == '/dev/null' else src[2:]  # new files are skipped
        elif line.startswith('+++ '):
            if cur is not None and line[4:].strip() == '/dev/null':
                cur = None  # deleted file
            if cur is not None:
                out[cur] = []
        elif line.startswith('@@') and cur is not None:
            m = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
            hunk = (int(m.group(1)), [])
            out[cur].append(hunk)
        elif hunk is not None and cur is not None and line[:1] in (' ', '-', '+'):
            hunk[1].append(line)
    return out


def apply_hunks(old, hunks):
    a = old.split('\n')
    res, i = [], 0
    for start, lines in hunks:
        s = max(start - 1, 0) if any(l[0] != '+' for l in lines) else start
        res += a[i:s]
        i = s
        for l in lines:
            if l[0] == ' ':
                if a[i] != l[1:]:
                    return None
                res.append(a[i]); i += 1
            elif l[0] == '-':
                if a[i] != l[1:]:
                    return None
                i += 1
            else:
                res.append(l[1:])
    return '\n'.join(res + a[i:])


def edit_snippet(old, new, comment):
    a, b = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = ['\n'.join(b[g[0][3]:g[-1][4]]) for g in sm.get_grouped_opcodes(CONTEXT)]
    marker = f'{comment} ... existing code ...'
    return f'{marker}\n' + f'\n{marker}\n'.join(parts) + f'\n{marker}'


def changed_fraction(old, new):
    a, b = old.splitlines(), new.splitlines()
    same = sum(bl.size for bl in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 1 - same / max(len(a), len(b), 1)


def fetch(repo, commit, path):
    dst = RAW / repo.replace('/', '__') / commit[:12] / path
    if dst.exists():
        return dst.read_text(encoding='utf-8', errors='replace')
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'research-data-prep'})
        text = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
    except Exception:  # noqa: BLE001
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding='utf-8')
    return text


def candidates(ds_id, source):
    df = pd.read_parquet(hf_hub_download(ds_id, 'data/test-00000-of-00001.parquet', repo_type='dataset'))
    jobs = []
    for r in df.itertuples():
        for path, hunks in file_patches(r.patch).items():
            lang = EXT.get(Path(path).suffix)
            if lang and hunks:
                title = r.problem_statement.strip().splitlines()[0][:150] if r.problem_statement.strip() else 'Fix the issue'
                jobs.append((r.repo, r.base_commit, path, hunks, lang, title, r.instance_id, source))
    with ThreadPoolExecutor(8) as ex:
        olds = list(ex.map(lambda j: fetch(j[0], j[1], j[2]), jobs))
    return [(j, o) for j, o in zip(jobs, olds) if o]


def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    n = lambda s: len(tok.encode(s, add_special_tokens=False))
    rng = random.Random(20261122)
    pools = {}
    for ds_id, source in (('princeton-nlp/SWE-bench_Verified', 'swebench-verified'),
                          ('SWE-bench/SWE-bench_Multilingual', 'swebench-multilingual')):
        good, stats = [], Counter()
        for (repo, commit, path, hunks, lang, title, iid, src), old in candidates(ds_id, source):
            new = apply_hunks(old, hunks)
            if new is None:
                stats['patch failed'] += 1; continue
            if '```' in old or '```' in new or old == new:
                stats['skip'] += 1; continue
            k = n(old)
            if not (LO <= k <= HI) or n(new) > HI + 200:
                stats['size'] += 1; continue
            frac = changed_fraction(old, new)
            if frac > 0.4:
                stats['too much changed'] += 1; continue
            good.append(dict(id=f"apply-{iid}-{Path(path).stem}", cohort='apply100', regime='file', source=src, repo=repo,
                             path=path, language=lang, format='apply', instruction=f'Fix for issue: {title}', old=old,
                             edit=edit_snippet(old, new, COMMENT.get(lang, '//')), ref=new, old_tokens=k,
                             changed_fraction=round(frac, 4)))
        print(source, 'kept', len(good), dict(stats))
        pools[source] = good
    picked = []
    by_repo = defaultdict(list)
    for it in pools['swebench-verified']:
        by_repo[it['repo']].append(it)
    py = []
    for v in by_repo.values():
        rng.shuffle(v)
    while len(py) < 50 and any(by_repo.values()):
        for repo in sorted(by_repo):
            if by_repo[repo] and len(py) < 50 and sum(p['repo'] == repo for p in py) < 8:
                py.append(by_repo[repo].pop())
        if all(not v or sum(p['repo'] == r for p in py) >= 8 for r, v in by_repo.items()):
            break
    by_lang = defaultdict(list)
    for it in pools['swebench-multilingual']:
        by_lang[it['language']].append(it)
    for v in by_lang.values():
        rng.shuffle(v)
    ml = []
    while len(ml) < 50 and any(by_lang.values()):
        for lang in sorted(by_lang):
            while by_lang[lang]:
                it = by_lang[lang].pop()
                if sum(p['repo'] == it['repo'] for p in ml) < 3:
                    ml.append(it); break
            if len(ml) >= 50:
                break
    picked = py + ml
    seen, items = set(), []
    for it in picked:  # unique ids (two files with the same stem in one task)
        while it['id'] in seen:
            it['id'] += '_'
        seen.add(it['id']); items.append(it)
    (HERE / 'data' / 'apply100.json').write_text(json.dumps(items))
    ks = sorted(it['old_tokens'] for it in items)
    fr = sorted(it['changed_fraction'] for it in items)
    print(len(items), Counter(it['language'] for it in items), '| repos', len({it['repo'] for it in items}),
          '| file tokens median', ks[len(ks) // 2], 'range', ks[0], ks[-1], '| changed lines median', fr[len(fr) // 2])


if __name__ == '__main__':
    main()
