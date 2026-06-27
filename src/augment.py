"""Training-time augmentation for text-to-SQL robustness.

The gretelai dataset only contains clean snake_case identifiers and
well-formed questions. Real inputs have spaced/mixed-case column names
(often unquoted in the DDL) and colloquial phrasing, so we synthesize those
variants on the fly.
"""
import random
import re

COL_AUG_PROB = 0.5      # chance an example gets identifier restyling at all
COL_RENAME_PROB = 0.5   # chance per eligible column
PROMPT_LOWER_PROB = 0.3
PROMPT_STRIP_PROB = 0.3

_SKIP_DEF_KEYWORDS = {"primary", "foreign", "unique", "check", "constraint", "key"}
_STYLES = ("title_spaced", "lower_spaced", "camel")


def _split_top_level(s: str):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def extract_columns(context: str):
    cols = []
    for m in re.finditer(r"CREATE\s+TABLE[^(]*\((.*?)\)\s*;", context, re.IGNORECASE | re.DOTALL):
        for part in _split_top_level(m.group(1)):
            tokens = part.strip().split()
            if tokens and re.fullmatch(r"[A-Za-z_]\w*", tokens[0]) and tokens[0].lower() not in _SKIP_DEF_KEYWORDS:
                cols.append(tokens[0])
    return cols


def restyle(name: str, style: str) -> str:
    words = [w for w in name.split("_") if w]
    if style == "title_spaced":
        return " ".join(w.capitalize() for w in words)
    if style == "lower_spaced":
        return " ".join(words)
    if style == "camel":
        return "".join(w.capitalize() for w in words)
    return name


def augment_example(prompt: str, context: str, sql: str, rng=random):
    if rng.random() < COL_AUG_PROB:
        for col in extract_columns(context):
            if "_" not in col or rng.random() > COL_RENAME_PROB:
                continue
            new = restyle(col, rng.choice(_STYLES))
            if new == col:
                continue
            if " " in new:
                sql_repl = f'"{new}"'
                # real-world DDL often leaves spaced names unquoted
                ctx_repl = sql_repl if rng.random() < 0.5 else new
            else:
                sql_repl = ctx_repl = new
            pattern = re.compile(rf"\b{re.escape(col)}\b")
            context = pattern.sub(ctx_repl, context)
            sql = pattern.sub(sql_repl, sql)

    if rng.random() < PROMPT_LOWER_PROB:
        prompt = prompt.lower()
    if rng.random() < PROMPT_STRIP_PROB:
        prompt = prompt.rstrip(" ?")

    return prompt, context, sql
