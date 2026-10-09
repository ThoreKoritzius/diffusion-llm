"""Eval data and SQL scoring shared by every benchmark backend."""

import re
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

TAGS = ["<PROMPT>", "</PROMPT>", "<CONTEXT>", "</CONTEXT>", "<SQL>", "</SQL>"]


def load_eval_set(n: int = 256, no_joins: bool = False) -> List[Dict[str, str]]:
    """First `n` rows of the gretelai test split with a non-empty prompt (same slice as the experiments)."""
    from datasets import load_dataset

    ds = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    ds = ds.filter(lambda ex: ex["sql_prompt"].strip() != "")
    if no_joins:
        ds = ds.filter(lambda ex: "join" not in ex["sql"].lower())
    ds = ds.select(range(min(n, len(ds))))
    return [{"prompt": ex["sql_prompt"], "context": ex.get("sql_context", "") or "", "sql": ex["sql"]} for ex in ds]


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _clean_context(context: str) -> str:
    """Drop schema declarations that SQLite cannot execute directly."""
    context = re.sub(r"\bCREATE\s+SCHEMA\s+[^;]+;", "", context, flags=re.I)
    return re.sub(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", r"\2", context)


def _tables(conn: sqlite3.Connection) -> Iterable[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _db_state(conn: sqlite3.Connection) -> Tuple[Any, ...]:
    out = []
    for table in _tables(conn):
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        order = ", ".join(f'"{c}"' for c in cols) if cols else "rowid"
        out.append((table, tuple(conn.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall())))
    return tuple(out)


def run_sql(context: str, sql: str) -> Tuple[str, Any]:
    """Execute `sql` on a fresh in-memory DB built from the example's CREATE/INSERT context.
    Returns ("rows", result) for queries or ("state", db_state) for DML. Raises on error."""
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


def score(context: str, gold: str, pred: str) -> Dict[str, bool]:
    """exact: normalized string match. valid: prediction executes. exec: same result as gold on this DB.
    gold_ok: gold itself executes in SQLite (rows where it doesn't are dialect-specific)."""
    out = {"exact": normalize_sql(pred) == normalize_sql(gold), "valid": False, "exec": False, "gold_ok": False}
    try:
        gold_res = run_sql(context, gold)
        out["gold_ok"] = True
    except Exception:
        gold_res = None
    if pred.strip():
        try:
            pred_res = run_sql(context, pred)
            out["valid"] = True
            out["exec"] = gold_res is not None and pred_res == gold_res
        except Exception:
            pass
    return out


def compiles(context: str, sql: str) -> bool:
    """Does `sql` compile against the schema in `context`? Uses only the user-provided schema (EXPLAIN plans the
    query without running it), so it is a legitimate inference-time check: it catches syntax errors and unknown
    tables/columns. Contexts SQLite cannot load count as "unknown" -> True (don't penalize dialect gaps)."""
    if not sql.strip():
        return False
    conn = sqlite3.connect(":memory:")
    try:
        try:
            conn.executescript(_clean_context(context))
        except Exception:
            return True
        conn.execute("EXPLAIN " + sql)
        return True
    except Exception:
        return False
    finally:
        conn.close()
