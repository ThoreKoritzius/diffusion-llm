"""Lightweight SQL benchmark helpers for training-time generation eval."""

import re
import sqlite3
from typing import Any, Iterable, Tuple


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _clean_context(context: str) -> str:
    """Drop schema declarations that SQLite cannot execute directly."""
    context = re.sub(r"\bCREATE\s+SCHEMA\s+[^;]+;", "", context, flags=re.I)
    context = re.sub(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", r"\2", context)
    return context


def _ordered_tables(conn: sqlite3.Connection) -> Iterable[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _db_state(conn: sqlite3.Connection) -> Tuple[Any, ...]:
    out = []
    for table in _ordered_tables(conn):
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        order = ", ".join(f'"{c}"' for c in cols) if cols else "rowid"
        rows = conn.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall()
        out.append((table, tuple(rows)))
    return tuple(out)


def _run(context: str, sql: str) -> Tuple[str, Any]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_clean_context(context or ""))
        before = _db_state(conn)
        cur = conn.execute(sql)
        if cur.description is not None:
            rows = cur.fetchall()
            return "rows", tuple(rows)
        conn.commit()
        after = _db_state(conn)
        return "state", after if after != before else before
    finally:
        conn.close()


def execution_match(context: str, gold_sql: str, pred_sql: str) -> Tuple[bool, bool]:
    """Return (valid_prediction, execution_equivalent_on_given_context).

    This is not a full semantic proof. It is a deterministic "true on this DB"
    metric that catches many exact-match false negatives without calling a judge.
    Unsupported dialect features simply count as not executable for this metric.
    """
    if not pred_sql or not pred_sql.strip():
        return False, False
    try:
        gold_kind, gold = _run(context, gold_sql)
        pred_kind, pred = _run(context, pred_sql)
    except Exception:
        return False, False
    return True, gold_kind == pred_kind and gold == pred
