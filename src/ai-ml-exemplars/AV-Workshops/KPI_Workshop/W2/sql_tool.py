import re
import sqlite3
from typing import List, Dict, Any

DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|ATTACH|DETACH|PRAGMA)\b",
    re.I,
)

# Strip common formatting the LLM might add.
_CODE_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.I)
_LEADING_COMMENTS = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/\s*)*", re.S)
_LABEL_PREFIX = re.compile(r"^\s*(sql\s*query|query)\s*:\s*", re.I)

def _normalize(query: str) -> str:
    q = query.strip()
    q = _CODE_FENCE.sub("", q).strip()
    q = _LEADING_COMMENTS.sub("", q).strip()
    q = _LABEL_PREFIX.sub("", q).strip()
    q = q.rstrip(";").strip()
    return q

def run_sql_readonly(db_path: str, query: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    q = _normalize(query)

    # Disallow multi-statement queries.
    if ";" in q:
        raise ValueError("Multiple SQL statements are not allowed.")

    # Block non-read-only keywords and joins SQLite doesn't support.
    if DISALLOWED.search(q):
        raise ValueError("Only read-only SELECT queries are allowed.")

    if not q.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT/WITH queries are allowed.")

    if re.search(r"\b(full\s+outer\s+join|right\s+join)\b", q, re.I):
        raise ValueError("SQLite does not support FULL OUTER JOIN / RIGHT JOIN. Use UNION + LEFT JOIN workaround.")

    if "limit" not in q.lower():
        q = f"{q} LIMIT {max_rows}"

    # Execute and return rows as dicts.
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        cur.execute(q)
    except sqlite3.Error as e:
        raise RuntimeError(f"SQLite error: {e}\n\nSQL was:\n{q}") from e
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]
