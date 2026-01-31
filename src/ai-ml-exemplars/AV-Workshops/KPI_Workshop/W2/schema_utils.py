import sqlite3

def get_sqlite_schema(db_path: str) -> dict:
    # Introspect tables and columns for prompt grounding.
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    schema = {}
    for (t,) in tables:
        cols = cur.execute(f"PRAGMA table_info({t});").fetchall()
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        schema[t] = [c[1] for c in cols]
    con.close()
    return schema

def format_schema_for_prompt(schema: dict) -> str:
    # Render schema as one table-per-line for LLM prompts.
    lines = []
    for t, cols in schema.items():
        lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)
