'''
Created on 1/17/26 at 10:27 AM
By yuvarajdurairaj
Module Name sql_tool
'''
import re
import sqlite3
from typing import List, Dict, Any

DISALLOWED = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|PRAGMA|ATTACH|DETACH)\b")

def _normalize(query:str) -> str:
    q = query.strip()
    q = query.rstrip(";").strip()
    return q

def run_sql(query:str, db_path:str, max_rows:int=100) -> List[Dict[str,Any]]:
    q = _normalize(query)

    #Disallowing Query
    if DISALLOWED.search(q):
        raise ValueError("Only read-only queries are allowed")

    if not q.lower().startswith(("select", "with")):
        raise ValueError("Only read-only queries are allowed - Query has to start with 'Select' or CTE 'with' verb")
    if re.search(r"\b(full\s+outer\s+join|right\s+join)\b",q, re.I):
        raise ValueError("SQLite does not support Full Outer Join / Right Join. User Union + LEFT Join as a workaround")

    if "limit" not in q.lower():
        q = f"{q} LIMIT {max_rows}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(q)
    except sqlite3.Error as e:
        raise RuntimeError(f"SQLite error: {e}\n\nSQL was:\n{q}") from e
    rows = cursor.fetchall()
    conn.close()

def main():
    pass


if __name__ == '__main__':
    main()
