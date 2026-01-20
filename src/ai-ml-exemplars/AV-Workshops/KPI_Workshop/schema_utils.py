'''
Created on 1/17/26 at 10:28 AM
By yuvarajdurairaj
Module Name schema_utils
'''
import sqlite3
from typing import Dict


def get_sqlite_schema(dp_path:str) -> Dict:
    conn = sqlite3.connect(dp_path)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name from sqlite_master where type='table' and name NOT Like 'sqlite_%'").fetchall()

    schema = {}

    for (t,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({t});").fetchall()
        schema[t] = [c[1] for c in cols]

    conn.close()
    return schema

def format_schema_for_prompts(schema:dict) -> str:
    lines = []
    for t, cols in schema.items():
        lines.append(f"{t}:({','.join(cols)})")

    return "\n".join(lines)


