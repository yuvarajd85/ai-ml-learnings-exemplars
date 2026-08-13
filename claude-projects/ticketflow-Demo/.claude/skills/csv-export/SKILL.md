---
name: csv-export
description: >
  Exports the current FILTERED ticket list to a UTF-8 CSV file containing only
  the logged-in user's rows (FR-EXP-03: filtered rows only; FR-EXP-04: user-scoped data).
  Invokable standalone via /csv-export <user_id> [--status OPEN] [--priority HIGH] …
  or imported from the Reports page via ticketflow/exports.generate_csv().
---

# Skill: csv-export

Generates a UTF-8 CSV file from the **currently filtered, user-scoped** ticket list.
Two requirements drive every design decision here:

- **FR-EXP-03** — the export contains *exactly* the rows visible under the active
  filters/search, nothing more, nothing less.
- **FR-EXP-04** — the export contains *only* the logged-in user's tickets
  (tickets where they are requester **or** assignee). This is enforced at the
  DB query layer, not post-hoc in Python.

---

## How this skill works

```
filter state (status, priority, category, date range, search text)
        +
session user_id  ──────────────────────────────────────────────────────┐
        │                                                               │
        ▼                                                               │
db_list_tickets(user_id, ...)        ← always WHERE requester_id=%s    │
        │                               OR assignee_id=%s  (FR-EXP-04) │
        │  returns only matching rows in newest-first order (FR-EXP-03) │
        ▼                                                               │
generate_csv(tickets)                                                   │
        │  header row + one data row per ticket                         │
        │  empty result → header row only, not an error                 │
        ▼                                                               │
bytes  ──►  st.download_button  /  write to file  ◄──────────────────--┘
```

---

## Complete self-contained implementation

When invoked standalone (e.g. for testing or seeding) this skill can be run as
the script below. When used from the Reports page, the same functions are imported
from `ticketflow/exports.py`.

```python
"""
Standalone runner for the csv-export skill.
Usage:
    python .claude/skills/csv-export/SKILL.md  # not runnable directly
    — instead run the equivalent in a Python REPL or test:

    from ticketflow.db import db_list_tickets
    from ticketflow.exports import generate_csv
    from ticketflow.session import get_session_user  # or pass user_id directly

Enforces:
    FR-EXP-03: only filtered rows are exported
    FR-EXP-04: only the logged-in user's tickets (enforced in db_list_tickets)
    BR-05:     user sees only tickets where they are requester or assignee
"""
import csv
import io
import sys
from ticketflow.db import db_list_tickets
from ticketflow.exports import generate_csv


def export_filtered_csv(
    user_id: int,
    statuses: list[str] | None = None,      # FR-EXP-03: pass active filter; None = all
    priorities: list[str] | None = None,    # FR-EXP-03: pass active filter; None = all
    categories: list[str] | None = None,    # FR-EXP-03: pass active filter; None = all
    date_from=None,                         # FR-EXP-03: inclusive lower bound
    date_to=None,                           # FR-EXP-03: inclusive upper bound
    search: str | None = None,              # FR-EXP-03: title/description substring
) -> bytes:
    """
    Fetch the user-scoped, filtered ticket list and return UTF-8 CSV bytes.

    FR-EXP-03: only the rows matching all active filters are exported.
    FR-EXP-04: db_list_tickets always includes
                  WHERE (requester_id = %s OR assignee_id = %s)
               bound to user_id — never omitted, never overridable by filter params.
    BR-05:     user_id must come from the authenticated session, never from
               a URL parameter or form field.
    """
    # Step 1 — fetch filtered, user-scoped rows (FR-EXP-03 + FR-EXP-04 enforced here)
    tickets = db_list_tickets(
        user_id=user_id,        # ← always from st.session_state["user"]["id"]
        statuses=statuses,
        priorities=priorities,
        categories=categories,
        date_from=date_from,
        date_to=date_to,
        search=search,
    )

    # Step 2 — generate bytes (no DB access inside; isolation already guaranteed)
    return generate_csv(tickets)


# ── Standalone: write to disk for manual inspection ───────────────────────────
if __name__ == "__main__":
    # Example: export all OPEN tickets for user 1
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    csv_bytes = export_filtered_csv(user_id, statuses=["OPEN"])
    path = "ticketflow_export_standalone.csv"
    with open(path, "wb") as f:
        f.write(csv_bytes)
    print(f"Written {len(csv_bytes)} bytes → {path}")
```

---

## Usage from the Reports page

The Reports page (`pages/4_Reports.py`) uses `generate_csv` from `ticketflow/exports.py`
(the function this skill backs). The critical wiring is:

```python
# pages/4_Reports.py — CSV download button

# 1. user_id from session only — never from URL/form (BR-05, FR-EXP-04)
user_id: int = st.session_state["user"]["id"]

# 2. fetch filtered, user-scoped rows (FR-EXP-03 + FR-EXP-04)
tickets = db_list_tickets(
    user_id=user_id,
    statuses=sel_statuses or None,    # empty multiselect → None → no restriction
    priorities=sel_priorities or None,
    categories=sel_categories or None,
    date_from=date_from or None,
    date_to=date_to or None,
    search=search or None,
)

# 3. generate bytes and offer download (skill: csv-export)
from ticketflow.exports import generate_csv   # backed by this skill
csv_bytes = generate_csv(tickets)

st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"ticketflow_export_{today_str}.csv",
    mime="text/csv",
)
```

---

## CSV format

| Column | Source field | Notes |
|---|---|---|
| `Ticket #` | `ticket_number` | e.g. `TKF-00042` |
| `Title` | `title` | full text, not truncated |
| `Status` | `status` | `OPEN`, `IN_PROGRESS`, `RESOLVED`, `CLOSED` |
| `Priority` | `priority` | `LOW`, `MEDIUM`, `HIGH`, `URGENT` |
| `Category` | `category` | `Bug`, `Feature Request`, `Access`, `Hardware`, `Other` |
| `Created` | `created_at` | `YYYY-MM-DD` |

- Encoding: UTF-8, no BOM.
- Row order: newest first (from `db_list_tickets` `ORDER BY created_at DESC`).
- Empty result: header row only, no data rows — valid CSV, not an error.

---

## FR-EXP-03 / FR-EXP-04 compliance checklist

When adding new code paths that call this skill, verify:

- [ ] `user_id` comes from `st.session_state["user"]["id"]`, not a URL or form field.
- [ ] All active filter values (status, priority, category, date range, search) are
      passed through to `db_list_tickets` — no silent defaults that broaden the scope.
- [ ] `generate_csv` is called with the *already-fetched* ticket list — no second
      DB call inside the export function.
- [ ] The download MIME type is `text/csv` and the filename contains the current date.

---

## BRD traceability

| Requirement | Enforcement point |
|---|---|
| FR-EXP-01 | `st.download_button(mime="text/csv")` in `pages/4_Reports.py` |
| FR-EXP-03 | Filter params passed to `db_list_tickets` before generating CSV |
| FR-EXP-04 | `db_list_tickets` `WHERE (requester_id=%s OR assignee_id=%s)` always present |
| FR-EXP-05 | This skill file at `.claude/skills/csv-export/SKILL.md` |
| BR-05 | `user_id` sourced exclusively from `st.session_state["user"]["id"]` |
| NFR-03 | Email and password_hash never queried or written to the CSV |
