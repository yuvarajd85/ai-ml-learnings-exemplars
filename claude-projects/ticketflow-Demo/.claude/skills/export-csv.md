---
name: export-csv
description: Generates a UTF-8 CSV file from a filtered, user-scoped ticket list. Callable from the Reports page (pages/4_Reports.py) or standalone via /export-csv for testing. Backed by ticketflow/exports.generate_csv(). FR-EXP-01, FR-EXP-03, FR-EXP-04, FR-EXP-05.
---

# Skill: export-csv

Generates a UTF-8 CSV file containing a user's filtered ticket list.

## Backed by

`ticketflow/exports.generate_csv(tickets: list[dict]) -> bytes`

The function accepts an already-fetched, user-scoped list of ticket dicts and returns raw CSV bytes. It performs no DB access — the caller is responsible for fetching the data with the correct user scope.

## CSV format

- **Header row**: `Ticket #,Title,Status,Priority,Category,Created`
- **One data row per ticket**, newest first (the order from `db_list_tickets`).
- **Created** column: `YYYY-MM-DD` format.
- **Encoding**: UTF-8 (no BOM).
- **Empty result**: header row only — no data rows, not an error.

## Usage from the Reports page

```python
from ticketflow.exports import generate_csv
from ticketflow.db import db_list_tickets

tickets = db_list_tickets(user_id, statuses=..., priorities=..., ...)
csv_bytes = generate_csv(tickets)

st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"ticketflow_export_{today_str}.csv",
    mime="text/csv",
)
```

## Standalone invocation (testing / seeding)

To generate a CSV for a specific user without the Streamlit UI:

```python
from ticketflow.db import db_list_tickets, get_connection
from ticketflow.exports import generate_csv

tickets = db_list_tickets(user_id=1)
csv_bytes = generate_csv(tickets)
with open("export.csv", "wb") as f:
    f.write(csv_bytes)
```

## Data isolation guarantee

`db_list_tickets` always includes `WHERE (requester_id = %s OR assignee_id = %s)`. The `generate_csv` function itself does not enforce isolation — it outputs whatever it receives. The caller must always pass a pre-scoped ticket list.

## BRD traceability

- FR-EXP-01: downloadable CSV
- FR-EXP-03: reflects exactly the filtered rows
- FR-EXP-04: data isolation enforced by the query layer
- FR-EXP-05: delivered as a reusable Claude Code skill
