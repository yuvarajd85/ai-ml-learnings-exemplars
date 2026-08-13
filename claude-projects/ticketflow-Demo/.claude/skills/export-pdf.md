---
name: export-pdf
description: Generates a branded PDF report from a filtered, user-scoped ticket list. Callable from the Reports page (pages/4_Reports.py) or standalone via /export-pdf for testing. Backed by ticketflow/exports.generate_pdf(). Reads brand colours from brand-styles skill. FR-EXP-02, FR-EXP-03, FR-EXP-04, FR-EXP-05, SC-06.
---

# Skill: export-pdf

Generates a branded A4 PDF report containing a user's filtered ticket list. Uses `reportlab` for PDF generation and reads all colour values from the [[brand-styles]] skill — no hex values are hardcoded in the PDF logic.

## Backed by

`ticketflow/exports.generate_pdf(tickets: list[dict], filter_summary: str) -> bytes`

- `tickets`: pre-fetched, user-scoped ticket dicts (same shape as `db_list_tickets` output).
- `filter_summary`: human-readable description of the active filters, e.g. `"Status: OPEN; Priority: HIGH"` or `"None (all tickets)"`. Included in the PDF header.

## PDF structure

1. **Document header** — "TicketFlow — Ticket Report" in purple (`#8A5CF6`).
2. **Metadata** — generation timestamp (UTC) and active filter summary in grey.
3. **Ticket table** — columns: Ticket #, Title, Status, Priority, Category, Created.
   - Header row: purple background, white text, bold.
   - Data rows: alternating white / light-purple (`#F9F7FF`), black text.
   - Status cell: colour-coded per brand-styles (RESOLVED = green, IN_PROGRESS = pink, OPEN = purple, CLOSED = black).
4. **Empty state** — if `tickets` is empty, the document contains the header and the message "No tickets match the selected filters." — it is never empty or corrupt.

## Usage from the Reports page

```python
from ticketflow.exports import generate_pdf
from ticketflow.db import db_list_tickets

tickets = db_list_tickets(user_id, statuses=..., ...)
filter_summary = "Status: OPEN; Priority: HIGH"
pdf_bytes = generate_pdf(tickets, filter_summary)

st.download_button(
    label="Download PDF",
    data=pdf_bytes,
    file_name=f"ticketflow_report_{today_str}.pdf",
    mime="application/pdf",
)
```

## Standalone invocation (testing / seeding)

```python
from ticketflow.db import db_list_tickets
from ticketflow.exports import generate_pdf

tickets = db_list_tickets(user_id=1)
pdf_bytes = generate_pdf(tickets, "None (all tickets)")
with open("report.pdf", "wb") as f:
    f.write(pdf_bytes)
```

## Dependencies

- `reportlab>=4.0.0` (in `requirements.txt`)
- [[brand-styles]] — colour constants; referenced in `ticketflow/exports.py`

## Data isolation guarantee

`generate_pdf` outputs whatever ticket list it receives. Data isolation is the responsibility of the caller: always pass the result of `db_list_tickets(user_id, ...)`.

## BRD traceability

- FR-EXP-02: downloadable PDF
- FR-EXP-03: reflects exactly the filtered rows
- FR-EXP-04: data isolation enforced by the query layer
- FR-EXP-05: delivered as a reusable Claude Code skill; brand styling from brand-styles skill
- SC-06: PDF matches the design system palette
