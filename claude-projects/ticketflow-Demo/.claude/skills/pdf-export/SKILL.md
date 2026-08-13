---
name: pdf-export
description: >
  Renders the current FILTERED ticket list to a branded A4 PDF report using
  reportlab. Enforces FR-EXP-03 (only filtered rows) and FR-EXP-04 (only the
  logged-in user's tickets). Brand colours sourced from the brand-styles skill.
  Invokable standalone or from the Reports page via ticketflow/exports.generate_pdf().
---

# Skill: pdf-export

Generates a branded A4 PDF report from the **currently filtered, user-scoped**
ticket list using the **reportlab** library (`reportlab>=4.0.0`).

Two requirements govern every design decision:

- **FR-EXP-03** — the report contains *exactly* the rows visible under the active
  filters/search. The filter state is passed to `db_list_tickets` before PDF
  generation begins; `generate_pdf` itself never touches the database.
- **FR-EXP-04** — the report contains *only* the logged-in user's tickets.
  `db_list_tickets` always includes `WHERE (requester_id=%s OR assignee_id=%s)`;
  this cannot be bypassed by any filter parameter.

Brand colours come from [[brand-styles]] (`#8A5CF6` purple, `#00E676` green,
`#FF2D95` pink) — no hex values are hardcoded in this skill.

---

## Data-flow

```
filter state  ──┐
                ▼
session user_id → db_list_tickets(user_id, statuses, priorities, ...)
                        │
                        │  WHERE (requester_id=%s OR assignee_id=%s)   ← FR-EXP-04
                        │  AND status IN (...)  AND ...                 ← FR-EXP-03
                        │  ORDER BY created_at DESC
                        ▼
                   tickets: list[dict]   (already scoped + filtered)
                        │
                        ▼
              generate_pdf(tickets, filter_summary)
                        │
                        │  reportlab SimpleDocTemplate on BytesIO
                        │  ├─ header paragraph  (purple, 18 pt)
                        │  ├─ metadata          (timestamp, filter summary)
                        │  └─ ticket Table      (header row purple; status cells
                        │                        colour-coded; alternating row tint)
                        ▼
                   bytes  →  st.download_button  /  write to file
```

---

## Complete reportlab implementation

The backing Python function lives in `ticketflow/exports.py`. The full
implementation is reproduced here so the skill is self-contained and can be
read, tested, or adapted without opening the source file.

```python
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand colours (brand-styles skill — never hardcode hex elsewhere) ──────────
PURPLE = colors.HexColor("#8A5CF6")  # headers, OPEN status, document title
GREEN  = colors.HexColor("#00E676")  # RESOLVED status
PINK   = colors.HexColor("#FF2D95")  # IN_PROGRESS status
BLACK  = colors.black                # CLOSED status, body text
WHITE  = colors.white
LIGHT  = colors.HexColor("#F9F7FF")  # alternating row tint
GRID   = colors.HexColor("#E0D9FF")  # table grid lines

STATUS_BG = {"OPEN": PURPLE, "IN_PROGRESS": PINK, "RESOLVED": GREEN, "CLOSED": BLACK}
STATUS_FG = {"OPEN": WHITE,  "IN_PROGRESS": WHITE, "RESOLVED": BLACK, "CLOSED": WHITE}

# Column widths (A4 usable width ≈ 180 mm with 15 mm margins each side)
COL_WIDTHS = [28*mm, 62*mm, 32*mm, 22*mm, 28*mm, 26*mm]
# Columns:    Ticket#  Title   Status  Priority  Category  Created


def generate_pdf(tickets: list[dict], filter_summary: str) -> bytes:
    """
    Render tickets to an A4 PDF and return the raw bytes.

    tickets       — pre-fetched, user-scoped list from db_list_tickets().
                    No DB access occurs inside this function.
    filter_summary — human-readable description of active filters for the
                    document header, e.g. "Status: OPEN; Priority: HIGH"
                    or "None (all tickets)".

    FR-EXP-03: caller must pass only the rows matching active filters.
    FR-EXP-04: caller must pass only rows scoped to the session user_id.
    """
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    base = getSampleStyleSheet()

    # ── Styles ─────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Heading1"],
        textColor=PURPLE,   # brand purple for document header
        fontSize=18,
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=base["Normal"],
        textColor=colors.grey,
        fontSize=9,
        spaceAfter=2,
    )

    # ── Story ──────────────────────────────────────────────────────────────────
    story = []

    story.append(Paragraph("TicketFlow — Ticket Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        meta_style,
    ))
    story.append(Paragraph(f"Filters: {filter_summary}", meta_style))
    story.append(Spacer(1, 6 * mm))

    # ── Empty state ────────────────────────────────────────────────────────────
    if not tickets:
        # Produce a valid, human-readable document rather than an empty file.
        story.append(Paragraph(
            "No tickets match the selected filters.",
            base["Normal"],
        ))
        doc.build(story)
        return buf.getvalue()

    # ── Table data ─────────────────────────────────────────────────────────────
    rows = [["Ticket #", "Title", "Status", "Priority", "Category", "Created"]]
    for t in tickets:
        created = t["created_at"]
        date_str = (
            created.strftime("%Y-%m-%d")
            if hasattr(created, "strftime")
            else str(created)[:10]
        )
        title = t["title"]
        if len(title) > 45:
            title = title[:44] + "…"  # ellipsis

        rows.append([
            t["ticket_number"],
            title,
            t["status"].replace("_", " "),  # "IN_PROGRESS" → "IN PROGRESS"
            t["priority"],
            t["category"],
            date_str,
        ])

    table = Table(rows, colWidths=COL_WIDTHS, repeatRows=1)

    # ── Base table style ───────────────────────────────────────────────────────
    ts = TableStyle([
        # Header row
        ("BACKGROUND",     (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0), 9),
        # Data rows
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",      (0, 1), (-1, -1), BLACK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        # Grid
        ("GRID",           (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ])

    # ── Per-row status colour (column 2) ───────────────────────────────────────
    # Applied after the base style so it overrides ROWBACKGROUNDS for column 2.
    for i, t in enumerate(tickets, start=1):
        status = t["status"]
        ts.add("BACKGROUND", (2, i), (2, i), STATUS_BG.get(status, PURPLE))
        ts.add("TEXTCOLOR",  (2, i), (2, i), STATUS_FG.get(status, WHITE))

    table.setStyle(ts)
    story.append(table)
    doc.build(story)
    return buf.getvalue()
```

---

## Reportlab component reference

| Component | Purpose in this skill |
|---|---|
| `SimpleDocTemplate` | Wraps a `BytesIO` buffer; builds the PDF from a `story` list |
| `Paragraph` | Renders styled text (title, metadata) |
| `ParagraphStyle` | Defines `textColor`, `fontSize`, `spaceAfter` for reusable text styles |
| `Spacer` | Inserts vertical whitespace between story elements |
| `Table` | Renders the ticket grid with fixed column widths and `repeatRows=1` (header on every page) |
| `TableStyle` | Applies background, text colour, font, grid, padding to cells and ranges |
| `colors.HexColor` | Converts brand hex strings to reportlab `Color` objects |
| `mm` | Unit constant — all measurements in millimetres for A4 sizing |

---

## Usage from the Reports page

```python
# pages/4_Reports.py — PDF download button
# pdf-export skill: .claude/skills/pdf-export/SKILL.md

from ticketflow.db import db_list_tickets
from ticketflow.exports import generate_pdf  # backed by this skill

# 1. user_id from session only (FR-EXP-04, BR-05)
user_id: int = st.session_state["user"]["id"]

# 2. fetch filtered, user-scoped rows (FR-EXP-03 + FR-EXP-04)
tickets = db_list_tickets(
    user_id=user_id,
    statuses=sel_statuses or None,
    priorities=sel_priorities or None,
    categories=sel_categories or None,
    date_from=date_from or None,
    date_to=date_to or None,
    search=search or None,
)

# 3. build human-readable filter description for the PDF header
filter_summary = "; ".join([
    f"Status: {', '.join(sel_statuses)}"   if sel_statuses   else "",
    f"Priority: {', '.join(sel_priorities)}" if sel_priorities else "",
    # ... etc.
]) or "None (all tickets)"

# 4. render to PDF and offer download (skill: pdf-export)
pdf_bytes = generate_pdf(tickets, filter_summary)

st.download_button(
    label="Download PDF",
    data=pdf_bytes,
    file_name=f"ticketflow_report_{today_str}.pdf",
    mime="application/pdf",
)
```

---

## Standalone invocation (testing)

```python
from ticketflow.db import db_list_tickets
from ticketflow.exports import generate_pdf

# Export all RESOLVED tickets for user 1
tickets = db_list_tickets(user_id=1, statuses=["RESOLVED"])
pdf_bytes = generate_pdf(tickets, "Status: RESOLVED")

with open("report_standalone.pdf", "wb") as f:
    f.write(pdf_bytes)

print(f"Written {len(pdf_bytes)} bytes")
assert pdf_bytes[:4] == b"%PDF", "output is not a valid PDF"
```

---

## FR-EXP-03 / FR-EXP-04 compliance checklist

When adding new code paths that call this skill, verify:

- [ ] `user_id` comes from `st.session_state["user"]["id"]`, not a URL or form field.
- [ ] All active filter values are forwarded to `db_list_tickets` — no silent defaults that widen scope.
- [ ] `generate_pdf` is called with the *already-fetched* list — it performs no DB access.
- [ ] `filter_summary` accurately reflects the active filters shown to the user.
- [ ] The download MIME type is `application/pdf` and the filename contains the current date.

---

## Document structure

```
┌─────────────────────────────────────────────┐
│  TicketFlow — Ticket Report          [purple]│
│  Generated: 2026-06-24 09:41 UTC     [grey] │
│  Filters: Status: OPEN               [grey] │
│                                             │
│ ┌────────┬──────────────┬────────┬─ … ─────┐│
│ │Ticket #│ Title        │ Status │ …       ││  ← purple bg, white text, bold
│ ├────────┼──────────────┼────────┼─ … ─────┤│
│ │TKF-001 │ Login broken │ OPEN   │ …       ││  ← white row
│ ├────────┼──────────────┼────────┼─ … ─────┤│
│ │TKF-002 │ Slow dash    │RESOLVED│ …       ││  ← light-purple row; status=green
│ └────────┴──────────────┴────────┴─ … ─────┘│
└─────────────────────────────────────────────┘
  Page 1 of N  (repeats header on subsequent pages via repeatRows=1)
```

---

## BRD traceability

| Requirement | Enforcement point |
|---|---|
| FR-EXP-02 | `st.download_button(mime="application/pdf")` in `pages/4_Reports.py` |
| FR-EXP-03 | Filter params forwarded to `db_list_tickets` before `generate_pdf` is called |
| FR-EXP-04 | `db_list_tickets` `WHERE (requester_id=%s OR assignee_id=%s)` always present |
| FR-EXP-05 | This skill at `.claude/skills/pdf-export/SKILL.md`; brand colours from [[brand-styles]] |
| BR-05 | `user_id` sourced exclusively from `st.session_state["user"]["id"]` |
| NFR-03 | Email and password_hash never queried or included in the PDF |
| SC-06 | Document header purple; status cells colour-coded; white background, black body text |
