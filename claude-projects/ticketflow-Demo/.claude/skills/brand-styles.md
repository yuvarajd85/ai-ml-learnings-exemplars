---
name: brand-styles
description: Brand colour palette and font constants for TicketFlow. Single source of truth for all visual components — the PDF export skill reads from here rather than hardcoding hex values. Reference these constants in any skill or page that renders branded output.
---

# Brand Styles — TicketFlow

These constants define the visual identity used consistently across all UI pages, PDF reports, and exported files. Any component that renders branded output should import from `ticketflow/exports.py` or reference the hex values below — never hardcode colours elsewhere.

## Colour Palette

| Name | Hex | Primary usage |
|---|---|---|
| Purple | `#8A5CF6` | Headers, sidebar, section titles, OPEN status, document header text |
| Green | `#00E676` | RESOLVED status, primary CTA buttons, user chat bubbles, success states |
| Pink | `#FF2D95` | IN_PROGRESS status, accent badges, URGENT priority, active states |
| Black | `#000000` | CLOSED status, body text, data-cell text |
| White | `#ffffff` | Page background, data cell backgrounds, text on dark badges |
| Light purple | `#F9F7FF` | Alternating table row tint (derived from purple, low saturation) |
| Grid line | `#E0D9FF` | Table grid / separator lines |

## Status badge colours

| Status | Background | Text |
|---|---|---|
| OPEN | `#8A5CF6` (purple) | `#ffffff` (white) |
| IN_PROGRESS | `#FF2D95` (pink) | `#ffffff` (white) |
| RESOLVED | `#00E676` (green) | `#000000` (black) |
| CLOSED | `#000000` (black) | `#ffffff` (white) |

## Priority badge colours

| Priority | Background | Text |
|---|---|---|
| URGENT | `#FF2D95` (pink) | `#ffffff` (white) |
| HIGH | `#000000` (black) | `#ffffff` (white) |
| MEDIUM | `#ffffff` (white) | `#000000` (black) — add border |
| LOW | `#ffffff` (white) | `#000000` (black) — add border |

## Typography

- Font family: Helvetica (PDF) / system sans-serif (Streamlit).
- Document/page titles: 18 pt, bold, purple.
- Section headers: 12–14 pt, bold, purple.
- Table header row: 9 pt, bold, white text on purple background.
- Table data rows: 8 pt, black on white/light-purple alternating.
- Meta / caption text: 9 pt, grey.

## Usage in Python (reportlab)

```python
from reportlab.lib import colors

PURPLE = colors.HexColor("#8A5CF6")
GREEN  = colors.HexColor("#00E676")
PINK   = colors.HexColor("#FF2D95")
```

These are the exact values used in `ticketflow/exports.py`. Any new export or branded document must import from that module rather than re-declaring the constants.
