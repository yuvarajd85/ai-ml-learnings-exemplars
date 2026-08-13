---
name: brand-styling
description: >
  Applies the TicketFlow brand palette (purple #8A5CF6, green #00E676, pink #FF2D95,
  white background, black text) to both the Streamlit UI and the PDF reports.
  Emits assets/styles.css (the CSS source of truth) and documents the reportlab
  colour equivalents used in ticketflow/exports.py. SC-06, NFR-08.
---

# Skill: brand-styling

Single source of truth for TicketFlow's visual identity. The same palette is
applied in two rendering environments:

| Environment | Mechanism | File |
|---|---|---|
| Streamlit UI | CSS injected via `st.markdown("<style>…</style>")` | `assets/styles.css` |
| PDF reports | `reportlab.lib.colors.HexColor(…)` constants | `ticketflow/exports.py` |
| Streamlit theme | `[theme]` in `.streamlit/config.toml` | `.streamlit/config.toml` |

---

## Palette

| Token | Hex | Usage |
|---|---|---|
| `--tf-purple` | `#8A5CF6` | Sidebar background, page headers (`h1`), section titles, OPEN status badge, PDF document title, Streamlit primary widgets |
| `--tf-green`  | `#00E676` | RESOLVED status badge, primary CTA buttons, form submit buttons, download buttons, user chat bubbles |
| `--tf-pink`   | `#FF2D95` | IN_PROGRESS status badge, URGENT priority badge, accent / active states |
| `--tf-black`  | `#000000` | CLOSED status badge, all body text, data cell text, HIGH priority badge |
| `--tf-white`  | `#ffffff` | Page background, card surfaces, sidebar text, text on dark badges |
| `--tf-light`  | `#F9F7FF` | Alternating table row tint (low-saturation purple derivative) |
| `--tf-grid`   | `#E0D9FF` | Table grid lines, subtle borders, MEDIUM/LOW badge borders |

---

## Emitted asset: `assets/styles.css`

This file is the CSS source of truth. It is **not** bundled by Streamlit automatically —
it must be injected at runtime by calling `ticketflow.session.apply_brand_css()`.

### How it is structured

```
assets/styles.css
 ├── :root { --tf-* custom properties }   ← palette in one place
 ├── .stApp / [data-testid="stHeader"]    ← white page background
 ├── [data-testid="stSidebar"]            ← purple sidebar, white text + nav links
 ├── .stFormSubmitButton, .stDownloadButton  ← green CTA buttons
 ├── .stButton > button                   ← purple-outline action buttons
 ├── [data-testid="stTabs"]              ← purple active tab indicator
 ├── inputs / multiselect tags            ← purple focus ring + purple tags
 ├── hr / alerts                          ← brand-coloured borders
 ├── .tf-badge / .tf-badge-{status}       ← reusable badge utility classes
 ├── .tf-chat-bubble.user / .assistant    ← chat message bubbles (FR-AI-01)
 └── .tf-card / .tf-card-value / .tf-card-label  ← dashboard summary cards
```

### Injecting in Python

```python
# ticketflow/session.py — already wired; no action needed on new pages
from ticketflow.session import apply_brand_css

apply_brand_css()   # reads assets/styles.css and injects <style> block
```

`apply_brand_css()` is called automatically by `render_sidebar_logout()`, so
every authenticated page (Dashboard, Create Ticket, Ticket List, Reports, Chat)
gets the stylesheet without extra imports. The login page (`app.py`) calls it
directly since it doesn't use `render_sidebar_logout()`.

### Utility badge classes

Instead of embedding inline `style=` attributes, prefer the CSS utility classes:

```python
# Old (inline styles — avoid adding new ones):
st.markdown("<span style='background:#8A5CF6;color:#fff;…'>OPEN</span>",
            unsafe_allow_html=True)

# Preferred (utility class from assets/styles.css):
st.markdown('<span class="tf-badge tf-badge-open">OPEN</span>',
            unsafe_allow_html=True)
```

| Status | Class |
|---|---|
| OPEN | `tf-badge tf-badge-open` |
| IN_PROGRESS | `tf-badge tf-badge-in-progress` |
| RESOLVED | `tf-badge tf-badge-resolved` |
| CLOSED | `tf-badge tf-badge-closed` |

| Priority | Class |
|---|---|
| URGENT | `tf-badge tf-badge-urgent` |
| HIGH | `tf-badge tf-badge-high` |
| MEDIUM | `tf-badge tf-badge-medium` |
| LOW | `tf-badge tf-badge-low` |

### Chat bubble classes

```python
# pages/5_Chat.py
user_html = """
<div class="tf-chat-row user">
  <div class="tf-chat-bubble user">Hello, what's still open?</div>
</div>"""

assistant_html = """
<div class="tf-chat-row assistant">
  <div class="tf-chat-bubble assistant">You have 3 open tickets…</div>
</div>"""

st.markdown(user_html + assistant_html, unsafe_allow_html=True)
```

---

## Streamlit theme: `.streamlit/config.toml`

Streamlit applies `primaryColor` to interactive widgets (sliders, checkboxes,
radio buttons, active tab underline, spinner) and `backgroundColor`/`textColor`
as global defaults. Custom CSS in `assets/styles.css` layers on top.

```toml
[theme]
primaryColor             = "#8A5CF6"   # purple
backgroundColor          = "#ffffff"   # white
secondaryBackgroundColor = "#F9F7FF"   # light purple tint (expanders, sidebar fallback)
textColor                = "#000000"   # black
```

---

## PDF reports: reportlab colour constants

`ticketflow/exports.py` defines these module-level constants. Any new PDF export
function must import from there rather than declaring new `HexColor(…)` calls.

```python
# ticketflow/exports.py  — do not redeclare elsewhere
from reportlab.lib import colors

_PURPLE = colors.HexColor("#8A5CF6")   # document header, table header row, OPEN cell
_GREEN  = colors.HexColor("#00E676")   # RESOLVED status cell
_PINK   = colors.HexColor("#FF2D95")   # IN_PROGRESS status cell
_BLACK  = colors.black                  # CLOSED status cell, body text
_WHITE  = colors.white                  # page background, header text
_LIGHT  = colors.HexColor("#F9F7FF")   # alternating row tint
_GRID   = colors.HexColor("#E0D9FF")   # table grid lines
```

PDF document structure (colour annotations):

```
┌──────────────────────────────────────┐
│ TicketFlow — Ticket Report  [PURPLE] │  ← title_style.textColor = _PURPLE
│ Generated: 2026-06-24 09:41 UTC [grey]│
│ Filters: Status: OPEN        [grey]  │
│                                      │
│ ┌────────┬────────────┬────────┬──…┐ │
│ │Ticket #│ Title      │ Status │…  │ │  ← BACKGROUND _PURPLE, TEXTCOLOR _WHITE
│ ├────────┼────────────┼────────┼──…┤ │
│ │TKF-001 │ Login bug  │  OPEN  │…  │ │  ← status cell: _PURPLE bg, _WHITE fg
│ │TKF-002 │ Slow dash  │RESOLVED│…  │ │  ← status cell: _GREEN bg, _BLACK fg
│ │TKF-003 │ New feature│IN PROG │…  │ │  ← status cell: _PINK bg, _WHITE fg
│ └────────┴────────────┴────────┴──…┘ │
└──────────────────────────────────────┘
```

---

## Consistency checklist

When adding a new page or component, verify:

- [ ] Page calls `render_sidebar_logout()` (authenticated) **or** `apply_brand_css()` (login page).
- [ ] Page `h1` headers use `style='color:#8A5CF6'` or `color: var(--tf-purple)`.
- [ ] Status values use `tf-badge-{status}` classes or the equivalent inline palette colours.
- [ ] Priority values use `tf-badge-{priority}` classes or the equivalent inline palette colours.
- [ ] No new `reportlab.lib.colors.HexColor(…)` call introduces a colour outside the palette.
- [ ] No raw hex string other than those in `assets/styles.css` `:root` block or `ticketflow/exports.py` constants.

---

## BRD traceability

| Requirement | Enforcement point |
|---|---|
| SC-06 | `assets/styles.css` injected on every page via `apply_brand_css()` |
| NFR-08 | `.streamlit/config.toml` theme + CSS stylesheet + PDF colour constants |
| NFR-04 | Consistent UI reduces cognitive load; clear colour coding of status/priority |
