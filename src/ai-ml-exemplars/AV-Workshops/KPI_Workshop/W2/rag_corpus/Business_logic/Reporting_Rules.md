# Reporting Rules

These rules reflect common real-world reporting patterns.

## Order inclusion
- **Completed revenue** uses orders where `status NOT IN ('cancelled','returned')`.
- Cancelled/returned orders are excluded from revenue and AOV in this workshop, unless a KPI explicitly includes them.

## Time windows
- Weekly grouping uses `strftime('%Y-%W', timestamp)` (ISO-ish year-week).
- When reporting weekly insights, always state the time window and timezone assumptions.

## Narrative guidelines (weekly update)
- Separate **facts** (from data) from **hypotheses** (possible drivers).
- Always restate KPI definitions briefly when the KPI is ambiguous (e.g., churn window).
- Call out data caveats: missing attribution for guest purchases, order cancellation handling, etc.

## Guardrails (recommended)
- SQL execution must be **read-only** (SELECT only).
- Apply row limits in exploratory queries (e.g., LIMIT 1000) before removing them.
- Avoid PII in outputs (none exists in this dataset by design).
