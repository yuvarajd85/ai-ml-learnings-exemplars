# Requirement: Insurance Claim Intake & Adjuster Workspace

## Context

A mid-size auto insurance company is replacing their email-and-spreadsheet claim intake process. Today, customers email PDFs to a shared inbox and adjusters manually triage by reading attachments and pasting fields into an internal spreadsheet. Average claim takes 3–5 days to assign to an adjuster. Leadership wants this down to under an hour and wants visibility into the pipeline.

## Functional requirements

1. **Customer claim submission.** Customers submit a claim via web or mobile app. A submission consists of a structured form (policy number, incident date, location, narrative description) plus uploaded documents — typically 1–10 files comprising photos of damage, scanned police reports, repair shop estimates, and medical bills if injury is involved.

2. **Document ingestion & extraction.** When a claim is submitted, the system extracts key fields from the uploaded documents: claim type (collision, comprehensive, liability, injury), implicated policy number, incident date, claimed amount per line item, vehicle VIN, and (where present) injury details.

3. **Triage & routing.** Each claim is classified by complexity (simple / moderate / complex / suspected-fraud) and routed to the appropriate adjuster pool. Suspected-fraud cases route to a separate investigations queue.

4. **Adjuster workspace.** Adjusters need to search claims by policy number, claimant name, date range, and free-text in the narrative or extracted documents. They view the extracted structured data side-by-side with the original document. They update claim status (received → in-review → approved → denied → paid, plus rejected branches).

5. **Customer status updates.** Every status change generates a notification to the customer (email + push to the mobile app).

6. **Management dashboards.** Operations leadership sees claim volume by type, average time-in-stage, adjuster workload distribution, and payout trends, refreshed at least daily. Finance needs monthly aggregate reports.

7. **Records retention.** Every document, every extracted field, every status transition, and every adjuster decision is retained for at least 7 years per state insurance commissioner requirements. State regulators can subpoena records on a per-claim basis with 30-day delivery SLA.

## Out of scope (do not design for)

- Underwriting / policy issuance — handled by a separate system.
- Payment disbursement — a downstream payment platform handles this; assume it exposes an API.
- Direct integration with body shops or medical providers.
