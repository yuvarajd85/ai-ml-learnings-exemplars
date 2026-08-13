---
description: Derive a feature spec from the BRD and write it to specs/. Produces requirement IDs, behaviour, and acceptance criteria. Does NOT write application code.
argument-hint: <feature-id> — e.g. FR-AUTH, FR-TKT, FR-AI
---

Read @docs/BRD_TicketFlow.md in full, then create the spec file for the feature area **$ARGUMENTS**.

## Steps

1. **Identify** every BRD entry that belongs to this feature area: functional requirements (FR-*), non-functional requirements (NFR-*), business rules (BR-*), assumptions/constraints (AC-*), and success criteria (SC-*) that are relevant to $ARGUMENTS.

2. **Determine the spec filename**: list the files currently in `specs/` to find the next available sequence number N (01, 02, …). Write the spec to `specs/<N>-$ARGUMENTS.md`. If `specs/` is empty, start at `01`.

3. **Write the spec file** using exactly this structure:

```
# Spec: $ARGUMENTS

## BRD References
<!-- List every BRD ID that this spec covers, one per line -->
- FR-...-01: <requirement text>
- FR-...-02: <requirement text>
- NFR-0N: <requirement text>
- BR-0N: <rule text>

## Overview
<!-- 2–4 sentences: what this feature does and why it exists, from the user's perspective -->

## Behaviour
<!-- Prose description of the complete feature behaviour, including:
     - Happy path
     - Validation rules and error cases
     - Any state transitions or workflow rules
     - Data constraints (field lengths, allowed values, etc.)
     - Security / isolation rules that apply (e.g. user-scoped data) -->

## Data Model (if applicable)
<!-- List only the tables/columns that this feature owns or writes.
     Cross-reference columns owned by other specs without redefining them. -->

## Acceptance Criteria
<!-- Numbered list. Each criterion must be:
     - Independently testable
     - Stated as an observable outcome (Given/When/Then or plain declarative)
     - Traceable to a BRD requirement ID in brackets -->

AC-01 [FR-...-01]: ...
AC-02 [FR-...-02]: ...

## Out of Scope
<!-- Explicitly name anything from the BRD Out-of-Scope list (Section 3.2) or
     adjacent features that might be confused with this one -->
```

## Constraints

- Do **not** write any Python, SQL, or other application code.
- Do **not** invent requirements that are not in the BRD.
- Every acceptance criterion must trace to at least one BRD ID in brackets.
- If a BRD requirement is ambiguous, note the ambiguity in the criterion rather than silently resolving it.
