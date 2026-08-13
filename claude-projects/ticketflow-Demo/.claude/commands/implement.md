---
description: Implement the feature described in the named spec. Implements only what the spec says, nothing more.
argument-hint: <spec-file-or-feature-id> — e.g. FR-AUTH or specs/01-FR-AUTH.md
---

Read @CLAUDE.md for project conventions, then implement the feature described in the spec for **$ARGUMENTS**.

## Steps

1. **Locate the spec file**: if `$ARGUMENTS` is a bare feature ID (e.g. `FR-AUTH`), glob `specs/*$ARGUMENTS*.md` to find the file. If it is already a path, read it directly. If no file is found, stop and tell the user to run `/spec $ARGUMENTS` first.

2. **Read the spec file in full.** Note every acceptance criterion (AC-01, AC-02, …) and every BRD reference. These are the complete definition of done.

3. **Survey the existing codebase** to understand what is already in place (db.py, app.py, relevant pages/, ticketflow/ modules) before writing any new code.

4. **Implement** only the behaviour described in the spec:
   - Write or extend Python files as needed.
   - Follow the architecture in CLAUDE.md (Streamlit pages → ticketflow package → db.py).
   - Every status/field change must be written in a single DB transaction together with its `ticket_activity` row (BR-04, FR-STAT-05).
   - Enforce allowed status transitions in code; never trust the UI (BR-02, BR-03).
   - Scope all DB queries to the logged-in user (BR-05).
   - Store passwords only as bcrypt hashes; never log secrets or PII (NFR-01, NFR-03).
   - If this feature includes the AI chat layer, wrap SQL generation in a LangGraph safety loop that validates the query is a single, read-only, user-scoped SELECT before execution (FR-AI-06, FR-AI-07).

5. **Do not implement** anything not described in the spec — no extra fields, no bonus error handling, no UI flourishes beyond what the spec requires.

6. After writing all files, summarise what was implemented and map each change back to its acceptance criterion (AC-0N → file:line).

## Constraints

- Never skip a spec acceptance criterion, even if it seems redundant.
- Never modify the spec file itself.
- If a requirement in the spec is unclear or contradicts another, stop and ask rather than guessing.
