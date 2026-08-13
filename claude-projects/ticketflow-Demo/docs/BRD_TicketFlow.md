# BRD — TicketFlow (AI-Assisted Ticket Management System)

> Markdown companion to `BRD_TicketFlow.docx`. This is the version Claude Code reads
> when generating specs. Keep both in sync; the `.docx` is the human-facing deliverable.

| Field | Value |
|---|---|
| Document Title | BRD — TicketFlow (AI-Assisted Ticket Management System) |
| Version | 1.0 |
| Status | Draft — pending review and sign-off |
| Owner | Rahul (Training Lead) |
| Purpose | Source-of-truth requirements feeding spec-driven development in Claude Code |
| Build Type | Teaching / demonstration application (no production data, no compliance scope) |

**How this document is used:** Every functional requirement carries an ID (e.g. `FR-TKT-01`). Each Claude Code spec references these IDs, and each test case references a spec's acceptance criteria — giving the traceability chain: **BRD requirement → spec acceptance criterion → test case.**

---

## 1. Executive Summary

TicketFlow is a web-based ticket management system built to teach end-to-end spec-driven development with Claude Code on a realistic, well-bounded domain. After logging in, a user lands on a dashboard showing their open ticket counts, recent tickets, and quick actions. They can raise a ticket, update its status through a controlled workflow, search and filter the ticket list, and download reports — while every change is captured in an immutable activity history.

The system also includes a conversational AI assistant: users can ask plain-English questions about their tickets and receive accurate, data-grounded answers. The assistant uses guarded text-to-SQL with conversation memory, wrapped in a LangGraph safety loop that validates every generated query as a single, read-only, user-scoped SELECT before it ever runs.

The application is built to look polished and deployment-ready, using the brand design system in Section 12. It favours clarity and correct patterns (specs, hooks, agents, skills) over real-world helpdesk completeness.

## 2. Business Objectives

| ID | Objective |
|---|---|
| BO-01 | Demonstrate end-to-end spec-driven development using Claude Code on a realistic, well-bounded domain. |
| BO-02 | Teach correct application fundamentals: validated state transitions, auditability, and data integrity. |
| BO-03 | Teach a practical pattern for embedding a conversational LLM feature (chat with memory + guarded text-to-SQL) over the user's own data. |
| BO-04 | Showcase the full Claude Code toolset — CLAUDE.md, specs, slash commands, hooks, subagents, skills, plugins — each introduced when a feature needs it. |
| BO-05 | Produce a polished, presentable, deployment-ready teaching asset that learners can run locally and demo. |

## 3. Scope

### 3.1 In Scope
- User authentication (register, login, logout, session).
- Dashboard / home: open/in-progress/resolved counts, my recent tickets, quick actions.
- Create a ticket with title, description, category, and priority.
- Ticket detail with a controlled status workflow (Open → In Progress → Resolved → Closed) and an immutable activity history.
- Ticket list with search and filters (status, priority, category, date range).
- Reports / export of the ticket list as CSV and PDF (delivered as reusable skills).
- GenAI conversational chat assistant with memory over the logged-in user's tickets.
- Demo data seeding for realistic analysis and testing.
- Polished Streamlit frontend using the brand design system; deployment-ready.

### 3.2 Out of Scope
- Real email/SMS notifications, SLA timers, escalation engines, and CSAT surveys.
- Multi-tenant organisations, teams, queues, and round-robin auto-assignment.
- File attachments, rich-text editors, and threaded public/private comments.
- SSO/LDAP, granular RBAC beyond the roles in Section 10, and production security hardening.
- Mobile-native app and real customer PII handling (this is a demo build).

## 4. Stakeholders & Personas

| Stakeholder | Interest |
|---|---|
| Course learners | Follow the build; understand each Claude Code feature in context. |
| End user (persona "Aarav") | Raises tickets, tracks their status, and chats with the assistant about his open items. |
| Support agent (persona "Neha") | Works assigned tickets, moves them through the workflow, and reports on the queue. |
| Training Lead (Rahul) | Owns the asset; uses it for training and as a deployable demo. |

**Primary persona — Aarav, 31, non-technical employee:** raises support tickets when something breaks, wants a clean view of what is open and what changed, and likes to casually ask "what's still pending on me?" in a chat that remembers the conversation.

## 5. Assumptions & Constraints

| ID | Assumption / Constraint |
|---|---|
| AC-01 | MySQL is installed locally and reachable; the app connects via PyMySQL using a connection string. |
| AC-02 | A single MySQL user (root) is used for all database access in this demo. (Production would use a least-privilege, read-only user for the AI layer.) |
| AC-03 | Ticket status values are constrained to a fixed set; transitions follow the workflow in Section 9. |
| AC-04 | All timestamps stored in UTC; the UI renders local time. |
| AC-05 | OpenAI API key supplied via environment variable; never hardcoded or logged. |
| AC-06 | This is a demo: no real customer PII and no production support obligations. |

## 6. Functional Requirements

### 6.1 Authentication (FR-AUTH)
| ID | Requirement |
|---|---|
| FR-AUTH-01 | A visitor can register with a unique username, email, and password. |
| FR-AUTH-02 | Passwords are stored only as bcrypt hashes; plaintext is never persisted or logged. |
| FR-AUTH-03 | A registered user can log in; invalid credentials return a generic failure (no user enumeration). |
| FR-AUTH-04 | A logged-in session persists across pages until logout or expiry. |
| FR-AUTH-05 | A user can log out, clearing the session. |
| FR-AUTH-06 | All ticket, reporting, and chat operations require an authenticated session. |

### 6.2 Dashboard / Home (FR-DASH)
| ID | Requirement |
|---|---|
| FR-DASH-01 | Immediately after login, the user sees a summary dashboard. |
| FR-DASH-02 | The dashboard displays counts of tickets by status (Open, In Progress, Resolved, Closed) scoped to the user. |
| FR-DASH-03 | The dashboard shows the user's most recent tickets (e.g. last 5) with title, status, priority, and date. |
| FR-DASH-04 | The dashboard offers quick actions / navigation to Create Ticket, Ticket List, Reports, and Chat Assistant. |
| FR-DASH-05 | All figures reflect live data and update after any ticket create or status change. |

### 6.3 Create Ticket (FR-TKT)
| ID | Requirement |
|---|---|
| FR-TKT-01 | An authenticated user can create a ticket with a title, description, category, and priority. |
| FR-TKT-02 | Title and description are required; empty submissions are rejected with a clear message. |
| FR-TKT-03 | Priority is one of LOW, MEDIUM, HIGH, URGENT; category is from a fixed list (Bug, Feature Request, Access, Hardware, How-to / Other). |
| FR-TKT-04 | On creation, the ticket gets a unique ticket number, status OPEN, and a creation entry in the activity history. |
| FR-TKT-05 | The creating user is recorded as the requester; tickets are scoped to their creator (and assignee). |

### 6.4 Status Workflow & Activity History (FR-STAT)
| ID | Requirement |
|---|---|
| FR-STAT-01 | A user can update a ticket's status only along an allowed transition (see Section 9). |
| FR-STAT-02 | An illegal transition (e.g. CLOSED → OPEN directly) is rejected with a clear message and never persisted. |
| FR-STAT-03 | A user can update priority, category, and assignee on a ticket they own or are assigned. |
| FR-STAT-04 | Every create, status change, and field update writes an immutable row to the activity history (who, what, old → new, when). |
| FR-STAT-05 | The status change and its history row are written in a single DB transaction; partial updates are impossible. |
| FR-STAT-06 | The ticket detail view renders the full activity history, newest first. |

### 6.5 Ticket List, Search & Filters (FR-LIST)
| ID | Requirement |
|---|---|
| FR-LIST-01 | A user can view a list of their tickets, newest first, with ticket number, title, status, priority, category, and date. |
| FR-LIST-02 | The list can be filtered by status, priority, category, and a date range, in any combination. |
| FR-LIST-03 | The user can search tickets by free text across title and description. |
| FR-LIST-04 | Filters and search compose (an active search further filters the current filter set). |
| FR-LIST-05 | The list shows only the logged-in user's tickets; cross-user access is impossible. |

### 6.6 Reports & Export (FR-EXP)
| ID | Requirement |
|---|---|
| FR-EXP-01 | A user can download the current (filtered) ticket list as a CSV file. |
| FR-EXP-02 | A user can download the current (filtered) ticket list as a PDF report. |
| FR-EXP-03 | Exports reflect exactly the rows visible under the active filters/search. |
| FR-EXP-04 | Exports contain only the logged-in user's data. |
| FR-EXP-05 | CSV and PDF generation are delivered as reusable Claude Code skills; brand styling is delivered as a skill so reports look consistent. |

### 6.7 GenAI Chat Assistant (FR-AI)
| ID | Requirement |
|---|---|
| FR-AI-01 | The assistant is presented as a chat: user messages right-aligned, assistant messages left-aligned, in a scrolling conversation. |
| FR-AI-02 | The conversation has memory: the assistant retains prior turns in the session and answers follow-up questions in context. |
| FR-AI-03 | The user can ask free-text questions about their own tickets (counts by status/priority/category, recent activity, trends) and receive accurate, data-grounded answers. |
| FR-AI-04 | The assistant answers using only the logged-in user's data; it never fabricates figures and never exposes another user's data. |
| FR-AI-05 | For data questions, the LLM (OpenAI API via LangChain) produces a parameterized SQL query scoped to the current user. |
| FR-AI-06 | A LangGraph safety loop validates each generated query: generate SQL → validate it is a single read-only, user-scoped SELECT → regenerate if unsafe → run → summarize conversationally. |
| FR-AI-07 | Any query that is not a single user-scoped SELECT is rejected and never executed (code-level runtime guard, since a single root DB user is in use). |
| FR-AI-08 | Conversation memory persists for the duration of the session; clearing the chat resets the memory. |

### 6.8 Demo Data Seeding (FR-SEED)
| ID | Requirement |
|---|---|
| FR-SEED-01 | A seeding capability generates N demo users, each with M realistic, categorized tickets across statuses and priorities. |
| FR-SEED-02 | Seeded tickets carry plausible activity histories (created, progressed, resolved) consistent with the workflow. |
| FR-SEED-03 | Seeding is idempotent or clearly resettable (re-running does not corrupt data). |
| FR-SEED-04 | Delivered as a Claude Code skill (`/seed-data <users> <tickets>`), invokable manually or autonomously during testing. |

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Security | Secrets only via env vars; never committed or logged. Bcrypt for passwords. AI-generated SQL validated as read-only and user-scoped before execution. |
| NFR-02 | Data integrity | Every status/field change is atomic (single DB transaction) and produces exactly one history row. |
| NFR-03 | Privacy | Email addresses and other PII are never written to logs; the chat never leaks cross-user data. |
| NFR-04 | Usability | Clean dashboard-first flow; clear validation messages; chat behaves like a familiar messaging app. |
| NFR-05 | Performance | Common operations respond within ~2s on local MySQL; chat responses return promptly. |
| NFR-06 | Maintainability | Code organized by feature; each feature maps to a spec; conventions defined in CLAUDE.md. |
| NFR-07 | Testability | Every acceptance criterion has at least one automated test (test-writer subagent); end-to-end flow is scripted. |
| NFR-08 | Presentation & Deployment | UI is visually polished, consistent with the design system (Section 12), and ready to deploy with documented, env-based setup. |

## 8. Data Model (High Level)

- **users** — id, username (unique), email, password_hash, role, created_at.
- **tickets** — id, ticket_number (unique), requester_id (FK), assignee_id (FK, nullable), title, description, category, priority, status, created_at, updated_at.
- **ticket_activity** — id, ticket_id (FK), actor_id (FK), action, field_changed, old_value, new_value, created_at. (Immutable audit log.)
- **chat_messages** (optional) — id, user_id (FK), role (user/assistant), content, created_at.

*Detailed schema (column types, indexes, constraints) is owned by the relevant spec, not the BRD.*

## 9. Business Rules

| ID | Rule |
|---|---|
| BR-01 | A new ticket always starts in status OPEN. |
| BR-02 | Allowed status transitions: OPEN → IN_PROGRESS; IN_PROGRESS → RESOLVED; RESOLVED → CLOSED; RESOLVED → IN_PROGRESS (reopen). |
| BR-03 | CLOSED is terminal; reopening requires going via RESOLVED, never a direct CLOSED → OPEN jump. |
| BR-04 | Every state or field change has a corresponding activity-history row (no silent edits). |
| BR-05 | A user can only ever read/write their own tickets (as requester or assignee) and chat. |
| BR-06 | The chat assistant reads only; it never writes to any table and is always user-scoped. |

## 10. Roles & Permissions

| Role | Permissions |
|---|---|
| Visitor | Register, log in. |
| End user (requester) | Create tickets, view/search/filter own tickets, update own tickets along allowed transitions, export, chat — all scoped to self. |
| Support agent | All requester abilities, plus work tickets assigned to them and progress them through the workflow. |
| AI chat layer (system) | Read-only SELECT scoped to the requesting user (enforced in code over the shared root connection). |

## 11. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit (custom-styled to the brand design system) |
| Database | MySQL (local) via PyMySQL; single root user for this demo |
| Auth | bcrypt + Streamlit session state |
| GenAI | OpenAI model via LangChain (chat with memory + guarded text-to-SQL); LangGraph for the validation loop |
| Reporting | CSV + PDF export delivered as Claude Code skills |
| Testing | pytest (unit + integration), scripted end-to-end |
| Dev tooling | Claude Code: CLAUDE.md, specs/, slash commands, hooks, subagents, skills, plugin |
| Deployment | Local run + deployable (Streamlit-compatible host); env-based config |

## 12. UI / UX & Design System

The application must look polished and presentable for demos and deployment. The visual language uses the brand palette in solid colours (no gradients), a white background, and black text throughout.

### 12.1 Colour Palette
| Colour | Hex | Primary usage |
|---|---|---|
| Green | `#00E676` | Positive / success states, primary call-to-action buttons, user chat bubbles. |
| Purple | `#8A5CF6` | Brand / primary surfaces: sidebar, headers, section titles, navigation highlights. |
| Pink | `#FF2D95` | Accents and emphasis: badges, active states, secondary highlights. |

### 12.2 Layout & Typography
- Background: white. Text: black for all content (labels, values, chat text).
- Clean sans-serif typography; consistent spacing; card-based dashboard with rounded corners and subtle shadows.
- Left sidebar navigation in purple: Dashboard, Create Ticket, Ticket List, Reports, Chat Assistant.
- Status shown as coloured badges (e.g. green for Resolved, pink for Urgent priority) with clear contrast.
- Buttons use solid brand colours (green for primary confirm actions, pink for secondary accents) with black or white labels as contrast requires.
- Ticket counts shown in large, prominent summary cards on the dashboard.

### 12.3 Chat Assistant (messaging-app style)
- Scrolling conversation view: assistant messages left-aligned in white bubbles with a purple border; user messages right-aligned in solid green (`#00E676`) bubbles.
- All bubble text is black for readability.
- A fixed input box at the bottom with a send button; pressing send appends to the conversation and the assistant replies in context.
- Typing/loading indicator while the assistant responds; conversation scrolls to the latest message.
- A visible "Clear chat" control resets the conversation and its memory.

## 13. Success Criteria (Business-Level Acceptance)

| ID | Criterion |
|---|---|
| SC-01 | A new user can register, log in, create a ticket, move it through the workflow, and view it in their list without errors. |
| SC-02 | After login the dashboard correctly shows status counts and the user's recent tickets. |
| SC-03 | An illegal status transition is always blocked, and every legal change appears in the activity history. |
| SC-04 | A user can hold a continuous, multi-turn chat where the assistant remembers context and answers at least 5 distinct ticket questions correctly from their own data. |
| SC-05 | A crafted prompt-injection input cannot read another user's data or mutate any data. |
| SC-06 | The UI matches the design system (palette, white background, black text) and is presentable / deployable. |
| SC-07 | Every functional requirement traces to a spec; every spec acceptance criterion traces to a passing test. |

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| AI layer runs on a full-privilege (root) DB connection | LangGraph runtime guard validates generated SQL is a single user-scoped SELECT before execution; demo-only; production would use a read-only user. |
| LLM fabricates figures or leaks other users' data | All data answers grounded in user-scoped query results; user scope enforced in code; no cross-user access path. |
| Illegal or silent status changes corrupt the audit trail | Transition rules enforced in code; every change atomic with a mandatory history row (BR-02, BR-04). |
| Destructive commands during the build | Pre-hooks block dangerous operations (e.g. DROP TABLE) and auto-approve safe ones; post-hook formats and tests on every edit. |
| Scope creep beyond teaching goal | Out-of-scope list (3.2) is binding; new ideas become future modules. |

## 15. Traceability & Delivery Flow

- BRD (this document) defines what the business needs (FR / NFR / BR with IDs).
- Specs (one per feature) translate requirements into precise behaviour + acceptance criteria, each referencing BRD IDs.
- Development implements against the approved spec only (spec → implement → verify loop via custom slash commands).
- Test cases are written from each spec's acceptance criteria (test-writer subagent); the spec-reviewer subagent checks implementation vs spec; the security-auditor subagent enforces NFR-01/02/03.
- Hooks enforce conventions continuously (format/test on edit; block destructive commands).
- End-to-end testing validates the full user journey and SC-01…SC-07.
- The complete toolset (commands, hooks, agents, skills) is packaged as a Claude Code plugin for reuse.

## 16. Glossary

- **Status workflow** — the fixed set of allowed ticket states and the legal transitions between them.
- **Activity history** — an immutable audit log; one row per create / status change / field update.
- **Text-to-SQL** — converting a natural-language question into a SQL query.
- **Guarded** — the generated SQL is validated (single, read-only, user-scoped SELECT) before execution.
- **Conversation memory** — prior turns retained so the assistant answers follow-ups in context.
- **Spec-driven development** — code is written only against an approved specification with explicit acceptance criteria.
- **Runtime guard vs build-time agent** — the validation that ships inside the app (LangGraph loop) versus the security-auditor subagent that only runs while building.

*End of BRD v1.0 — pending review and sign-off before specs are derived.*
