# Claude Code Project Workshop

---

## Module 1: Project Introduction and the Claude Code Way of Building

### Overview
Introduction to building an AI-assisted Ticket Management WebApp end-to-end using Claude Code.

### Topics
- Overview of the AI-assisted Ticket Management WebApp we will build
- How Claude Code changes app development: spec-driven development instead of ad-hoc prompting
- End-to-end architecture of the solution:
  - Streamlit UI
  - MySQL database
  - LangChain/LangGraph chat assistant powered by an OpenAI model
- Key concepts:
  - The `.claude` folder
  - `CLAUDE.md` project rulebook
  - The context window (Claude's working memory)
- Project outcomes and learning goals

### Outcome
Participants understand what they will build and the core philosophy of building applications with Claude Code.

---

## Module 2: Environment and Project Setup

### Overview
Configure a fully working Claude Code project with a Streamlit and MySQL foundation.

### Topics
- Installing required Python libraries and dependencies:
  - Streamlit
  - PyMySQL
  - LangChain
  - LangGraph
  - OpenAI integration
- Setting up the `.claude` folder and `settings.json`:
  - Permissions
  - Guardrails
- Writing the project rulebook in `CLAUDE.md`:
  - Conventions
  - Security rules
  - Ticket status rules
- Creating custom slash commands:
  - `/spec`
  - `/implement`
  - `/verify`
- Running a Streamlit "Hello World" and connecting it to MySQL via PyMySQL

### Outcome
Participants have a configured Claude Code project with a working Streamlit + MySQL foundation.

---

## Module 3: Building the Core App — Tickets, Dashboard, and Guardrails

### Overview
Build the core application features using the spec-driven loop and secure the project with hooks.

### Topics
- Spec-driven build of login and dashboard:
  - Ticket counts
  - Recent tickets
  - Quick actions
- Create-ticket feature using the spec → implement → test loop
- Introducing Hooks:
  - **Pre-hooks**: block destructive commands (e.g., `DROP TABLE`)
  - **Allow hooks**: auto-approve safe commands (tests, read-only queries)
  - **Post-hook**: auto-format code and run tests after every edit
- Ticket update and status workflow with activity history tracking

### Outcome
Participants can build features using the spec-driven loop and secure the project using hooks.

---

## Module 4: Search, Filters, Export, and Quality Agents

### Overview
Build advanced features using Skills and improve code quality and security with agents.

### Topics
- Ticket list with filters:
  - Status
  - Priority
  - Category
  - Date range
- Exporting reports as CSV and PDF using reusable Skills
- Brand styling delivered as a Skill for consistent UI
- Introducing agents:
  - **Spec-reviewer agent**
  - **Security-auditor agent**
- Context window management:
  - Clearing
  - Compacting
  - Working feature-by-feature

### Outcome
Participants can build advanced features using Skills and improve code quality and security with agents.

---

## Module 5: The AI Chat Assistant and Packaging

### Overview
Build a safe, memory-enabled AI chat assistant and package the system as a reusable Claude Code Plugin.

### Topics
- Building a chat assistant with LangChain and an OpenAI model for ticket queries
- Connecting OpenAI to MySQL (text-to-SQL) with conversation memory
- LangGraph safety loop:
  1. SQL generation
  2. Validation
  3. Execution
  4. Summarization
- Two-layer safety model:
  - Runtime guard
  - Security auditor agent
- Testing against:
  - Prompt injection risks
  - Data leakage risks
- Packaging the system as a Claude Code Plugin for reuse and sharing
- Recap and next steps

### Outcome
Participants can build a safe, memory-enabled AI assistant and package it as a reusable plugin.

---
