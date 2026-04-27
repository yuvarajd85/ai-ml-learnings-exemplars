Generate a complete AWS technical specification with architecture diagram for a system described in natural language.

## What this command does

1. Invokes the `generate-tech-spec` skill to produce a structured `tech-stack.md` covering compute, storage, messaging, orchestration, security, observability, cost profile, rejected alternatives, and Well-Architected analysis.
2. The `generate-tech-spec` skill internally invokes the `ar-diagram` skill to produce both a Mermaid `.md` and a DrawIO `.drawio` architecture diagram.
3. All three artifacts land under `output/<system_name>/`.

## Usage

```
/tech-spec <describe the system you want to design>
```

**Examples:**
```
/tech-spec a real-time fraud detection pipeline that processes 50k transactions per minute
/tech-spec an order management system for an e-commerce platform with async settlement
/tech-spec a document ingestion and Q&A service using RAG on AWS Bedrock
```

## Steps to execute

1. **Resolve the requirement source:**
   - If `$ARGUMENTS` is non-empty: use it as the requirement text directly.
   - If `$ARGUMENTS` is empty (default): search `input/` in the project root for any `.md` file whose name contains `requirement` (case-insensitive). Use `find input/ -iname "*requirement*.md"` to locate candidates. If exactly one file is found, read it and use its content as the requirement. If multiple files are found, list them and ask the user which to use. If none are found, ask: "No requirement file found in `input/`. Describe the system or add a `requirement*.md` file there."

2. Invoke the `generate-tech-spec` skill with the resolved requirement as input.
3. Follow the skill's 8-step process exactly — do not shortcut.
4. After the skill completes, confirm the three output files with their absolute paths:
   - `output/<system_name>/tech-stack.md`
   - `output/<system_name>/<system_name>.md` (Mermaid diagram)
   - `output/<system_name>/<system_name>.drawio` (DrawIO diagram)
5. Echo the Mermaid diagram inline so the user sees it immediately.
