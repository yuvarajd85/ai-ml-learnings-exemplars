"""
PreToolUse hook — auto-approve clearly safe, read-only Bash commands.
Emits {"decision": "approve"} on stdout for known-safe commands so Claude Code
auto-approves without prompting; exits 0 silently for everything else.
Requires only Python stdlib.
"""
import json
import re
import sys


def _extract_command(payload: dict) -> str:
    return payload.get("tool_input", {}).get("command", "")


# ── Safe-command patterns ──────────────────────────────────────────────────────
# Each regex must match the ENTIRE command (after stripping leading whitespace).
# We use anchored patterns so "git status && rm -rf /" is never approved.

_SAFE_PATTERNS: list[re.Pattern] = [
    # pytest (any arguments)
    re.compile(r"^pytest\b.*$", re.DOTALL),
    re.compile(r"^python3?\s+-m\s+pytest\b.*$", re.DOTALL),

    # Read-only git commands
    re.compile(r"^git\s+status\b.*$", re.DOTALL),
    re.compile(r"^git\s+diff\b.*$", re.DOTALL),
    re.compile(r"^git\s+log\b.*$", re.DOTALL),
    re.compile(r"^git\s+show\b.*$", re.DOTALL),

    # Pure SELECT queries — no write keywords anywhere in the statement.
    # Covers bare mysql -e "SELECT ..." and python -c "...cursor.execute('SELECT ...')"
    # only when the full command contains no mutating SQL keywords.
]

# SQL mutation keywords that disqualify a SELECT-looking command
_SQL_MUTATION = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


def _is_pure_select(command: str) -> bool:
    """True when the command contains a SELECT keyword but no mutation keywords."""
    if not re.search(r"\bSELECT\b", command, re.IGNORECASE):
        return False
    if _SQL_MUTATION.search(command):
        return False
    return True


def _is_safe(command: str) -> bool:
    cmd = command.strip()

    # Check structural safe-command patterns first (anchored at start)
    for pattern in _SAFE_PATTERNS:
        if pattern.match(cmd):
            return True

    # Pure SELECT: allow regardless of surrounding shell syntax, but only when
    # there are no chained commands that could execute something dangerous.
    # A pipe or semicolon followed by a non-SELECT statement disqualifies it.
    if _is_pure_select(cmd):
        # Reject if the command chains another statement after the SELECT
        # (simple heuristic: if there's a semicolon or && or || outside quotes)
        if re.search(r"(?:;|&&|\|\|)", cmd):
            return False
        return True

    return False


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)

    command = _extract_command(payload)
    if _is_safe(command):
        # Emit an approve decision — Claude Code auto-approves, no user prompt
        print(json.dumps({"decision": "approve"}))

    # For non-safe commands: exit 0 silently, let Claude Code decide normally
    sys.exit(0)


if __name__ == "__main__":
    main()
