"""
PreToolUse hook — block destructive Bash operations.
Reads a tool-call JSON object from stdin.
Exits 2 (DENY) with a reason on stderr if the command contains a destructive pattern.
Exits 0 (allow) otherwise.
Requires only Python stdlib.
"""
import json
import re
import sys


def _extract_command(tool_input: dict) -> str:
    """Return the Bash command string from the tool input, or empty string."""
    return tool_input.get("command", "")


# Patterns that are always blocked, regardless of context.
# Each entry is (human-readable label, compiled regex).
_ALWAYS_BLOCK: list[tuple[str, re.Pattern]] = [
    (
        "DROP TABLE",
        re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    ),
    (
        "DROP DATABASE",
        re.compile(r"\bDROP\s+DATABASE\b", re.IGNORECASE),
    ),
    (
        "TRUNCATE TABLE",
        re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
    ),
    (
        "rm -rf",
        re.compile(r"\brm\s+(-\w*r\w*f|-\w*f\w*r)\b", re.IGNORECASE),
    ),
    (
        "DELETE without WHERE clause",
        # Matches DELETE FROM <table> with no WHERE keyword following it
        # (looks ahead to end-of-string or a semicolon / closing quote)
        re.compile(
            r"\bDELETE\s+FROM\s+\S+\s*(?:;|$|[\"'`])",
            re.IGNORECASE,
        ),
    ),
]

# DELETE that has no WHERE at all anywhere after the FROM clause
_DELETE_NO_WHERE = re.compile(
    r"\bDELETE\b(?!.*\bWHERE\b)",
    re.IGNORECASE | re.DOTALL,
)


def _is_destructive(command: str) -> tuple[bool, str]:
    """
    Return (True, reason) if the command is destructive, else (False, "").
    """
    for label, pattern in _ALWAYS_BLOCK:
        if pattern.search(command):
            return True, label

    # Separate check: DELETE present but no WHERE anywhere in the statement
    if re.search(r"\bDELETE\b", command, re.IGNORECASE):
        if not re.search(r"\bWHERE\b", command, re.IGNORECASE):
            return True, "DELETE without a WHERE clause"

    return False, ""


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Not valid JSON — let Claude handle it, don't block
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    command = _extract_command(payload.get("tool_input", {}))
    blocked, reason = _is_destructive(command)

    if blocked:
        print(
            f"BLOCKED: '{reason}' detected in Bash command.\n"
            f"This operation is not allowed during spec-driven development.\n"
            f"If you need to reset the database schema, run db/schema.sql manually.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
