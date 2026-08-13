"""
PostToolUse hook — auto-format changed Python files and run pytest.
Reads the tool-call JSON from stdin.
For .py files: runs ruff format or black (whichever is installed; skips if neither),
then runs pytest and surfaces all output via stderr.
Always exits 0 — this hook is advisory, never blocking.
Requires only Python stdlib (ruff/black/pytest are external tools).
"""
import json
import shutil
import subprocess
import sys


def _file_path(payload: dict) -> str:
    """Extract the file path from Write or Edit tool input."""
    return payload.get("tool_input", {}).get("file_path", "")


def _emit(label: str, output: str) -> None:
    """Write labelled output to stderr, skipping if there is nothing to show."""
    text = output.strip()
    if text:
        print(f"[{label}]\n{text}", file=sys.stderr)


def _run(args: list[str], label: str) -> None:
    """Run a subprocess and forward combined stdout+stderr to our stderr."""
    try:
        result = subprocess.run(args, capture_output=True, text=True)
        _emit(label, result.stdout + result.stderr)
    except FileNotFoundError:
        # Binary not found at runtime despite shutil.which check — skip silently
        pass


def _format(path: str) -> None:
    """Format a single Python file with ruff or black, whichever is installed."""
    if shutil.which("ruff"):
        _run(["ruff", "format", path], "ruff format")
    elif shutil.which("black"):
        _run(["black", path], "black")
    # Neither installed — skip without noise


def _pytest() -> None:
    """Run the full pytest suite and surface output via stderr."""
    if shutil.which("pytest"):
        runner = ["pytest"]
    elif shutil.which("python3"):
        runner = ["python3", "-m", "pytest"]
    elif shutil.which("python"):
        runner = ["python", "-m", "pytest"]
    else:
        print("[pytest] pytest not found — skipping", file=sys.stderr)
        return

    _run(runner, "pytest")


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    if payload.get("tool_name") not in ("Write", "Edit"):
        sys.exit(0)

    path = _file_path(payload)
    if not path.endswith(".py"):
        sys.exit(0)

    _format(path)
    _pytest()

    sys.exit(0)  # always — hook is advisory only


if __name__ == "__main__":
    main()
