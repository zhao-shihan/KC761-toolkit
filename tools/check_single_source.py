#!/usr/bin/env python3
"""Mechanical single-source check (plan section 5.1, decisions D-60/D-75).

Two independent checks run over the numeric core:

1. **Formula bodies.** Every expression line of the owning core modules and of
   the generated kernels is folded to a whitespace-free snippet; a snippet of
   at least ``MIN_SNIPPET`` characters may only appear in its owning file, its
   generated counterpart and the generator itself. Copy-pasted formula bodies
   in any other module fail the check.
2. **Generated integrity.** ``kc761tool.core._gen.stale_files()`` must be empty:
   the committed kernels have to match the manifest written by
   ``tools/generate_kernels.py``.

Usage: ``python tools/check_single_source.py`` (exit code 0 or 1).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "kc761tool" / "core"
SCAN_ROOTS = (REPO_ROOT / "kc761tool", REPO_ROOT / "tools")
GENERATED_DIR = CORE_DIR / "_gen"
GENERATOR = "tools/generate_kernels.py"
MIN_SNIPPET = 60
OPERATORS = ("*", "+", "-", "/", "**")


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _expression_snippets(path: Path) -> set[str]:
    snippets: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not any(operator in stripped for operator in OPERATORS):
            continue
        compact = re.sub(r"\s+", "", stripped)
        if len(compact) >= MIN_SNIPPET:
            snippets.add(compact)
    return snippets


def _snippet_owners(files: list[Path]) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for path in files:
        relative = _relative(path)
        for snippet in _expression_snippets(path):
            owners.setdefault(snippet, set()).add(relative)
    return owners


def _allowed(snippet_owner: str) -> set[str]:
    """Files that may legitimately repeat a snippet owned by ``snippet_owner``."""
    allowed = {snippet_owner}
    if snippet_owner.startswith("kc761tool/core/_gen/"):
        allowed.add(GENERATOR)
    return allowed


def check_snippets(files: list[Path]) -> list[str]:
    owners = _snippet_owners(files)
    violations: list[str] = []
    for path in files:
        relative = _relative(path)
        text = re.sub(r"\s+", "", path.read_text(encoding="utf-8"))
        for snippet, snippet_owners in owners.items():
            if relative in snippet_owners:
                continue
            for owner in snippet_owners:
                if relative in _allowed(owner):
                    continue
                if snippet in text:
                    violations.append(
                        f"{relative}: contains a formula body owned by {owner}: "
                        f"{snippet[:80]}..."
                    )
    return violations


def check_generated() -> list[str]:
    sys.path.insert(0, str(REPO_ROOT))
    from kc761tool.core import _gen  # noqa: PLC0415 - deferred so the CLI stays light

    stale = _gen.stale_files()
    if stale:
        return ["generated kernels are stale: " + ", ".join(stale)]
    return []


def check_stubs() -> list[str]:
    violations: list[str] = []
    for path in sorted(CORE_DIR.glob("*.py")):
        if "NotImplementedError" in path.read_text(encoding="utf-8"):
            violations.append(f"{_relative(path)}: still contains NotImplementedError")
    return violations


def main() -> int:
    files = _source_files()
    violations = check_snippets(files) + check_generated() + check_stubs()
    if violations:
        print("single-source check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print(f"single-source check passed ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
