"""Shared infrastructure for the ROOT-macro frontend scripts in this project.

Both frontend scripts (csv2root.py, subbkg.py, ...) follow the same pattern:
locate the ROOT executable, assemble a `root -l -b -q 'macro(args)'` command
line and run it as a subprocess.  The common pieces live here.

Macros live in subdirectories of the project root (kc761util/, kc761ana/, ...)
and are addressed by their relative path, e.g. "kc761ana/subbkg.cxx".
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Project root = parent of kc761util/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def escape(s) -> str:
    """Escape backslashes and double quotes so the value survives ROOT's
    command-line macro-argument parsing."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def quote(s) -> str:
    return '"' + escape(s) + '"'


def find_root(root_exe: str | None = None) -> str | None:
    """Return the ROOT executable path, or None if it cannot be found."""
    if root_exe:
        return root_exe
    return shutil.which("root")


def add_root_option(parser) -> None:
    """Add the common ``--root`` option to an argparse parser."""
    parser.add_argument(
        "--root", default=None,
        help="path to the ROOT executable (default: 'root' found on PATH)",
    )


def run_macro(macro_rel: str, macro_args: list[str], *,
              root_exe: str | None = None,
              cwd: Path | None = None,
              echo_prefix: str = "frontend") -> int:
    """Assemble ``root -l -b -q 'macro("arg1","arg2",...)'`` and run it.

    ``macro_rel`` is the macro path relative to the project root, e.g.
    ``"kc761util/csv2root.cxx"`` or ``"kc761ana/subbkg.cxx"``.

    Returns the ROOT process exit code (non-zero on failure).
    """
    root = find_root(root_exe)
    if not root:
        print(f"[{echo_prefix}] error: 'root' executable not found on PATH (use --root)",
              file=sys.stderr)
        return 1

    macro = PROJECT_ROOT / macro_rel
    if not macro.is_file():
        print(f"[{echo_prefix}] error: macro not found: {macro}", file=sys.stderr)
        return 1

    # Quote the arguments only; the macro path itself must stay unquoted so
    # ROOT's command-line parser recognizes it as the macro name.
    script = f'{escape(macro)}({",".join(quote(a) for a in macro_args)})'
    cmd = [root, "-l", "-b", "-q", script]
    print(f"[{echo_prefix}] running:", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    except OSError as exc:
        print(f"[{echo_prefix}] error: failed to run ROOT: {exc}", file=sys.stderr)
        return 1
    return proc.returncode
