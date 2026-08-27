"""Shared infrastructure for the ROOT-macro frontend scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def escape(s) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def quote(s) -> str:
    return '"' + escape(s) + '"'


def find_root(root_exe: str | None = None) -> str | None:
    if root_exe:
        return root_exe
    return shutil.which("root")


def add_root_option(parser) -> None:
    parser.add_argument(
        "--root", default=None,
        help="path to the ROOT executable (default: 'root' found on PATH)",
    )


def run_macro(macro_rel: str, macro_args: list[str], *,
              root_exe: str | None = None,
              cwd: Path | None = None,
              echo_prefix: str = "frontend") -> int:
    root = find_root(root_exe)
    if not root:
        print(f"[{echo_prefix}] error: 'root' executable not found on PATH (use --root)",
              file=sys.stderr)
        return 1

    macro = PROJECT_ROOT / macro_rel
    if not macro.is_file():
        print(f"[{echo_prefix}] error: macro not found: {macro}", file=sys.stderr)
        return 1

    script = f'{escape(macro)}({",".join(quote(a) for a in macro_args)})'
    cmd = [root, "-l", "-b", "-q", script]
    print(f"[{echo_prefix}] running:", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    except OSError as exc:
        print(f"[{echo_prefix}] error: failed to run ROOT: {exc}", file=sys.stderr)
        return 1
    return proc.returncode
