"""Shared infrastructure for the ROOT C++ macro frontend scripts."""

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


def format_macro_cmd(macro_rel: str, macro_args: list[str], *,
                     root_exe: str | None = None) -> list[str] | None:
    """The exact command ``run_macro`` executes, or None when unavailable.

    Resolves the ROOT executable and the macro path exactly like
    :func:`run_macro`, so callers can display the identical command that
    ran (or would run) instead of re-deriving its quoting.
    """
    root = find_root(root_exe)
    if not root:
        return None
    macro = PROJECT_ROOT / macro_rel
    if not macro.is_file():
        return None
    script = f'{escape(macro)}({",".join(quote(a) for a in macro_args)})'
    return [root, "-l", "-b", "-q", script]


def run_macro(macro_rel: str, macro_args: list[str], *,
              root_exe: str | None = None,
              cwd: Path | None = None,
              echo_prefix: str = "rootcxxfrontend") -> int:
    cmd = format_macro_cmd(macro_rel, macro_args, root_exe=root_exe)
    if cmd is None:
        if find_root(root_exe) is None:
            print(f"[{echo_prefix}] error: 'root' executable not found on PATH "
                  "(use --root)", file=sys.stderr)
        else:
            print(f"[{echo_prefix}] error: macro not found: "
                  f"{PROJECT_ROOT / macro_rel}", file=sys.stderr)
        return 1

    # flush=True keeps the buffered caller output ahead of this line and the
    # macro's own (unbuffered) output, so piped logs stay chronological.
    print(f"[{echo_prefix}] running:", " ".join(cmd), flush=True)

    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    except OSError as exc:
        print(f"[{echo_prefix}] error: failed to run ROOT: {exc}", file=sys.stderr)
        return 1
    return proc.returncode
