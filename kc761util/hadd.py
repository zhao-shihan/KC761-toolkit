"""Merge ROOT files with the ``hadd`` executable (ROOT's file merger).

The merge is delegated to ROOT's ``hadd`` rather than rewritten with
uproot: ``hadd`` merges TTrees entry-by-entry and sums TH* histograms
together with their ``sumw2`` buffers, so the merged file keeps exactly the
per-bin weight-squared sums that carry the Monte Carlo statistical errors
of simulated spectra.  The interface mirrors
:mod:`kc761util.rootcxxfrontend`: an explicit executable path wins,
otherwise ``hadd`` is looked up next to the ROOT executable (they ship in
the same ``bin/`` directory) and on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_hadd(hadd_exe: str | None = None,
              root_exe: str | None = None) -> str | None:
    """Locate the ``hadd`` executable.

    Resolution order: the explicit ``hadd_exe``; the sibling ``hadd`` next
    to the ROOT executable selected by ``root_exe`` -- or next to ``root``
    on PATH (they ship in the same ``bin/`` directory, so an explicit
    ``--root <rootdir>/bin/root`` selects the matching hadd); and ``hadd``
    on PATH.
    """
    if hadd_exe:
        return hadd_exe
    root = root_exe or shutil.which("root")
    if root:
        sibling = Path(root).with_name("hadd")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("hadd")


def add_hadd_option(parser) -> None:
    """Add the ``--hadd`` option selecting the hadd executable."""
    parser.add_argument(
        "--hadd", default=None,
        help="path to the hadd executable used to merge ROOT files "
             "(default: 'hadd' found on PATH)",
    )


def merge_root_files(output_path: str, input_paths: list[str], *,
                     hadd_exe: str | None = None,
                     root_exe: str | None = None,
                     force: bool = True,
                     echo_prefix: str = "hadd") -> int:
    """Merge ROOT files into one with hadd; returns the hadd exit code.

    TTrees are merged entry-by-entry and TH* histograms are summed together
    with their ``sumw2`` buffers.  ``force`` overwrites an existing output
    file (``-f``).  The command is printed before it runs, like
    :func:`kc761util.rootcxxfrontend.run_macro`; callers may rename the
    echo prefix to match their tool.
    """
    hadd = find_hadd(hadd_exe, root_exe)
    if hadd is None:
        print(f"[{echo_prefix}] error: 'hadd' executable not found on PATH "
              "(use --hadd)", file=sys.stderr)
        return 1
    if not input_paths:
        raise ValueError("merge_root_files: no input files to merge")

    cmd = [hadd] + (["-f"] if force else []) + [str(output_path)]
    cmd += [str(path) for path in input_paths]
    print(f"[{echo_prefix}] running:", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd)
    except OSError as exc:
        print(f"[{echo_prefix}] error: failed to run hadd: {exc}",
              file=sys.stderr)
        return 1
    return proc.returncode
