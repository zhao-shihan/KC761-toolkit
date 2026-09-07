"""Serialize an UnfoldResult and convert it to ROOT via unfold2root.cxx.

The payload is a small binary file (magic line, raw blocks) read by the
C++ macro, following the kc761calib export convention; the macro writes
the TH1D/TH2D/TParameter objects and deletes the temporary file on
success (it is kept for inspection on failure).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from kc761util.rootcxxfrontend import format_macro_cmd, run_macro

from .types import UnfoldResult

_MAGIC = "kc761unfold export v1\n"


def _put(fh, arr: np.ndarray) -> None:
    np.asarray(arr, dtype=float).tofile(fh)


def _put_i64(fh, value: int) -> None:
    np.int64(value).tofile(fh)


def write_export_file(result: UnfoldResult) -> Path:
    """Serialize the result to a temporary file; return its path."""
    fh = tempfile.NamedTemporaryFile(mode="wb", suffix=".kc761unfold",
                                     delete=False)
    with fh:
        fh.write(_MAGIC.encode("ascii"))
        _put_i64(fh, 1 if result.calib_only else 0)
        _put_i64(fh, result.n_bins)
        _put_i64(fh, result.channel_low)
        _put_i64(fh, result.channel_high)
        _put(fh, result.energy_edges)
        _put(fh, result.counts)
        _put(fh, result.sigma_total)
        _put_i64(fh, 1 if result.refolded is not None else 0)
        if result.refolded is not None:
            _put(fh, result.refolded)
        _put_i64(fh, 1 if result.stat_cov is not None else 0)
        if result.stat_cov is not None:
            _put(fh, np.asarray(result.stat_cov, dtype=float).ravel())
        _put_i64(fh, 1 if result.syst_cov is not None else 0)
        if result.syst_cov is not None:
            _put(fh, np.asarray(result.syst_cov, dtype=float).ravel())
        s = result.settings
        _put(fh, np.array([s.syst_frac, s.alpha, s.mask_p0, s.mask_floor,
                           s.resol_frac, s.energy_low, s.energy_high]))
        _put_i64(fh, s.k)
        _put_i64(fh, s.snip_iter)
        _put(fh, np.array([result.chi2 if result.chi2 is not None else 0.0,
                           result.pen_cost if result.pen_cost is not None
                           else 0.0]))
        _put_i64(fh, result.ndof if result.ndof is not None else 0)
        _put_i64(fh, result.n_iter if result.n_iter is not None else 0)
    return Path(fh.name)


def run_export(result: UnfoldResult, root_out: str | Path,
               root_exe: str | None = None) -> int:
    """Write the ROOT output file; returns the macro exit code.

    On failure the temporary export file is kept and its path plus the
    manual re-run command are printed, following the kc761calib
    convention.
    """
    export_file = write_export_file(result)
    rc = run_macro("kc761unfold/unfold2root.cxx",
                   [str(export_file), str(root_out)],
                   root_exe=root_exe, echo_prefix="unfold2root")
    if rc != 0:
        cmd = format_macro_cmd("kc761unfold/unfold2root.cxx",
                               [str(export_file), str(root_out)],
                               root_exe=root_exe)
        re_run = (f"\n[unfold]   manual re-run: {' '.join(cmd)}"
                  if cmd is not None else "")
        print(f"[unfold] error: ROOT export failed (exit code {rc}); the "
              f"temporary export file was kept for inspection:"
              f"\n[unfold]   {export_file}{re_run}",
              file=sys.stderr)
    return rc
