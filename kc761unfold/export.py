"""Serialize an UnfoldResult and convert it to ROOT via unfold2root.cxx.

The payload is a small binary file (magic line, raw blocks) read by the
C++ macro, following the kc761calib export convention; the macro writes
the TH1D/TH2D/TParameter objects and deletes the temporary file on
success (it is kept for inspection on failure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from kc761util.binexport import ExportWriter
from kc761util.rootcxxfrontend import format_macro_cmd, run_macro

from .types import UnfoldResult

_MAGIC = b"kc761unfold export v2\n"


def write_export_file(result: UnfoldResult) -> Path:
    """Serialize the result to a temporary file; return its path.

    The per-bin uncertainty bands (total, statistical and systematic)
    are exported explicitly; the full covariance matrices are not part
    of the output (the diagonals carry the exported 1-sigma bands, and
    the zero-content/bound bins keep their 0 entries).
    """
    writer = ExportWriter("kc761unfold-export-", _MAGIC)
    with writer:
        writer.put_i64(1 if result.calib_only else 0)
        writer.put_i64(result.n_bins)
        writer.put_i64(result.channel_low)
        writer.put_i64(result.channel_high)
        writer.put_f64(result.energy_edges)
        writer.put_f64(result.counts)
        writer.put_f64(result.sigma_total)
        writer.put_f64(result.sigma_stat)
        writer.put_f64(result.sigma_syst)
        writer.put_i64(1 if result.refolded is not None else 0)
        if result.refolded is not None:
            writer.put_f64(result.refolded)
        s = result.settings
        writer.put_f64(np.array([s.alpha, s.mask_z0, s.mask_floor,
                                  s.syst_frac, s.energy_low, s.energy_high]))
        writer.put_i64(s.k)
        writer.put_i64(s.snip_iter)
        writer.put_f64(np.array(
            [result.chi2 if result.chi2 is not None else 0.0,
             result.pen_cost if result.pen_cost is not None else 0.0]))
        writer.put_i64(result.ndof if result.ndof is not None else 0)
        writer.put_i64(result.n_iter if result.n_iter is not None else 0)
    return Path(writer.path)


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
