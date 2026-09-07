"""Composite-response composition: R = C @ G with full error propagation.

Pipeline glue for the kc761sim matrix modes: reads the merged G ROOT file
(the hadd output of the worker histograms) and composes the true response
R = C @ G from the input calibration file's deposition response C -- via
:func:`kc761util.respcomp.compose_response` with the analytic Jacobian of
:mod:`kc761calib.matrixjac` -- writing the final ROOT file through
:file:`kc761sim/matrix2root.cxx` (the temporary-binary export convention
of kc761calib/kc761unfold):

    gamma(E_gamma) -> crystal deposition (physics, G) -> channel
    (calibration + resolution smearing, C);  R = C @ G.

G is normalized per column with the totals including the zero-deposition
events, so the R column sum equals the detection efficiency of that
primary energy (see :mod:`kc761util.respcomp` for the error propagation).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np
import uproot

from kc761calib.matrixjac import build_matrix_jacobian
from kc761util.calibfile import MAX_CHANNELS, load_calib_file
from kc761util.respcomp import compose_response
from kc761util.rootcxxfrontend import format_macro_cmd, run_macro

from .paths import MATRIX_G_HIST_NAME, MATRIX_ZERO_HIST_NAME
from .sources import MatrixSource, mode_metadata

_MAGIC = b"kc761sim-matrix-export-v1\n"

# Column-sum tolerance of the composed matrix (absolute efficiency is
# always <= 1; C columns may exceed 1 by ~1e-8 rounding, as the
# calibration export does).
_COL_SUM_TOL = 1e-6


def _put_i64(fh, value: int) -> None:
    fh.write(np.int64(value).tobytes())


def _put_f64(fh, values) -> None:
    fh.write(np.asarray(values, dtype=np.float64).tobytes())


def _put_text(fh, text: str) -> None:
    fh.write(str(text).encode("ascii", errors="replace") + b"\n")


def _put_block(fh, block_id: int, values) -> None:
    """Write one payload block: its id, then its float64 values.

    ``matrix2root.cxx`` reads each id back and verifies it, so a layout
    drift between the writer and the reader fails loudly instead of
    silently swapping same-shaped blocks.
    """
    _put_i64(fh, block_id)
    _put_f64(fh, values)


def _load_g_histograms(path: str) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           np.ndarray]:
    """Read the merged G counts, zero counts and the shared edge arrays.

    uproot returns ``values()[deposition_bin, primary_bin]`` -- exactly the
    layout the composition works on (the G x axis is the matrix output
    side, the y axis the input side).  Returns ``(g_counts, zero_counts,
    edges, edges)`` where the last two are the G x-axis and y-axis edge
    arrays (identical by construction).
    """
    with uproot.open(path) as f:
        try:
            g_hist = f[MATRIX_G_HIST_NAME]
            zero_hist = f[MATRIX_ZERO_HIST_NAME]
        except KeyError as exc:
            raise RuntimeError(
                f"merged G file {path!r} is missing the worker histograms "
                f"({exc.args[0]}); is it a kc761sim matrix-mode output?") \
                from exc
        # Validate the bin counts BEFORE materializing the dense contents
        # (a corrupt file can declare huge axes whose sparse content
        # compresses to a tiny file; the same cap the calibration reader
        # enforces).
        n_x = len(g_hist.axis(0).edges()) - 1
        n_y = len(g_hist.axis(1).edges()) - 1
        if n_x != n_y or n_x < 1 or n_x > MAX_CHANNELS:
            raise ValueError(
                f"G histogram bin counts {n_x} x {n_y} are inconsistent "
                f"or beyond the supported maximum of {MAX_CHANNELS} in "
                f"{path}")
        if len(zero_hist.axis(0).edges()) - 1 != n_x:
            raise ValueError(
                f"zero-deposition histogram bin count does not match the "
                f"G histogram ({n_x}) in {path}")
        g_counts = np.asarray(
            g_hist.values(), dtype=float)  # [deposition, primary]
        zero_counts = np.asarray(zero_hist.values(), dtype=float)
        x_edges = np.asarray(g_hist.axis(0).edges(), dtype=float)
        y_edges = np.asarray(g_hist.axis(1).edges(), dtype=float)
        z_edges = np.asarray(zero_hist.axis(0).edges(), dtype=float)
    return g_counts, zero_counts, x_edges, y_edges, z_edges


def _validate_inputs(calib, g_counts, zero_counts, x_edges, y_edges,
                     z_edges, source, n_events) -> np.ndarray:
    """Fail fast on every binning/accounting inconsistency before composing.

    Returns the per-column primary-event totals (zero-deposition events
    included), the normalization denominator of the composition.
    """
    n = calib.n_channels
    edges = calib.energy_edges
    if calib.matrix_errors is None:
        raise RuntimeError(
            "the input calibration file stores no per-element errors "
            "(no fSumw2); the composite output needs them for the "
            "deposition response copy and the error propagation")
    if not np.array_equal(np.asarray(source.axis.edges), edges):
        raise ValueError(
            "the source primary axis does not match the calibration "
            "deposition-energy edges of the input file")
    if g_counts.shape != (n, n):
        raise ValueError(
            f"G histogram shape {g_counts.shape} does not match the "
            f"calibration matrix {(n, n)}")
    if not np.array_equal(x_edges, edges) or not np.array_equal(y_edges, edges):
        raise ValueError(
            "G histogram binning does not match the calibration "
            "deposition-energy edges (both axes must reuse them exactly)")
    if zero_counts.shape != (n,) or not np.array_equal(z_edges, edges):
        raise ValueError(
            "zero-deposition histogram binning does not match the "
            "calibration deposition-energy edges")
    if (g_counts < 0).any() or (zero_counts < 0).any():
        raise ValueError("G or zero-deposition counts contain negative values")

    totals = g_counts.sum(axis=0) + zero_counts
    if int(totals.sum()) != n_events:
        raise ValueError(
            f"total scored events {int(totals.sum())} do not match the "
            f"requested n_events {n_events}")
    axis = source.axis
    expected = n_events // axis.n_active
    active = np.array(axis.active, dtype=int)
    if np.any(np.abs(totals[active] - expected) > 1.5):
        raise ValueError(
            "per-column event counts deviate from the fixed-per-column "
            f"scheme (expected {expected} +- 1 per active column)")
    if (totals[np.setdiff1d(np.arange(n), active)] != 0).any():
        raise ValueError(
            "skipped (negative-energy) primary columns received events")
    return totals


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_matrix_export(calib, g_counts, composed, source,
                        n_events, seed, calib_path: str) -> str:
    """Serialize everything matrix2root.cxx needs into a temporary file.

    Returns the temporary file path; the ROOT writer deletes it after a
    successful conversion, and callers keep it on failure for inspection
    (the kc761calib convention).
    """
    n = calib.n_channels
    mode, mode_name, geometry_name, geometry_param = mode_metadata(source)
    edges = calib.energy_edges

    fd, path = tempfile.mkstemp(prefix="kc761sim-matrix-export-",
                                suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_MAGIC)
            _put_text(fh, mode_name)
            _put_text(fh, geometry_name)
            _put_text(fh, source.angular)
            _put_text(fh, os.path.abspath(calib_path))
            _put_text(fh, _sha256(calib_path))
            _put_text(fh, calib.calib_formula)
            _put_text(fh, calib.resol_formula)
            _put_text(fh, calib.param_order)
            _put_i64(fh, n)
            _put_i64(fh, mode)
            _put_f64(fh, edges)
            _put_f64(fh, geometry_param)
            _put_f64(fh, [n_events, seed])
            _put_f64(fh, calib.calib_coeffs)
            _put_f64(fh, calib.calib_errors)
            _put_f64(fh, calib.resol_params)
            _put_f64(fh, calib.resol_errors)
            _put_f64(fh, calib.resol_e_ref)
            _put_f64(fh, calib.param_cov.ravel())
            _put_block(fh, 1, calib.matrix.ravel())
            _put_block(fh, 2, calib.matrix_errors.ravel())
            # G is carried as [deposition, primary]; the macro writes
            # row-major with row = x (deposition), so no transpose is
            # needed.  Unit-weight fills: the stored sumw2 buffer equals
            # the counts.
            _put_block(fh, 3, g_counts.ravel())
            _put_block(fh, 4, g_counts.ravel())
            _put_block(fh, 5, composed.matrix.ravel())
            _put_block(fh, 6, composed.variance.ravel())
            _put_block(fh, 7, composed.efficiency)
            _put_block(fh, 8, composed.efficiency_variance)
    except BaseException:
        os.unlink(path)
        raise
    return path


def compose_matrix_output(
    calib_path: str,
    merged_g_path: str,
    output_path: str,
    source: MatrixSource,
    n_events: int,
    seed: int,
    root_exe: str | None = None,
    calib=None,
) -> int:
    """Compose R = C @ G and write the composite ROOT file; returns 0/1.

    Prints progress like the other frontends; on a ROOT failure the
    temporary export file is kept and the manual re-run command printed.
    ``calib`` optionally carries an already-loaded calibration snapshot
    (the CLI pre-validates the input and passes it to avoid decompressing
    the dense matrix twice); when omitted, the file is loaded here so the
    function stays self-contained for re-running a composition after a
    crash.
    """
    if calib is None:
        calib = load_calib_file(calib_path)
    g_counts, zero_counts, x_edges, y_edges, z_edges = _load_g_histograms(
        merged_g_path)
    totals = _validate_inputs(calib, g_counts, zero_counts, x_edges,
                              y_edges, z_edges, source, n_events)

    print(f"[matrix] composing {calib.n_channels} x {calib.n_channels} "
          f"response from C (calibration) and G (Monte Carlo)")
    jacobian = build_matrix_jacobian(
        calib.energy_edges, calib.calib_coeffs, calib.resol_params)
    composed = compose_response(
        calib.matrix, jacobian, calib.param_cov, g_counts, totals)

    # Physical sanity of the composed matrix before writing anything.
    matrix = composed.matrix
    if (matrix < -1e-12).any():
        raise RuntimeError("composed matrix contains negative entries")
    col_sums = matrix.sum(axis=0)
    if (col_sums > 1.0 + _COL_SUM_TOL).any():
        raise RuntimeError(
            f"composed matrix column sums exceed 1 (max {col_sums.max():.6g})")
    if not np.allclose(col_sums, composed.efficiency, atol=1e-9):
        raise RuntimeError(
            "composed matrix column sums disagree with the efficiency "
            "vector (internal inconsistency)")
    active = np.asarray(source.axis.active, dtype=int)
    print(f"[matrix] detection efficiency: min "
          f"{composed.efficiency[active].min():.4g} max "
          f"{composed.efficiency[active].max():.4g} "
          f"(over {active.size} active primary columns)")

    export_file = write_matrix_export(
        calib, g_counts, composed, source,
        n_events, seed, calib_path)
    rc = run_macro("kc761sim/matrix2root.cxx",
                   [export_file, str(output_path)],
                   root_exe=root_exe, echo_prefix="matrix2root")
    if rc != 0:
        cmd = format_macro_cmd("kc761sim/matrix2root.cxx",
                               [export_file, str(output_path)],
                               root_exe=root_exe)
        re_run = (f"\n[matrix]   manual re-run: {' '.join(cmd)}"
                  if cmd is not None else "")
        print(f"[matrix] error: ROOT export failed (exit code {rc}); the "
              f"temporary export file was kept for inspection:"
              f"\n[matrix]   {export_file}{re_run}",
              file=sys.stderr)
    return rc
