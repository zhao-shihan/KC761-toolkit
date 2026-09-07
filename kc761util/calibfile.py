"""Shared reader/validator for kc761calib exports and kc761sim composite files.

Both file kinds carry one response-matrix TH2D plus the calibration
metadata; they differ only in what the matrix's energy (y) axis means:

* kc761calib exports: ``deposition_response_matrix``, channel <-
  energy deposition (the calibration image of the channel bins);
* kc761sim matrix-mode composite files: ``response_matrix``, channel <-
  true primary gamma energy (``deposition_response_matrix`` and
  ``primary_deposition_matrix`` are also present as the intermediate
  factors);

``response_matrix`` is also the name legacy kc761calib exports wrote, so
the reader accepts either name and treats the y axis as "the energy axis
the matrix maps from" without further interpretation.

The snapshot returned is dense and keeps the stored NaN semantics:
``param_cov`` rows/columns may be NaN (undetermined parameters, as
kc761calib writes them) and ``matrix_errors`` may contain NaN; callers
apply their own NaN policy (:mod:`kc761unfold.reader` treats undetermined
parameters as fixed, the kc761sim composition does the same).

Validation covers the geometry assumptions the whole toolkit relies on:
square matrix, uniform channel bins of width 1, a strictly increasing
energy binning, finite parameters, and the toolkit-wide 2^14 bin-count
cap (enforced by the calib2root/unfold2root writers, and previously by
:mod:`kc761unfold.reader` itself).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import uproot

#: Response-matrix object name of kc761sim matrix-mode composite files
#: (channel <- true primary gamma energy) and of legacy kc761calib
#: exports written before the deposition naming.
RESPONSE_HIST_NAME = "response_matrix"
#: Response-matrix object name of kc761calib exports (channel <-
#: energy deposition).
DEPOSITION_RESPONSE_HIST_NAME = "deposition_response_matrix"
#: Transport-matrix object name of kc761sim composite files (x =
#: energy deposition, y = primary energy, Monte Carlo counts).
TRANSPORT_HIST_NAME = "primary_deposition_matrix"

#: Reported-basis parameter order of ``param_cov``.
PARAM_NAMES = ("c0", "c1", "c2", "c3", "b0", "b1", "b2")

#: Sanity cap for the channel count (8 * 2^28 values = 2 GiB per block).
MAX_CHANNELS = 1 << 14


@dataclass
class CalibFile:
    """Validated dense snapshot of a calibration or composite ROOT file.

    ``matrix[i, j]`` is the probability that a count in energy bin ``j``
    is detected in channel bin ``i``.  Channel bins are the uniform
    width-1 bins with integer centers (edges ``-0.5 .. n-0.5``); the
    energy bins are the variable-width bins of the y axis (strictly
    increasing).  ``matrix_errors`` are the stored per-element 1-sigma
    errors (same layout; may contain NaN, or be None when the file stores
    no sumw2 buffer).  ``param_cov`` is the stored 7x7 covariance in the
    reported basis ``(c0, c1, c2, c3, b0, b1, b2)``; NaN rows/columns mark
    undetermined parameters, exactly as written by kc761calib.
    """

    n_channels: int
    channel_edges: np.ndarray  # n + 1, uniform width 1
    energy_edges: np.ndarray  # n + 1, strictly increasing
    matrix: np.ndarray  # (n, n) dense, [channel, energy]
    matrix_errors: np.ndarray | None  # (n, n) stored per-element 1-sigma
    #: Composite files only: the conditional transport matrix
    #: ``p_tilde[deposition, primary] = G / column_sum(G)`` (columns
    #: normalized over the deposited events, the zero-deposition category
    #: excluded).  Combined with the stored response's column sums it
    #: rebuilds the full transport model (efficiency included) for the
    #: systematic-error propagation; None for calibration files.
    transport: np.ndarray | None
    calib_coeffs: np.ndarray  # (c0, c1, c2, c3)
    calib_errors: np.ndarray  # stored 1-sigma of c0..c3
    resol_params: np.ndarray  # (b0, b1, b2)
    resol_errors: np.ndarray  # stored 1-sigma of b0..b2
    resol_e_ref: float
    param_cov: np.ndarray  # (7, 7) reported basis, NaN preserved
    calib_formula: str
    resol_formula: str
    param_order: str


def _response_hist_name(f) -> str:
    """Name of the response matrix present in an open ROOT file, or fail."""
    if RESPONSE_HIST_NAME in f:
        return RESPONSE_HIST_NAME
    if DEPOSITION_RESPONSE_HIST_NAME in f:
        return DEPOSITION_RESPONSE_HIST_NAME
    raise ValueError(
        f"the file does not contain the TH2D {RESPONSE_HIST_NAME!r} (or "
        f"{DEPOSITION_RESPONSE_HIST_NAME!r}); is it a kc761calib export or "
        f"a kc761sim composite-response file?")


def _label(source) -> str:
    """Human-readable file name for an open ROOT file object."""
    path = getattr(source, "file_path", None)
    return str(path) if path else "<open ROOT file>"


def load_calib_file(
    source: str | os.PathLike | uproot.ReadOnlyDirectory,
) -> CalibFile:
    """Read and validate a calibration/composite ROOT file into dense form.

    ``source`` is a ROOT file path (opened here) or an already-open
    ``uproot`` file/directory object; all arrays are materialized before
    returning, so a path-opened file is closed on exit.
    """
    opened = not isinstance(source, uproot.ReadOnlyDirectory)
    if opened:
        path = source
        try:
            file = uproot.open(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"ROOT file not found: {path}") from exc
        label = str(path)
    else:
        file = source
        label = _label(source)

    try:
        hist_name = _response_hist_name(file)
        hist = file[hist_name]
        channel_edges = np.asarray(hist.axis(0).edges(), dtype=float)
        energy_edges = np.asarray(hist.axis(1).edges(), dtype=float)
        # Validate the file-provided bin counts BEFORE materializing the
        # dense contents (the same cap calib2root.cxx enforces).
        if (len(channel_edges) != len(energy_edges)
                or len(channel_edges) < 2
                or len(channel_edges) > MAX_CHANNELS + 1):
            raise ValueError(
                f"response matrix bin counts {len(channel_edges) - 1} x "
                f"{len(energy_edges) - 1} are inconsistent or beyond the "
                f"supported maximum of {MAX_CHANNELS} in {label}")
        values = np.asarray(hist.values(), dtype=float)  # [channel, energy]
        try:
            hist.member("fSumw2")
            errors = np.asarray(hist.errors(), dtype=float)
        except KeyError:
            errors = None  # no sumw2 buffer stored
        transport = None
        if TRANSPORT_HIST_NAME in file:
            # Conditional transport: normalize each primary column over
            # the deposited events (zero-deposition excluded).
            t_hist = file[TRANSPORT_HIST_NAME]
            t_values = np.asarray(
                t_hist.values(), dtype=float)  # [dep, primary]
            t_totals = t_values.sum(axis=0)
            transport = np.divide(t_values, t_totals,
                                  out=np.zeros_like(t_values),
                                  where=t_totals > 0.0)
        try:
            params = np.array(
                [file[name].member("fVal") for name in PARAM_NAMES],
                dtype=float)
            cov_up = np.array(file["param_cov"].member("fElements")[:28],
                              dtype=float)
            calib_errors = np.array(
                [file[f"{name}_err"].member("fVal")
                 for name in PARAM_NAMES[:4]], dtype=float)
            resol_errors = np.array(
                [file[f"{name}_err"].member("fVal")
                 for name in PARAM_NAMES[4:]], dtype=float)
            resol_e_ref = float(file["resol_e_ref"].member("fVal"))
            calib_formula = str(file["calib_formula"].member("fTitle"))
            resol_formula = str(file["resol_formula"].member("fTitle"))
            param_order = str(file["param_order"].member("fTitle"))
        except KeyError as exc:
            raise ValueError(
                f"{label} is missing the expected calibration objects "
                f"({exc.args[0]}); is it a kc761calib export or a kc761sim "
                f"composite-response file?") from exc
    finally:
        if opened:
            file.close()

    n = values.shape[0]
    if values.shape[1] != n:
        raise ValueError(f"response matrix must be square, got {values.shape}")
    if len(channel_edges) != n + 1 or len(energy_edges) != n + 1:
        raise ValueError("response matrix axis lengths inconsistent with the "
                         "matrix dimensions")
    if not np.allclose(np.diff(channel_edges), 1.0, atol=1e-9):
        raise ValueError("channel bins of the response matrix must be "
                         "uniform with width 1")
    if np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError("energy binning of the response matrix is not "
                         "strictly increasing")
    if not np.all(np.isfinite(params)):
        raise ValueError("calibration parameters contain non-finite entries")

    # TMatrixDSym stores the symmetric 7x7 matrix by its upper triangle
    # (28 entries); rebuild the full symmetric form.
    cov = np.zeros((7, 7))
    cov[np.triu_indices(7)] = cov_up
    cov = cov + np.triu(cov, 1).T

    return CalibFile(
        n_channels=n,
        channel_edges=channel_edges,
        energy_edges=energy_edges,
        matrix=values,
        matrix_errors=errors,
        transport=transport,
        calib_coeffs=params[:4],
        calib_errors=calib_errors,
        resol_params=params[4:],
        resol_errors=resol_errors,
        resol_e_ref=resol_e_ref,
        param_cov=cov,
        calib_formula=calib_formula,
        resol_formula=resol_formula,
        param_order=param_order,
    )
