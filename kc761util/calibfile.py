"""Shared reader/validator for kc761calib exports.

The file carries one TH2D deposition-to-channel matrix plus the
calibration metadata.  The returned snapshot is dense and keeps the
stored NaN semantics: ``param_cov`` rows/columns may be NaN
(undetermined parameters, as kc761calib writes them) and
``to_channel_uncertainties`` may contain NaN; callers apply their own
NaN policy (see :mod:`kc761unfold.reader`).

Validation covers the geometry assumptions the whole toolkit relies on:
square matrix, uniform channels of width 1, a strictly increasing
energy binning, finite parameters, and the toolkit-wide 2^14 bin-count
cap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import uproot

# To-channel matrix object name of kc761calib exports (channel <-
# energy deposition).
DEPOSITION_TO_CHANNEL_HIST_NAME = "deposition_to_channel"

# Reported-basis parameter order of ``param_cov``.
# ROOT-file format marker of kc761calib exports (see rootmacros.h):
# 3 = the stored matrix uses the fit's 5-sigma + column-renormalization
# convention.
FORMAT_VERSION = "3"
PARAM_ORDER = "c0 c1 c2 c3 b0 b1 b2"
PARAM_NAMES = ("c0", "c1", "c2", "c3", "b0", "b1", "b2")

# Sanity cap for the channel count (8 * 2^28 values = 2 GiB per block).
MAX_CHANNELS = 1 << 14


@dataclass
class CalibFile:
    """Validated dense snapshot of a calibration ROOT file.

    ``matrix[i, j]`` is the probability that a count in energy bin ``j``
    is detected in channel ``i``.  The channels are the uniform
    width-1 bins with integer centers (edges ``-0.5 .. n-0.5``); the
    energy bins are the variable-width bins of the y axis (strictly
    increasing).  ``to_channel_uncertainties`` are the stored per-element
    1-sigma uncertainties (same layout; may contain NaN, or be None when
    the file stores no sumw2 buffer).  ``param_cov`` is the stored 7x7
    covariance in the reported basis ``(c0, c1, c2, c3, b0, b1, b2)``;
    NaN rows/columns mark undetermined parameters, exactly as written by
    kc761calib.
    """

    n_channels: int
    channel_edges: np.ndarray  # n + 1, uniform width 1
    energy_edges: np.ndarray  # n + 1, strictly increasing
    to_channel: np.ndarray  # (n, n) dense, [channel, energy]
    to_channel_uncertainties: np.ndarray | None  # (n, n) per-element 1-sigma
    calib_coeffs: np.ndarray  # (c0, c1, c2, c3)
    calib_uncertainties: np.ndarray  # stored 1-sigma of c0..c3
    resol_params: np.ndarray  # (b0, b1, b2)
    resol_uncertainties: np.ndarray  # stored 1-sigma of b0..b2
    resol_e_ref: float
    param_cov: np.ndarray  # (7, 7) reported basis, NaN preserved
    calib_formula: str
    resol_formula: str
    param_order: str


def _channel_hist_name(f) -> str:
    """Name of the response matrix present in an open ROOT file, or fail."""
    if DEPOSITION_TO_CHANNEL_HIST_NAME in f:
        return DEPOSITION_TO_CHANNEL_HIST_NAME
    raise ValueError(
        f"the file does not contain the TH2D {DEPOSITION_TO_CHANNEL_HIST_NAME!r}; "
        f"is it a kc761calib export file?")


def _label(source) -> str:
    """Human-readable file name for an open ROOT file object."""
    path = getattr(source, "file_path", None)
    return str(path) if path else "<open ROOT file>"


def load_calib_file(
    source: str | os.PathLike | uproot.ReadOnlyDirectory,
) -> CalibFile:
    """Read and validate a calibration ROOT file into dense form.

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
        hist_name = _channel_hist_name(file)
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
            uncertainties = np.asarray(hist.errors(), dtype=float)
        except KeyError:
            uncertainties = None  # no sumw2 buffer stored
        try:
            params = np.array(
                [file[name].member("fVal") for name in PARAM_NAMES],
                dtype=float)
            cov_up = np.array(file["param_cov"].member("fElements")[:28],
                              dtype=float)
            calib_uncertainties = np.array(
                [file[f"{name}_err"].member("fVal")
                 for name in PARAM_NAMES[:4]], dtype=float)
            resol_uncertainties = np.array(
                [file[f"{name}_err"].member("fVal")
                 for name in PARAM_NAMES[4:]], dtype=float)
            resol_e_ref = float(file["resol_e_ref"].member("fVal"))
            calib_formula = str(file["calib_formula"].member("fTitle"))
            resol_formula = str(file["resol_formula"].member("fTitle"))
            version = str(file["format_version"].member("fTitle"))
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{label} has calibration format version {version!r}, "
                    f"expected {FORMAT_VERSION!r}; files predating the "
                    f"5-sigma/column-renormalization convention must not be "
                    f"used with the current response model")
            param_order = str(file["param_order"].member("fTitle"))
        except KeyError as exc:
            raise ValueError(
                f"{label} is missing the expected calibration objects "
                f"({exc.args[0]}); is it a kc761calib export file?") from exc
        if param_order != PARAM_ORDER:
            raise ValueError(
                f"unexpected calibration parameter order {param_order!r} in "
                f"{label} (expected {PARAM_ORDER!r}); the covariance and "
                f"parameter blocks would be misinterpreted")
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
        raise ValueError("channels of the response matrix must be "
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
        to_channel=values,
        to_channel_uncertainties=uncertainties,
        calib_coeffs=params[:4],
        calib_uncertainties=calib_uncertainties,
        resol_params=params[4:],
        resol_uncertainties=resol_uncertainties,
        resol_e_ref=resol_e_ref,
        param_cov=cov,
        calib_formula=calib_formula,
        resol_formula=resol_formula,
        param_order=param_order,
    )
