"""Load and validate kc761calib ROOT exports for kc761unfold.

Reads the response matrix (TH2D ``response_matrix``), the reported
parameters (c0..c3, b0..b2) with their 7x7 covariance (``param_cov``),
then validates the geometry assumptions the unfolding relies on: square
matrix, uniform channel bins of width 1, a strictly increasing energy
binning, finite parameters/covariance.  The dense response is
thresholded to a CSR matrix (relative 1e-12 per column), which is
banded in practice because the Gaussian kernel decays super-fast.
:func:`slice_calibration` produces the working subrange view without
re-reading the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot
from scipy import sparse

from .types import CalibrationFile

RESPONSE_HIST_NAME = "response_matrix"
MATRIX_THRESHOLD = 1e-12

_PARAM_NAMES = ["c0", "c1", "c2", "c3", "b0", "b1", "b2"]


def _threshold(dense: np.ndarray) -> sparse.csr_matrix:
    """Threshold a dense response block to CSR, relative per column."""
    colmax = np.abs(dense).max(axis=0, keepdims=True)
    mask = np.abs(dense) > MATRIX_THRESHOLD * colmax
    return sparse.csr_matrix(np.where(mask, dense, 0.0))


def load_calibration(path: str | Path) -> CalibrationFile:
    """Read and validate the calibration file (full channel range)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"calibration file not found: {path}")

    with uproot.open(path) as f:
        if RESPONSE_HIST_NAME not in f:
            raise ValueError(
                f"{path} does not contain the TH2D {RESPONSE_HIST_NAME!r}; "
                "is it a kc761calib export?")
        hist = f[RESPONSE_HIST_NAME]
        channel_edges = np.asarray(hist.axis(0).edges(), dtype=float)
        energy_edges_full = np.asarray(hist.axis(1).edges(), dtype=float)
        # Validate the file-provided bin counts BEFORE materializing the
        # dense contents (the same 2^14 cap unfold2root.cxx enforces).
        if (len(channel_edges) != len(energy_edges_full)
                or len(channel_edges) < 2 or len(channel_edges) > (1 << 14) + 1):
            raise ValueError(
                f"response matrix bin counts {len(channel_edges) - 1} x "
                f"{len(energy_edges_full) - 1} are inconsistent or beyond "
                f"the supported maximum of {1 << 14}")
        values = np.asarray(hist.values(), dtype=float)  # R[ch, E]
        try:
            params = np.array([f[name].member("fVal") for name in _PARAM_NAMES],
                              dtype=float)
            cov_up = np.array(f["param_cov"].member("fElements")[:28],
                              dtype=float)
        except KeyError as exc:
            raise ValueError(
                f"{path} is missing the expected calibration objects "
                f"({exc.args[0]}); is it a kc761calib export?") from exc

    n = values.shape[0]
    if values.shape[1] != n:
        raise ValueError(f"response matrix must be square, got {values.shape}")
    if len(channel_edges) != n + 1 or len(energy_edges_full) != n + 1:
        raise ValueError("response matrix axis lengths inconsistent with the "
                         "matrix dimensions")
    if not np.allclose(np.diff(channel_edges), 1.0, atol=1e-9):
        raise ValueError("channel bins of the response matrix must be "
                         "uniform with width 1")
    if np.any(np.diff(energy_edges_full) <= 0.0):
        raise ValueError("energy binning of the response matrix is not "
                         "strictly increasing")
    if not np.all(np.isfinite(params)):
        raise ValueError("calibration parameters contain non-finite entries")

    cov = np.zeros((7, 7))
    cov[np.triu_indices(7)] = cov_up
    cov = cov + np.triu(cov, 1).T
    # kc761calib marks undetermined parameters with all-NaN rows/columns;
    # treat them as zero-variance (fixed) directions instead of rejecting
    # the export.
    undetermined = np.all(np.isnan(cov), axis=1)
    if undetermined.any():
        print(f"[unfold] warning: {int(undetermined.sum())} undetermined "
              f"calibration parameter(s) marked by NaN covariance; their "
              f"uncertainties are treated as zero")
        cov = np.where(np.isnan(cov), 0.0, cov)

    return CalibrationFile(
        n_channels=n,
        channel_low=0,
        channel_high=n - 1,
        energy_edges_full=np.asarray(energy_edges_full, dtype=float),
        energy_edges=np.asarray(energy_edges_full, dtype=float),
        centers=0.5 * (energy_edges_full[:-1] + energy_edges_full[1:]),
        widths=np.diff(energy_edges_full),
        matrix=_threshold(values),
        calib_coeffs=params[:4],
        resol_params=params[4:],
        param_cov=cov,
    )


def slice_calibration(calib: CalibrationFile, channel_low: int,
                      channel_high: int) -> CalibrationFile:
    """Restrict a full-range calibration to a channel subrange."""
    n = calib.n_channels
    if channel_low < 0 or channel_high >= n or channel_low > channel_high:
        raise ValueError(
            f"channel range [{channel_low}, {channel_high}] must satisfy "
            f"0 <= chlo <= chhi < {n}")
    if channel_low == calib.channel_low and channel_high == calib.channel_high:
        return calib
    dense = calib.matrix.toarray()
    sub = dense[channel_low:channel_high + 1, channel_low:channel_high + 1]
    col_sums = sub.sum(axis=0)
    n_trunc = int((col_sums < 1.0 - 1e-6).sum())
    if n_trunc > 0:
        print(f"[unfold] warning: {n_trunc} response columns truncated at "
              f"the detector range edges (column sums < 1)")
    return CalibrationFile(
        n_channels=n,
        channel_low=channel_low,
        channel_high=channel_high,
        energy_edges_full=calib.energy_edges_full,
        energy_edges=calib.energy_edges_full[channel_low:channel_high + 2],
        centers=calib.centers[channel_low:channel_high + 1],
        widths=calib.widths[channel_low:channel_high + 1],
        matrix=_threshold(sub),
        calib_coeffs=calib.calib_coeffs,
        resol_params=calib.resol_params,
        param_cov=calib.param_cov,
    )


def energy_to_channels(calib: CalibrationFile, energy_low: float,
                       energy_high: float) -> tuple[int, int]:
    """Map an energy window to the channel bins whose centers fall inside.

    Returns ``(channel_low, channel_high)`` (0-based, inclusive).  The
    bins are selected by their center energy: the first bin with center
    >= ``energy_low`` through the last with center <= ``energy_high``.
    """
    if energy_low > energy_high:
        raise ValueError(
            f"energy range [{energy_low}, {energy_high}] must satisfy "
            f"elo <= ehi (keV)")
    centers = calib.centers
    ch_lo = int(np.searchsorted(centers, energy_low, side="left"))
    ch_hi = int(np.searchsorted(centers, energy_high, side="right")) - 1
    if ch_lo > ch_hi:
        raise ValueError(
            f"energy range [{energy_low:g}, {energy_high:g}] keV contains no "
            f"channel-bin centers (available: [{centers[0]:.2f}, "
            f"{centers[-1]:.2f}] keV)")
    return max(0, ch_lo), min(calib.n_channels - 1, ch_hi)
