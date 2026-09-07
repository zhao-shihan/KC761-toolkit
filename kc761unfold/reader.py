"""Load and validate kc761calib exports (and kc761sim composite files) for
kc761unfold.

Delegates the raw reading and geometry validation to
:func:`kc761util.calibfile.load_calib_file` (which accepts both channel
matrix object names and treats the energy axis as the axis the matrix maps
from), then applies the unfold-side policies: the dense matrix is
thresholded to a CSR matrix (relative 1e-12 per column) and undetermined
calibration parameters (NaN covariance rows) are treated as fixed.
:func:`slice_calibration` produces the working subrange view without
re-reading the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from kc761util.calibfile import load_calib_file

from .types import CalibrationFile

MATRIX_THRESHOLD = 1e-12


def _threshold(dense: np.ndarray) -> sparse.csr_matrix:
    """Threshold a dense matrix block to CSR, relative per column."""
    colmax = np.abs(dense).max(axis=0, keepdims=True)
    mask = np.abs(dense) > MATRIX_THRESHOLD * colmax
    return sparse.csr_matrix(np.where(mask, dense, 0.0))


def load_calibration(path: str | Path) -> CalibrationFile:
    """Read and validate the calibration file (full channel range)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"calibration file not found: {path}")

    data = load_calib_file(path)

    # kc761calib marks undetermined parameters with all-NaN rows/columns;
    # treat them as zero-variance (fixed) directions instead of rejecting
    # the export.
    cov = np.asarray(data.param_cov, dtype=float)
    undetermined = np.all(np.isnan(cov), axis=1)
    if undetermined.any():
        print(f"[unfold] warning: {int(undetermined.sum())} undetermined "
              f"calibration parameter(s) marked by NaN covariance; their "
              f"uncertainties are treated as zero")
        cov = np.where(np.isnan(cov), 0.0, cov)

    return CalibrationFile(
        n_channels=data.n_channels,
        channel_low=0,
        channel_high=data.n_channels - 1,
        energy_edges_full=np.asarray(data.energy_edges, dtype=float),
        energy_edges=np.asarray(data.energy_edges, dtype=float),
        centers=0.5 * (data.energy_edges[:-1] + data.energy_edges[1:]),
        widths=np.diff(data.energy_edges),
        channel_matrix=_threshold(data.channel_matrix),
        primary_to_deposition=(None if data.primary_to_deposition is None
                   else _threshold(data.primary_to_deposition)),
        calib_coeffs=data.calib_coeffs,
        resol_params=data.resol_params,
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
    dense = calib.channel_matrix.toarray()
    sub = dense[channel_low:channel_high + 1, channel_low:channel_high + 1]
    col_sums = sub.sum(axis=0)
    n_trunc = int((col_sums < 1.0 - 1e-6).sum())
    if n_trunc > 0:
        print(f"[unfold] warning: {n_trunc} matrix columns with column "
              f"sums < 1 (absolute detection efficiency, or truncation at "
              f"the detector range edges)")
    return CalibrationFile(
        n_channels=n,
        channel_low=channel_low,
        channel_high=channel_high,
        energy_edges_full=calib.energy_edges_full,
        energy_edges=calib.energy_edges_full[channel_low:channel_high + 2],
        centers=calib.centers[channel_low:channel_high + 1],
        widths=calib.widths[channel_low:channel_high + 1],
        channel_matrix=_threshold(sub),
        primary_to_deposition=(None if calib.primary_to_deposition is None
                   else _threshold(calib.primary_to_deposition[
                       channel_low:channel_high + 1,
                       channel_low:channel_high + 1].toarray())),
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
