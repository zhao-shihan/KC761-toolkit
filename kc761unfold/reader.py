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

Both :func:`load_calibration` and :func:`slice_calibration` are cached
session-locally (the calibration snapshot is immutable), so batch
workflows that reuse one file or several working windows pay the file
read and the matrix conversions only once.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numba
import numpy as np
from scipy import sparse

from kc761util.calibfile import load_calib_file

from .types import CalibrationFile

MATRIX_THRESHOLD = 1e-12


@numba.njit(cache=True)
def _threshold_entries(dense: np.ndarray, colmax: np.ndarray,
                       threshold: float) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
    """Column-relative mask entries of a dense block, gathered in one pass."""
    n_rows, n_cols = dense.shape
    count = 0
    for j in range(n_cols):
        for i in range(n_rows):
            if abs(dense[i, j]) > threshold * colmax[j]:
                count += 1
    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    data = np.empty(count, dtype=np.float64)
    k = 0
    for j in range(n_cols):
        for i in range(n_rows):
            v = dense[i, j]
            if abs(v) > threshold * colmax[j]:
                rows[k] = i
                cols[k] = j
                data[k] = v
                k += 1
    return data, rows, cols


def threshold_matrix(dense: np.ndarray) -> sparse.csr_matrix:
    """Threshold a dense matrix block to CSR, relative per column.

    The surviving entries are gathered from the mask indices directly, so
    no full-size ``np.where`` copy of the dense block is materialized.
    """
    dense = np.ascontiguousarray(dense, dtype=float)
    colmax = np.abs(dense).max(axis=0)
    data, rows, cols = _threshold_entries(dense, colmax, MATRIX_THRESHOLD)
    return sparse.csr_matrix((data, (rows, cols)), shape=dense.shape)


def _load_calibration_uncached(path: str) -> CalibrationFile:
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
        to_channel=threshold_matrix(data.to_channel),
        primary_to_deposition=(None if data.primary_to_deposition is None
                               else threshold_matrix(data.primary_to_deposition)),
        calib_coeffs=data.calib_coeffs,
        resol_params=data.resol_params,
        param_cov=cov,
    )


def load_calibration(path: str | Path) -> CalibrationFile:
    """Read and validate the calibration file, cached per file identity.

    The session-local cache keys on ``(path, mtime_ns, ctime_ns, size)``:
    an in-place rewrite changes the ctime even when the size and mtime
    survive (cp -p, rsync -a), so the stale snapshot cannot be served for
    the rest of the session.  The returned :class:`CalibrationFile` is an
    immutable snapshot, so identical re-reads (batch workflows over many
    data files) skip the uproot decompression and the matrix conversions
    entirely.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"calibration file not found: {path}")
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_ctime_ns,
                        stat.st_size)


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int, ctime_ns: int,
                 size: int) -> CalibrationFile:
    return _load_calibration_uncached(path)


def slice_calibration(calib: CalibrationFile, channel_low: int,
                      channel_high: int) -> CalibrationFile:
    """Restrict a full-range calibration to a channel subrange, cached.

    The slice is cached on the source snapshot, so repeated window
    selections (a per-configuration loop in a batch script) reuse the
    same subrange objects.  The CSR blocks are sliced directly without
    densifying the full-channel matrix first.
    """
    n = calib.n_channels
    if channel_low < 0 or channel_high >= n or channel_low > channel_high:
        raise ValueError(
            f"channel range [{channel_low}, {channel_high}] must satisfy "
            f"0 <= chlo <= chhi < {n}")
    if channel_low == calib.channel_low and channel_high == calib.channel_high:
        return calib
    cached = calib.slice_cache.get((channel_low, channel_high))
    if cached is not None:
        return cached
    sub = calib.to_channel[channel_low:channel_high + 1,
                           channel_low:channel_high + 1].toarray()
    col_sums = sub.sum(axis=0)
    n_trunc = int((col_sums < 1.0 - 1e-6).sum())
    if n_trunc > 0:
        print(f"[unfold] warning: {n_trunc} matrix columns with column "
              f"sums < 1 (total detection efficiency, or truncation at "
              f"the detector range edges)")
    if calib.primary_to_deposition is not None:
        p_sub = calib.primary_to_deposition[
            channel_low:channel_high + 1,
            channel_low:channel_high + 1].toarray()
    else:
        p_sub = None
    result = CalibrationFile(
        n_channels=n,
        channel_low=channel_low,
        channel_high=channel_high,
        energy_edges_full=calib.energy_edges_full,
        energy_edges=calib.energy_edges_full[channel_low:channel_high + 2],
        centers=calib.centers[channel_low:channel_high + 1],
        widths=calib.widths[channel_low:channel_high + 1],
        to_channel=threshold_matrix(sub),
        primary_to_deposition=(
            None if p_sub is None else threshold_matrix(p_sub)),
        calib_coeffs=calib.calib_coeffs,
        resol_params=calib.resol_params,
        param_cov=calib.param_cov,
    )
    calib.slice_cache[(channel_low, channel_high)] = result
    return result


def energy_to_channels(calib: CalibrationFile, energy_low: float,
                       energy_high: float) -> tuple[int, int]:
    """Map an energy window to the channels whose centers fall inside.

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
