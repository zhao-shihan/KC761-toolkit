"""Response-matrix rebuild for the systematic-error propagation.

The nominal response comes from the calibration file (kc761unfold.reader);
the propagation of the calibration-parameter uncertainty needs the
response at perturbed parameters ``(c0..c3, b0..b2)``.  The rebuild
reuses the same model kernels as kc761calib (cubic calibration,
Bernstein resolution, Gaussian smearing, midpoint quadrature), so the
perturbed matrices are exactly consistent with the stored one at the
nominal point.  :func:`energy_geometry` is the single source of the
channel-to-energy binning (also used by the finite-difference geometry
in kc761unfold.errors).
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse

from kc761calib.response import gaussian_pdf, resol_sigma_model

from .reader import MATRIX_THRESHOLD


def energy_geometry(calib_coeffs: np.ndarray, channel_low: int,
                    channel_high: int) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Energy edges/centers/widths of a channel subrange (reported cubic).

    The channel bin edges are the half-integer positions ch - 0.5,
    matching the calibration export's binning; returns
    ``(energy_edges, centers, widths)`` with ``n_bins + 1`` / ``n_bins``
    entries.
    """
    ch_edges = np.arange(channel_low, channel_high + 2, dtype=float) - 0.5
    e_edges = (calib_coeffs[0] + calib_coeffs[1] * ch_edges
               + calib_coeffs[2] * ch_edges ** 2
               + calib_coeffs[3] * ch_edges ** 3)
    return e_edges, 0.5 * (e_edges[:-1] + e_edges[1:]), np.diff(e_edges)


@numba.njit(parallel=True, cache=True)
def _rebuild_dense(e_edges, resol_params, ch_lo, n_bins):
    """Dense subrange response R[i, j] = gauss(c_i - c_j; sigma_j) * dE_i."""
    centers_full = 0.5 * (e_edges[:-1] + e_edges[1:])
    widths_full = e_edges[1:] - e_edges[:-1]
    sigma_full = resol_sigma_model(resol_params, centers_full)
    out = np.empty((n_bins, n_bins), dtype=np.float64)
    for j in numba.prange(n_bins):
        c_j = centers_full[ch_lo + j]
        s_j = sigma_full[ch_lo + j]
        for i in range(n_bins):
            c_i = centers_full[ch_lo + i]
            out[i, j] = gaussian_pdf(c_i - c_j, s_j) * widths_full[ch_lo + i]
    return out


def rebuild_response(calib_coeffs: np.ndarray, resol_params: np.ndarray,
                     n_channels: int, channel_low: int,
                     channel_high: int) -> sparse.csr_matrix:
    """Rebuild the thresholded subrange response at perturbed parameters."""
    n_bins = channel_high - channel_low + 1
    e_edges, _, _ = energy_geometry(np.asarray(calib_coeffs, dtype=float),
                                    0, n_channels - 1)
    dense = _rebuild_dense(e_edges, np.asarray(resol_params, dtype=float),
                           int(channel_low), int(n_bins))
    colmax = np.abs(dense).max(axis=0, keepdims=True)
    mask = np.abs(dense) > MATRIX_THRESHOLD * colmax
    return sparse.csr_matrix(np.where(mask, dense, 0.0))
