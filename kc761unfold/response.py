"""Deposition-to-channel response and its calibration-parameter derivatives.

The stored response matrix comes from the calibration file; the
systematic-uncertainty propagation needs the matrix rebuilt at the nominal
parameters *and* differentiated with respect to the calibration
parameters ``(c0..c3, b0..b2)``.  One fused numba kernel
(:func:`_deposition_to_channel_kernel`) produces the nominal dense matrix and
its seven parameter derivatives in a single pass, reusing the model
kernels of kc761calib (:func:`~kc761calib.response.gaussian_pdf_grad`,
:func:`~kc761calib.response.resol_sigma_model`,
:func:`~kc761calib.response.resol_sigma_model_grad`), so the value and
its derivatives share the exact same arithmetic and can never drift
apart.

For kc761sim composite files the primary-to-channel model is
``R(q) = C(q) p_tilde diag(eta)`` with the q-independent conditional
deposition distribution ``p_tilde`` and the zero-deposition complement
``eta`` (:func:`zero_deposition_eta`); the derivatives become
``dR/dq = (dC/dq) p_tilde diag(eta)``.  :func:`energy_geometry` is the
single source of the channel-to-energy binning (also used by the penalty
geometry in kc761unfold.penalty and the uncertainties in
kc761unfold.uncertainties).
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse

from kc761calib.response import (gaussian_pdf_grad, resol_sigma_model,
                                 resol_sigma_model_grad)

from .reader import threshold_matrix
from .types import CalibrationFile


def energy_geometry(calib_coeffs: np.ndarray, channel_low: int,
                    channel_high: int) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Energy edges/centers/widths of a channel subrange (reported cubic).

    The channel edges are the half-integer positions ch - 0.5,
    matching the calibration export's binning; returns
    ``(energy_edges, centers, widths)`` with ``n_bins + 1`` / ``n_bins``
    entries.
    """
    ch_edges = np.arange(channel_low, channel_high + 2, dtype=float) - 0.5
    e_edges = (calib_coeffs[0] + calib_coeffs[1] * ch_edges
               + calib_coeffs[2] * ch_edges ** 2
               + calib_coeffs[3] * ch_edges ** 3)
    return e_edges, 0.5 * (e_edges[:-1] + e_edges[1:]), np.diff(e_edges)


def energy_geometry_derivatives(channel_low: int, channel_high: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Exact derivatives of the center/width geometry wrt (c0..c3).

    Returns ``(d_centers, d_widths)``, each ``(4, n_bins)`` with
    ``[p] = d/dc_p``.  The bin edges are the cubic at the half-integer
    positions (bin ``k`` spanning ``k-0.5`` and ``k+0.5``), so

        d center_k/dc_p = 0.5 ((k-0.5)^p + (k+0.5)^p),
        d width_k/dc_p  = (k+0.5)^p - (k-0.5)^p.

    The same expressions are used inside :func:`_deposition_to_channel_kernel`,
    keeping the response derivatives and the penalty geometry derivatives
    consistent: the kernel receives these arrays (indexed by the local bin
    number) instead of recomputing the powers itself.
    """
    k = np.arange(channel_low, channel_high + 1, dtype=float)
    lo = k - 0.5
    hi = k + 0.5
    powers = np.arange(4, dtype=float)[:, None]
    d_centers = 0.5 * (lo ** powers + hi ** powers)
    d_widths = hi ** powers - lo ** powers
    return d_centers, d_widths


@numba.njit(parallel=True, cache=True)
def _deposition_to_channel_kernel(e_edges, resol_params, ch_lo, n_bins,
                                  d_centers, d_widths):
    """Nominal dense response R[i, j] and its parameter derivatives.

    ``e_edges`` is the full-range cubic energy binning (n_channels + 1
    entries, bin ``k`` spanning the half-integer edges ``k-0.5`` and
    ``k+0.5``); the kernel fills the ``n_bins x n_bins`` subrange
    starting at global channel index ``ch_lo``.  Returns
    ``(r_dense, g_dense)`` with ``g_dense[p] = dR/dq_p`` at the nominal
    parameters, ``p = 0..3`` the cubic coefficients and ``p = 4..6`` the
    resolution parameters.

    R[i, j] = phi(c_i - c_j; s_j) * w_i with ``phi`` the normal density,
    so (chain rule, exact)

        dR/dc_p = (dphi_dd (dc_i - dc_j) + dphi_ds ds_j/dc_p) w_i
                  + phi dw_i/dc_p,
        dR/db_k = dphi_ds (ds_j/db_k) w_i

    where the bin-geometry derivatives ``d_centers``/``d_widths``
    (``(4, n_bins)``, local bin index) come from
    :func:`energy_geometry_derivatives` -- the single source of these
    formulas, shared with the penalty geometry in kc761unfold.penalty.
    """
    centers_full = 0.5 * (e_edges[:-1] + e_edges[1:])
    widths_full = e_edges[1:] - e_edges[:-1]
    sigma_full = resol_sigma_model(resol_params, centers_full)
    ds_de_full, ds_db_full = resol_sigma_model_grad(resol_params, centers_full)
    out = np.empty((n_bins, n_bins), dtype=np.float64)
    grads = np.empty((7, n_bins, n_bins), dtype=np.float64)
    for j in numba.prange(n_bins):
        k_j = ch_lo + j
        c_j = centers_full[k_j]
        s_j = sigma_full[k_j]
        de_j = ds_de_full[k_j]
        for i in range(n_bins):
            k_i = ch_lo + i
            w_i = widths_full[k_i]
            phi, dphi_dd, dphi_ds = gaussian_pdf_grad(centers_full[k_i] - c_j,
                                                      s_j)
            out[i, j] = phi * w_i
            for p in range(4):
                grads[p, i, j] = (
                    dphi_dd * (d_centers[p, i] - d_centers[p, j]) * w_i
                    + dphi_ds * (de_j * d_centers[p, j]) * w_i
                    + phi * d_widths[p, i])
            for b in range(3):
                grads[4 + b, i, j] = dphi_ds * ds_db_full[k_j][b] * w_i
    return out, grads


def zero_deposition_eta(calib: CalibrationFile, r_dense: np.ndarray
                        ) -> np.ndarray | None:
    """Per-primary-column zero-deposition complement of the composite model.

    The stored composite response's column sums equal the detection
    efficiency, i.e. ``colsum(R)_b = eta_b * colsum(C(q0) p_tilde)_b``
    with ``eta_b = 1 - p_zero_b`` the q-independent zero-deposition
    complement; ``R(q) = C(q) p_tilde diag(eta)`` then reproduces the
    stored response exactly at the nominal parameters and carries the
    correct q-derivative without needing the Monte Carlo column totals.
    Computed once per snapshot and cached on it; ``None`` for calibration
    files (no composite step).
    """
    if calib.primary_to_deposition is None:
        return None
    if calib.zero_deposition_eta is not None:
        return calib.zero_deposition_eta
    col_model = np.asarray((r_dense @ calib.primary_to_deposition).sum(axis=0)
                           ).ravel()
    col_stored = np.asarray(calib.to_channel.sum(axis=0)).ravel()
    eta = np.divide(col_stored, col_model,
                    out=np.ones_like(col_stored),
                    where=col_model > 0.0)
    calib.zero_deposition_eta = eta
    return eta


def to_channel_and_derivatives(calib: CalibrationFile
                               ) -> tuple[sparse.csr_matrix,
                                          list[sparse.csr_matrix]]:
    """Nominal to-channel matrix and its seven parameter derivatives.

    Returns ``(r, [dr_0 .. dr_6])`` as thresholded CSR matrices on the
    snapshot's subrange, with the composite step ``R = C p_tilde
    diag(eta)`` applied where the snapshot carries a
    ``primary_to_deposition`` block (calibration files pass the rebuilt
    deposition-to-channel matrix through unchanged).  All eight matrices
    come from one kernel pass over the same geometry and resolution
    arrays.  The result depends only on the immutable snapshot and is
    cached on it (``calib.to_channel_cache``), so batch workflows that
    unfold many spectra against one calibration pay the kernel and the
    thresholdings once.
    """
    cached = calib.to_channel_cache
    if cached is not None:
        return cached
    e_edges, _, _ = energy_geometry(np.asarray(calib.calib_coeffs,
                                               dtype=float),
                                    0, calib.n_channels - 1)
    d_centers, d_widths = energy_geometry_derivatives(calib.channel_low,
                                                      calib.channel_high)
    r_dense, g_dense = _deposition_to_channel_kernel(
        e_edges, np.asarray(calib.resol_params, dtype=float),
        int(calib.channel_low), int(calib.n_bins), d_centers, d_widths)
    eta = zero_deposition_eta(calib, r_dense)

    def compose(dense: np.ndarray) -> sparse.csr_matrix:
        if calib.primary_to_deposition is not None:
            dense = (dense @ calib.primary_to_deposition) @ sparse.diags(eta)
        return threshold_matrix(dense)

    result = (compose(r_dense), [compose(g_dense[p]) for p in range(7)])
    calib.to_channel_cache = result
    return result
