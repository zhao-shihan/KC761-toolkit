"""Analytic Jacobian of the deposition response matrix w.r.t. stored parameters.

The composite pipeline (kc761sim matrix modes) propagates the calibration
fit uncertainty through R = C @ G by the full covariance linearization

    Cov_C(C[a,j], C[a',k]) = J[a,j]^T param_cov J[a',k],

so it needs the per-element gradient J[a,j,p] = dC[a,j]/d q_p in the
*reported* basis q = (c0, c1, c2, c3, b0, b1, b2) -- the same basis the
stored ``param_cov`` uses.  The gradient is assembled from the stored
parameters alone; unlike the fit's internal (c0, k1, k2, k3)
parameterization, the reported-basis expression needs no ``channel_max``.

C[a,j] = gaussian_pdf(c_a - c_j; sigma_j) * width_a, with c_k and
width_k the midpoint and width of energy bin k of the *stored* energy
edges, so the Jacobian is evaluated exactly at the file's matrix.  The
stored parameters enter only through the edges (the calibration image
E(ch +- 0.5)); the derivatives of the midpoint and width of bin k are

    dc_k/dc_p   = 0.5 * [(ch_k+0.5)^p + (ch_k-0.5)^p],
    dw_k/dc_p   = (ch_k+0.5)^p - (ch_k-0.5)^p,

so with d = c_a - c_j, dg_dd = -d/sigma_j^2 * pdf,
dg_ds = (d^2/sigma_j^2 - 1)/sigma_j * pdf, and the sigma derivatives from
:func:`kc761calib.response.resol_sigma_model_grad`:

    dC[a,j]/dc_p = dg_dd * (dc_a/dc_p - dc_j/dc_p) * width_a
                    + pdf * dw_a/dc_p
                    + dg_ds * ds_dE_j * dc_j/dc_p * width_a,
    dC[a,j]/db_p = dg_ds * ds_db[j,p] * width_a,

(both clamps of sigma(E) carry exact one-sided derivatives through
:func:`kc761calib.response.resol_sigma_model_grad`).

Note on the stored per-element uncertainties: kc761calib computes them in
its internal basis; the reported-basis quadratic form here is the same
value up to floating-point rounding (a constant, invertible change of
basis), so Var_C agrees with the file's ``C_unc^2`` to ~1e-16 relative.
The C copy in the composite output inherits the file's uncertainties
bitwise; only the propagated R uncertainties use this recomputed
Jacobian.

Memory: the returned tensor has shape (n, n, 7); for the 2048-bin
calibration that is ~235 MB of float64, acceptable for a one-shot
composition.
"""

from __future__ import annotations

import numba
import numpy as np

from .response import gaussian_pdf, resol_sigma_model, resol_sigma_model_grad

N_PARAMS = 7  # reported basis (c0..c3, b0..b2)


@numba.njit(parallel=True, cache=True)
def _jacobian_kernel(ch, centers, widths, sigma, ds_dE, ds_db):
    """Per-element gradient of C w.r.t. the reported basis (module docstring).

    Column-parallel: each ``(i, j)`` entry is written exactly once.  The
    row-dependent calibration derivatives ``dm``/``dw`` are precomputed
    once per row (they would otherwise be recomputed per column).
    """
    n = centers.shape[0]
    dm = np.empty((n, 4), dtype=np.float64)  # d(center_i)/dc_p
    dw = np.empty((n, 4), dtype=np.float64)  # d(width_i)/dc_p
    for i in range(n):
        ch_i = ch[i]
        for p in range(4):
            dm[i, p] = 0.5 * ((ch_i + 0.5) ** p + (ch_i - 0.5) ** p)
            dw[i, p] = (ch_i + 0.5) ** p - (ch_i - 0.5) ** p
    jac = np.empty((n, n, N_PARAMS), dtype=np.float64)
    for j in numba.prange(n):
        c_j = centers[j]
        s_j = sigma[j]
        s2 = s_j * s_j
        ds_dE_j = ds_dE[j]
        ch_j = ch[j]
        dm_j = np.empty(4, dtype=np.float64)
        for p in range(4):
            dm_j[p] = 0.5 * ((ch_j + 0.5) ** p + (ch_j - 0.5) ** p)
        for i in range(n):
            d = centers[i] - c_j
            pdf = gaussian_pdf(d, s_j)
            dg_dd = -d / s2 * pdf
            dg_ds = pdf * (d * d / s2 - 1.0) / s_j
            width_i = widths[i]
            for p in range(4):
                jac[i, j, p] = (
                    (dg_dd * (dm[i, p] - dm_j[p]) + dg_ds * ds_dE_j * dm_j[p])
                    * width_i
                    + pdf * dw[i, p]
                )
            for p in range(4, 7):
                jac[i, j, p] = dg_ds * ds_db[j, p - 4] * width_i
    return jac


def build_matrix_jacobian(energy_edges, calib_coeffs,
                          resol_params) -> np.ndarray:
    """(n, n, 7) gradient of C w.r.t. (c0..c3, b0..b2).

    ``energy_edges`` are the stored deposition-energy edges (keV, length
    n + 1, strictly increasing); ``calib_coeffs`` = (c0..c3) and
    ``resol_params`` = (b0..b2) are the stored parameters.  The returned
    tensor is J[i, j, p] = dC[i, j]/dq_p in the reported basis.
    """
    edges = np.asarray(energy_edges, dtype=float)
    calib = np.asarray(calib_coeffs, dtype=float)
    resol = np.asarray(resol_params, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"energy_edges must be a 1-D array with >= 2 "
                         f"entries, got shape {edges.shape}")
    if calib.shape != (4,):
        raise ValueError(f"calib_coeffs must have shape (4,), got "
                         f"{calib.shape}")
    if resol.shape != (3,):
        raise ValueError(f"resol_params must have shape (3,), got "
                         f"{resol.shape}")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges are not strictly increasing")

    n = edges.size - 1
    # The file's matrix was built from these exact edges (midpoint
    # quadrature nodes of the Gaussian kernel), so evaluating the kernels
    # at the stored edges reproduces the file's matrix up to fp rounding.
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ch = np.arange(n, dtype=np.float64)
    sigma = resol_sigma_model(resol, centers)
    ds_dE, ds_db = resol_sigma_model_grad(resol, centers)
    return _jacobian_kernel(ch, centers, widths, sigma, ds_dE, ds_db)
