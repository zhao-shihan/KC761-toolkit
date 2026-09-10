"""Analytic uncertainty propagation for the unfolding.

At the optimum, the estimator is the implicit function ``mu_hat(y, q)``
of the data ``y`` and the calibration parameters ``q``; linearizing the
stationarity condition gives

    dmu = 2 Ht^-1 R^T W dy - Ht^-1 (d2L / d mu d q) dq,

with ``Ht`` the embedded Hessian on the free set.  The statistical
covariance is ``C_stat = M Sigma_y M^T`` (``M = 2 Ht^-1 R^T W``); the
systematic covariance propagates the calibration parameters through the
same implicit differentiation, ``J = dmu/dq = -Ht^-1 G_q``.  The cross
gradient ``G_q = d2L / d mu d q`` is evaluated exactly at the fixed
optimum: the response derivatives ``dR/dq`` come from the same fused
kernel that rebuilds the nominal matrix
(:func:`kc761unfold.response.to_channel_and_derivatives`) and the
penalty derivatives ``dD/dq`` from
:func:`kc761unfold.penalty.penalty_operator_grad`, so the propagation
shares one source of truth with the model and carries no finite
difference truncation error.  Both covariances are computed directly on
the solved variable.  The calibration-only mode expresses the
calibration uncertainty vertically through the spectrum derivative
instead.
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse

from kc761calib.response import resol_sigma_model

from .penalty import penalty_operator_grad
from .response import deposition_to_channel, energy_geometry_derivatives
from .solver import UnfoldProblem, embed_and_factor, solve_embedded
from .types import CalibrationFile

_RNG = np.random.default_rng(20240517)


def _hat_trace(prob: UnfoldProblem, ab, u, free,
               n_probes: int = 30, factor=None) -> float:
    """Hutchinson estimate of ``tr(2 H^-1 R^T W R)`` on the free set.

    ``prob`` carries the current ``r``/``w`` and its ``alpha`` through
    the banded Hessian ``(ab, u)``; each probe solves the exact banded
    system on the embedded free set.
    """
    r = prob.r
    w = prob.w
    n_free = int(np.sum(free))
    if n_free == 0:
        return 0.0
    if factor is None:
        factor = embed_and_factor(ab, u, free)
    trace = 0.0
    for _ in range(n_probes):
        z = np.where(_RNG.random(prob.n) < 0.5, -1.0, 1.0)
        z[~free] = 0.0
        rz = r @ z
        rhs = 2.0 * (r.T @ (w * rz))  # 2 R^T W R z
        sol = solve_embedded(ab, u, free, rhs, factor=factor)
        trace += float(z @ sol)
    return trace / n_probes


def effective_ndof(prob: UnfoldProblem, ab, u, free, n_data: int,
                   n_probes: int = 30, factor=None) -> int:
    """Effective residual degrees of freedom of the constrained fit.

    ``n_data - tr(A)`` with ``A = 2 H^-1 R^T W R`` the hat matrix (the
    trace is the effective number of fitted parameters, restricted to
    the solved free set) and ``n_data`` the number of positively
    weighted data bins.  Using the free-bin count instead of the data
    count understates the dof badly (bound bins are not data
    observations).
    """
    trace = _hat_trace(prob, ab, u, free, n_probes, factor)
    return max(0, int(round(float(n_data) - trace)))


def _cross_gradient(prob: UnfoldProblem, mu: np.ndarray, rtw: sparse.csr_matrix,
                    calib: CalibrationFile, sigma: np.ndarray,
                    mask: np.ndarray, k: int,
                    derivatives: list[sparse.csr_matrix]) -> np.ndarray:
    """Exact cross gradients G_q[:, p] = d2L/d mu d q_p at the fixed optimum.

    The chi2 part follows from the response derivative
    ``dR_p = dR/dq_p`` (the full composed model, so ``p_tilde`` and the
    detection efficiency ``eta`` are already folded in):

        d/dq_p [-2 R^T W (y - R mu)] =
            -2 dR_p^T W (y - R mu) + 2 R^T W (dR_p mu),

    with ``rtw = R^T diag(w)`` the weight-conjugated response, shared
    with the statistical-covariance solves.  The penalty part
    ``2 a D^T D mu`` moves only with the calibration coefficients (the
    resolution parameters do not enter the penalty geometry), through
    ``dD_p`` of :func:`penalty_operator_grad`.
    """
    w_resid = prob.w * (prob.y - prob.r @ mu)
    g_q = np.empty((prob.n, 7))
    for p in range(7):
        dr = derivatives[p]
        g_q[:, p] = (-2.0 * (dr.T @ w_resid)
                     + 2.0 * (rtw @ (dr @ mu)))

    d_centers, d_widths = energy_geometry_derivatives(calib.channel_low,
                                                      calib.channel_high)
    d_mu = prob.d_op @ mu
    for p in range(4):
        # p = 0 (c0, a pure energy offset) leaves all bin-center
        # differences and widths unchanged, so dD/dc0 comes out exactly
        # zero -- the loop keeps the four coefficients uniform instead of
        # special-casing the translation invariance.
        dd = penalty_operator_grad(calib.widths, calib.centers, sigma, mask,
                                   k, d_centers[p], d_widths[p])
        if prob.d_op.shape[0] != dd.shape[0]:
            # The pad anchor rows appended to D are constant (no
            # parameter dependence), so their gradient rows are zero.
            extra = prob.d_op.shape[0] - dd.shape[0]
            dd = sparse.vstack(
                [dd, sparse.csr_matrix((extra, prob.n))]).tocsr()
        g_q[:, p] += 2.0 * prob.alpha * (dd.T @ d_mu
                                         + prob.d_op.T @ (dd @ mu))
    return g_q


def _sim_mc_diagonal(prob: UnfoldProblem, mu: np.ndarray, free: np.ndarray,
                   calib: CalibrationFile, sim, factor) -> np.ndarray:
    """Per-bin Monte Carlo variance of R, propagated per response column.

    The composed response's Monte Carlo (multinomial) column covariance
    is ``Cov(R_·b) = (C diag(p_·b) C^T - R_·b R_·b^T) / totals_b`` with
    ``p = G/totals`` the per-primary-event deposition distribution
    (zero-deposition included through the totals) and C the analytic
    deposition-to-channel matrix.  Only the *diagonal* of the propagated
    covariance is computed: with the column adjoint

        G_b = dmu/dR_·b = 2 w_b r_b H^-1 - mu_b M,   M = 2 H^-1 R^T W,

    the per-bin variance contribution is the row-square of
    ``G_b C diag(sqrt(p_·b))`` minus ``(G_b R_·b)^2``, summed over the
    columns with the multinomial 1/totals_b.  The row-squares of the
    b-dependent coefficient matrix are expanded as
    ``a_b^2 P^2 + 2 a_b c_b P Q + c_b^2 Q^2`` with ``P = H^-1 C`` and
    ``Q = M C`` (precomputed once), so each column costs three
    matrix-vector products -- O(n^3) in total instead of the O(n^4) of
    the full column-mode covariance.
    """
    if sim is None:
        return np.zeros(prob.n)
    n = prob.n
    c = deposition_to_channel(calib)
    c_dense = np.asarray(c.toarray(), dtype=float)
    g = np.asarray(sim.counts, dtype=float)
    eta = np.asarray(sim.efficiency, dtype=float)
    col_sums = g.sum(axis=0)
    totals = np.divide(col_sums, eta,
                       out=np.zeros(n), where=eta > 0.0)
    p = np.divide(g, totals[None, :],
                  out=np.zeros_like(g), where=totals > 0.0)

    ab, u = prob.hessian_banded()
    if factor is None:
        factor = embed_and_factor(ab, u, free)
    rtw = (prob.r.T @ sparse.diags(prob.w)).tocsr()
    p_mat = solve_embedded(ab, u, free, c_dense, factor=factor)
    q_mat = solve_embedded(ab, u, free, 2.0 * (rtw @ c_dense), factor=factor)
    p2 = p_mat * p_mat
    pq = p_mat * q_mat
    q2 = q_mat * q_mat
    w_resid = prob.w * (prob.y - prob.r @ mu)
    # Blocked accumulation: the per-column row-squares are three
    # matrix-vector products, so stacking a block of columns turns them
    # into three BLAS-threaded GEMMs (and the v terms into two more).
    target_bytes = 1 << 26  # ~64 MB per block
    chunk = max(1, min(n, target_bytes // max(1, n * 8)))
    diag = np.zeros(n)
    for b0 in range(0, n, chunk):
        b1 = min(n, b0 + chunk)
        cols = p[:, b0:b1]
        t_blk = totals[b0:b1]
        a = 2.0 * w_resid[b0:b1]
        c = -mu[b0:b1]
        row_sq = (p2 @ cols) * (a * a)[None, :] \
            + (pq @ cols) * (2.0 * a * c)[None, :] \
            + (q2 @ cols) * (c * c)[None, :]
        v = (p_mat @ cols) * a[None, :] + (q_mat @ cols) * c[None, :]
        good = t_blk > 0.0
        if good.any():
            diag += ((row_sq - v * v)[:, good] / t_blk[good][None, :]).sum(
                axis=1)
    return np.maximum(diag, 0.0)


def compute_covariances(prob: UnfoldProblem, mu: np.ndarray, free: np.ndarray,
                        calib: CalibrationFile, sim, sigma: np.ndarray,
                        mask: np.ndarray, k: int,
                        derivatives: list[sparse.csr_matrix],
                        ab=None, u=None, factor=None,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin statistical and systematic uncertainties at the optimum.

    Only the diagonals are computed (the full covariance matrices are no
    longer produced or exported): ``sigma_stat`` from
    ``diag(M Sigma_y M^T)`` with ``M = 2 Ht^-1 R^T W``, ``sigma_syst``
    from the calibration-parameter quadratic form ``diag(J Sigma_q
    J^T)`` plus the simulation-file Monte Carlo column diagonal
    (:func:`_sim_mc_diagonal`).  Bins pinned at ``mu = 0`` carry zero
    rows in the embedded solves and report ``0 +/- 0`` (marked in the
    report).
    """
    if ab is None:
        ab, u = prob.hessian_banded()
    if factor is None:
        factor = embed_and_factor(ab, u, free)
    rtw = (prob.r.T @ sparse.diags(prob.w)).tocsr()

    # statistical: M = 2 Ht^-1 (R_clip^T W); diag(M Sigma_y M^T) with
    # Sigma_y = diag(1/w) is the row-square sum of M/sqrt(w).  Data bins
    # with zero weight (the window padding) contribute nothing and must
    # not divide by zero.
    rt_w = (sparse.diags(free.astype(float)) @ rtw).toarray()
    m = 2.0 * solve_embedded(ab, u, free, rt_w, factor=factor)
    sw = np.sqrt(prob.w)
    m_scaled = np.zeros_like(m)
    np.divide(m, sw[None, :], out=m_scaled, where=sw[None, :] > 0.0)
    diag_stat = np.maximum(np.sum(m_scaled * m_scaled, axis=1), 0.0)

    # systematic: J = -Ht^-1 G_q; diag(J Sigma_q J^T) per row.
    g_q = _cross_gradient(prob, mu, rtw, calib, sigma, mask, k, derivatives)
    j = -solve_embedded(ab, u, free, g_q, factor=factor)
    diag_syst = np.einsum("ip,pq,iq->i", j, calib.param_cov, j)
    diag_syst = np.maximum(diag_syst, 0.0)

    # simulation-file Monte Carlo column diagonal.
    diag_syst = diag_syst + _sim_mc_diagonal(prob, mu, free, calib, sim,
                                           factor)

    return np.sqrt(diag_stat), np.sqrt(diag_syst)

@numba.njit(cache=True)
def _resolution_smooth(counts: np.ndarray, s_ch: np.ndarray) -> np.ndarray:
    """Per-bin Gaussian smoothing of ``counts`` with local widths ``s_ch``.

    Each bin is convolved with a Gaussian of width ``s_ch[j]`` (in
    channels), truncated at +/- 6 sigma, weighted average normalized per
    bin.  Compiled version of the pure-Python loop; ``n >= 1`` required.
    """
    n = counts.shape[0]
    smooth = np.empty(n, dtype=np.float64)
    for j in range(n):
        lo = max(0, int(j - 6.0 * s_ch[j]))
        hi = min(n, int(j + 6.0 * s_ch[j]) + 1)
        s_j = max(s_ch[j], 1e-6)
        num = 0.0
        den = 0.0
        for k in range(lo, hi):
            g = np.exp(-0.5 * ((k - j) / s_j) ** 2)
            num += g * counts[k]
            den += g
        smooth[j] = num / max(den, 1e-30)
    return smooth


def calibration_vertical_term(counts: np.ndarray, sigma_E: np.ndarray,
                              centers: np.ndarray, resol_params: np.ndarray
                              ) -> np.ndarray:
    """Calibration uncertainty expressed vertically: |dy/dE| * sigma_E.

    The counts are smoothed with the local resolution before the central
    difference, so statistical noise does not blow up the derivative.
    """
    n = len(counts)
    if n < 2:
        return np.zeros(n)
    s_keV = resol_sigma_model(np.asarray(resol_params, dtype=float), centers)
    h = np.diff(centers)
    s_ch = s_keV / np.maximum(np.concatenate([h, h[-1:]]), 1e-9)

    # resolution-smoothed counts (per-bin Gaussian kernel, truncated)
    smooth = _resolution_smooth(np.ascontiguousarray(counts, dtype=float),
                                np.ascontiguousarray(s_ch))

    dydE = np.empty(n)
    denom = centers[2:] - centers[:-2]
    dydE[1:-1] = (smooth[2:] - smooth[:-2]) / denom
    dydE[0] = (smooth[1] - smooth[0]) / max(h[0], 1e-30)
    dydE[-1] = (smooth[-1] - smooth[-2]) / max(h[-1], 1e-30)
    return np.abs(dydE) * sigma_E


def energy_center_uncertainties(centers_ch: np.ndarray, param_cov: np.ndarray
                                ) -> np.ndarray:
    """1-sigma energy-center uncertainties from the calibration block."""
    g = np.stack([np.ones_like(centers_ch), centers_ch,
                  centers_ch ** 2, centers_ch ** 3], axis=1)
    var = np.einsum("ij,jk,ik->i", g, param_cov[:4, :4], g)
    return np.sqrt(np.maximum(var, 0.0))
