"""Analytic error propagation for the SWR unfolding.

Statistical covariance
----------------------
At the optimum, the estimator is the implicit function mu_hat(y, q) of
the data y and the calibration parameters q.  Linearizing the
stationarity condition gives

    dmu = 2 Ht^-1 R^T W dy - Ht^-1 (d2L / d mu d q) dq,

where Ht is the embedded Hessian of L on the free set (mu_j > 0).  The
statistical covariance is C_stat = M Sigma_y M^T with M = 2 Ht^-1 R^T W
and Sigma_y the per-bin data variance (subtraction statistics plus the
fractional systematic).  The estimator's nonlinearity through the
Huber weights and the positivity constraint is second-order: repeated
unfolding of resampled data gave an ensemble-spread/analytic ratio of
1.000 [0.944, 1.036].

Systematic covariance
---------------------
The calibration-parameter uncertainty propagates through the same
implicit differentiation: J = dmu/dq = -Ht^-1 G_q with
G_q = d2L/d mu d q evaluated by central finite differences of the
gradient at the fixed optimum (7 parameters; the response and the
penalty operators are rebuilt at the perturbed parameters, which
captures their q-dependence exactly to the FD accuracy).  The
systematic covariance is C_sys = J Sigma_q J^T with Sigma_q the 7x7
reported-basis parameter covariance of the calibration file; repeated
unfolding under calibration-parameter resampling gave an
ensemble-spread/analytic ratio of 0.974.  The two sources are
independent and added in quadrature.

Calibration-only vertical term
------------------------------
With no estimator to differentiate, the calibration error of the
relabeled spectrum is expressed vertically through the spectrum
derivative: sigma_calib,j = |dy/dE|_j sigma_E,j, where the derivative
is taken on resolution-smoothed counts (noise suppression) and
sigma_E,j is the energy-center uncertainty from the calibration block
of the parameter covariance.  The same term appears as part of the
systematic band of the calibrated-spectrum layer in unfold plots.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kc761calib.response import resol_sigma_model

from .penalty import swr_operator
from .response import energy_geometry, rebuild_response
from .solver import SWRProblem, embed_and_factor, solve_embedded, swr_gradient
from .types import CalibrationFile

FD_REL = 1e-5


def _response_gradient_fd(prob: SWRProblem, mu: np.ndarray,
                          calib: CalibrationFile, sigma: np.ndarray,
                          mask: np.ndarray, k: int) -> np.ndarray:
    """FD cross gradients G_q[:, p] = d2L/d mu d q_p at the fixed optimum."""
    q0 = np.concatenate([calib.calib_coeffs, calib.resol_params])
    g_q = np.empty((prob.n, 7))
    for p in range(7):
        step_p = FD_REL * max(abs(q0[p]), 1e-12)
        grads = []
        for sgn in (+1.0, -1.0):
            qp = q0.copy()
            qp[p] += sgn * step_p
            rp = rebuild_response(qp[:4], qp[4:], calib.n_channels,
                                  calib.channel_low, calib.channel_high)
            _, centers_p, widths_p = energy_geometry(
                qp[:4], calib.channel_low, calib.channel_high)
            d_p = swr_operator(widths_p, centers_p, sigma, mask, k)
            # the FULL SWR gradient at the fixed optimum (Huber clip
            # included), evaluated on the perturbed operators.
            grads.append(swr_gradient(rp, prob.y, prob.w, d_p, prob.alpha,
                                      prob.delta, mu))
        g_q[:, p] = (grads[0] - grads[1]) / (2.0 * step_p)
    return g_q


def compute_covariances(prob: SWRProblem, mu: np.ndarray, free: np.ndarray,
                        sigma2: np.ndarray, calib: CalibrationFile,
                        sigma: np.ndarray, mask: np.ndarray,
                        k: int) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """Statistical and systematic covariance matrices at the optimum.

    Returns ``(C_stat, C_sys, sigma_stat, sigma_syst)``; the per-bin
    total error is ``sqrt(diag(C_stat) + diag(C_sys))``.
    """
    ab, u = prob.hessian_banded(mu)
    factor = embed_and_factor(ab, u, free)

    # statistical: M = 2 Ht^-1 (R_clip^T W), C_stat = M Sigma_y M^T
    r_clip = prob.r @ sparse.diags(free.astype(float))
    rt_w = (r_clip.T @ sparse.diags(prob.w)).toarray()
    m = 2.0 * solve_embedded(ab, u, free, rt_w, factor=factor)
    m_scaled = m * np.sqrt(sigma2)[None, :]
    c_stat = m_scaled @ m_scaled.T

    # systematic: J = -Ht^-1 G_q, C_sys = J Sigma_q J^T
    g_q = _response_gradient_fd(prob, mu, calib, sigma, mask, k)
    # solve_embedded returns zero rows for non-free bins (masked right
    # sides against the identity embedding), so J is zero there already.
    j = -solve_embedded(ab, u, free, g_q, factor=factor)
    # the direct quadratic form: numerically stable against the
    # ill-conditioned parameter covariance (no Cholesky whitening).
    c_sys = j @ calib.param_cov @ j.T

    diag_stat = np.maximum(np.diag(c_stat), 0.0)
    diag_syst = np.maximum(np.diag(c_sys), 0.0)
    return c_stat, c_sys, np.sqrt(diag_stat), np.sqrt(diag_syst)


def calibration_vertical_term(counts: np.ndarray, sigma_E: np.ndarray,
                              centers: np.ndarray, resol_params: np.ndarray
                              ) -> np.ndarray:
    """Calibration error expressed vertically: |dy/dE| * sigma_E.

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
    smooth = np.empty(n)
    for j in range(n):
        lo = max(0, int(j - 6.0 * s_ch[j]))
        hi = min(n, int(j + 6.0 * s_ch[j]) + 1)
        kk = np.arange(lo, hi, dtype=float)
        g = np.exp(-0.5 * ((kk - j) / max(s_ch[j], 1e-6)) ** 2)
        smooth[j] = float(g @ counts[lo:hi]) / max(float(g.sum()), 1e-30)

    dydE = np.empty(n)
    denom = centers[2:] - centers[:-2]
    dydE[1:-1] = (smooth[2:] - smooth[:-2]) / denom
    dydE[0] = (smooth[1] - smooth[0]) / max(h[0], 1e-30)
    dydE[-1] = (smooth[-1] - smooth[-2]) / max(h[-1], 1e-30)
    return np.abs(dydE) * sigma_E


def energy_center_errors(centers_ch: np.ndarray, param_cov: np.ndarray
                         ) -> np.ndarray:
    """1-sigma energy-center uncertainties from the calibration block."""
    g = np.stack([np.ones_like(centers_ch), centers_ch,
                  centers_ch ** 2, centers_ch ** 3], axis=1)
    var = np.einsum("ij,jk,ik->i", g, param_cov[:4, :4], g)
    return np.sqrt(np.maximum(var, 0.0))
