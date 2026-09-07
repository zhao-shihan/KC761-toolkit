"""Model-independent response composition and full error propagation.

Composes R = C @ G for the kc761sim matrix modes, where

* C is the calibration deposition response (n x n), random through the
  fit parameters with the per-element gradient tensor J (n x n x 7) and
  the 7x7 parameter covariance ``param_cov`` (reported basis; NaN
  rows/columns mark undetermined parameters and are treated as fixed,
  matching :mod:`kc761unfold.reader`);
* G holds Monte Carlo primary->deposition counts (n x n) whose columns
  are multinomial with the per-column totals ``totals`` -- the number of
  *primary* events, zero-deposition events included.  The zero-deposition
  category is one more outcome of the same multinomial and enters the
  covariance through the total-probability constraint (its score is 0 in
  every observable, so it needs no explicit term).

Linearized error propagation (C and G are independent):

    R[a,b]     = sum_j C[a,j] p_jb,         p_jb = G[j,b] / totals_b,
    Var_G[a,b] = ( sum_j C[a,j]^2 p_jb - ( sum_j C[a,j] p_jb )^2 ) / totals_b,
    h[a,b]     = sum_j p_jb J[a,j],         (7-vector per element)
    Var_C[a,b] = h[a,b]^T cov_q h[a,b],
    Var_R[a,b] = Var_C[a,b] + Var_G[a,b],

and the per-column detection efficiency

    eff_b        = sum_a R[a,b] = sum_j s_j p_jb,   s_j = sum_a C[a,j],
    Var_G_eff_b  = ( sum_j s_j^2 p_jb - ( sum_j s_j p_jb )^2 ) / totals_b,
    Var_C_eff_b  = ( sum_j p_jb S_j )^T cov_q ( sum_j p_jb S_j ),
                   S_j = sum_a J[a,j],

All outputs are per-element variances -- the diagonal of the full
linearized covariance -- matching the kc761calib convention of storing
1-sigma squared errors in fSumw2.  Columns with ``totals_b == 0``
(skipped negative-energy primary columns) yield an all-zero R column and
efficiency 0.  This module is model-independent: the Jacobian J is
supplied by the caller (:mod:`kc761calib.matrixjac`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PARAMS = 7


@dataclass
class ComposedResponse:
    """R = C @ G with the per-element variances and the detection efficiency."""

    matrix: np.ndarray  # R, (n, n)
    variance: np.ndarray  # Var_R, (n, n)
    efficiency: np.ndarray  # per-column detection efficiency, (n,)
    efficiency_variance: np.ndarray  # (n,)


def compose_response(C, C_jac, param_cov, G_counts, totals) -> ComposedResponse:
    """Compose R = C @ (column-normalized G) with full linearized errors.

    ``C`` is the dense deposition response (n x n); ``C_jac`` its
    gradient tensor J (n x n x 7) in the ``param_cov`` basis; ``G_counts``
    the Monte Carlo counts (n x n); ``totals`` the per-column number of
    primary events (n,), zero-deposition events included.
    """
    C = np.asarray(C, dtype=float)
    J = np.asarray(C_jac, dtype=float)
    cov = np.asarray(param_cov, dtype=float)
    G = np.asarray(G_counts, dtype=float)
    totals = np.asarray(totals, dtype=float)

    n = C.shape[0]
    if C.shape != (n, n) or G.shape != (n, n) or J.shape != (n, n, N_PARAMS):
        raise ValueError(
            f"shape mismatch: C {C.shape}, J {J.shape}, G {G.shape} must "
            f"satisfy (n, n), (n, n, 7), (n, n)")
    if cov.shape != (N_PARAMS, N_PARAMS):
        raise ValueError(f"param_cov must have shape (7, 7), got {cov.shape}")
    if totals.shape != (n,):
        raise ValueError(f"totals must have shape ({n},), got {totals.shape}")
    if (C < 0).any():
        raise ValueError("deposition response C contains negative entries")
    if (G < 0).any():
        raise ValueError("G counts contain negative entries")
    if (totals < 0).any():
        raise ValueError("per-column totals contain negative entries")
    if np.any(totals < G.sum(axis=0)):
        raise ValueError(
            "per-column totals must include the zero-deposition events "
            "(totals_b >= sum_a G[a, b])")

    # Undetermined parameters (NaN covariance rows/columns) are treated as
    # fixed -- the same policy as kc761unfold.reader.
    cov_w = np.where(np.isnan(cov), 0.0, cov)

    # Column normalization; zero-total (skipped) columns keep all zeros.
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    p = G / safe_totals

    R = C @ p

    # Monte Carlo (multinomial) variance, exact for the full column
    # multinomial including the zero-deposition category.  The first term
    # inside the column sums is the same product R = C @ p.
    C2p = (C * C) @ p
    var_g = np.where(
        totals > 0.0, (C2p - R * R) / np.maximum(totals, 1.0)[None, :], 0.0)

    # Calibration-fit variance: h[a,b] = sum_j p_jb J[a,j] (7-vector per
    # element); Var_C = h^T cov h.  Contracted in column chunks so the
    # intermediate stays at (n, chunk, 7) next to the caller's J instead
    # of materializing a second full (n, n, 7) tensor.
    var_c = np.empty((n, n))
    chunk = max(1, (1 << 22) // (n * N_PARAMS))  # ~32 MB per chunk tensor
    for b0 in range(0, n, chunk):
        b1 = min(n, b0 + chunk)
        H_b = np.einsum("ajp,jb->abp", J, p[:, b0:b1], optimize=True)
        var_c[:, b0:b1] = np.einsum(
            "abp,pq,abq->ab", H_b, cov_w, H_b, optimize=True)

    variance = np.maximum(var_c + var_g, 0.0)

    # Detection efficiency per primary column and its variance.
    s = C.sum(axis=0)  # column sums of C, (n,)
    eff = s @ p  # (n,)
    var_g_eff = np.where(
        totals > 0.0,
        ((s * s) @ p - eff * eff) / np.maximum(totals, 1.0),
        0.0)
    S = J.sum(axis=0)  # (n, 7): S_j = sum_a J[a,j]
    h_eff = p.T @ S  # (n, 7): sum_j p_jb S_j
    var_c_eff = np.einsum("bp,pq,bq->b", h_eff, cov_w, h_eff, optimize=True)
    efficiency_variance = np.maximum(var_c_eff + var_g_eff, 0.0)

    return ComposedResponse(
        matrix=R,
        variance=variance,
        efficiency=eff,
        efficiency_variance=efficiency_variance,
    )
