"""Hybrid regularized unfolding: penalty operator for unfolding.

The density ``rho = mu / w`` (w the energy bin width) is regularized by
the quadratic penalty ``Omega = ||D mu||^2`` with
``D = diag(m/sigma_d) L_k diag(1/w)``: the k-th difference of rho in
units of its exact statistical noise ``sigma_d``, weighted by the SNIP
peak mask ``m`` (see Ryan et al., Nucl. Instrum. Methods B 34 (1988)
396-402).  The significance normalization makes the strength ``alpha`` a
dimensionless parameter that transfers across spectra.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def diff_operator(centers: np.ndarray, k: int) -> sparse.csr_matrix:
    """k-th difference operator on the non-uniform energy grid.

    Row r of the returned (n - k) x n matrix carries the k-th finite
    difference centered on bin r + k // 2, divided by the local energy
    spacing so the result is a derivative per keV.
    """
    n = len(centers)
    c = centers
    if k == 1:
        rows = np.arange(n - 1)
        h = c[1:] - c[:-1]
        return sparse.csr_matrix(
            (np.concatenate([-1.0 / h, 1.0 / h]),
             (np.concatenate([rows, rows]),
              np.concatenate([rows, rows + 1]))),
            shape=(n - 1, n))
    if k == 2:
        j = np.arange(1, n - 1)
        rows = j - 1
        h_l = c[j] - c[j - 1]
        h_r = c[j + 1] - c[j]
        denom = 0.5 * (c[j + 1] - c[j - 1])
        coef_l = 2.0 / (denom * h_l)
        coef_r = 2.0 / (denom * h_r)
        return sparse.csr_matrix(
            (np.concatenate([coef_l, -(coef_l + coef_r), coef_r]),
             (np.concatenate([rows, rows, rows]),
              np.concatenate([j - 1, j, j + 1]))),
            shape=(n - 2, n))
    raise ValueError("k must be 1 or 2")


def snip_baseline(counts: np.ndarray, n_iter: int) -> np.ndarray:
    """SNIP background estimate: log-domain iterative peak clipping.

    The log spectrum is clipped with a shrinking window
    (v[i] = min(v[i], (v[i-p] + v[i+p]) / 2) for p = n_iter..1) and
    exp() recovers a smooth background running under the peaks through
    the continuum.  Non-positive bins are clamped to 1 before the log.
    """
    y = np.maximum(np.asarray(counts, dtype=float), 1.0)
    v = np.log(y)
    n = len(y)
    for p in range(n_iter, 0, -1):
        vp = np.empty_like(v)
        for i in range(n):
            lo = max(i - p, 0)
            hi = min(i + p, n - 1)
            vp[i] = min(v[i], 0.5 * (v[lo] + v[hi]))
        v = vp
    return np.exp(v)


def peak_mask(counts: np.ndarray, sigma: np.ndarray, baseline: np.ndarray,
              mask_z0: float, mask_floor: float) -> np.ndarray:
    """Regularization mask from the local peak significance.

    ``m = 1 / (1 + max(0, p)^2 / mask_z0^2)`` with the significance
    ``p = (y - baseline) / sigma`` (a z-score; the mask reaches 1/2 at
    ``p = mask_z0``), floored at ``mask_floor`` so high-significance
    peaks keep a minimum of regularization (prevents sharpening below
    the detector resolution).
    """
    p = (np.asarray(counts, dtype=float) - np.asarray(baseline, dtype=float)
         ) / np.asarray(sigma, dtype=float)
    m = 1.0 / (1.0 + np.maximum(p, 0.0) ** 2 / mask_z0 ** 2)
    return np.maximum(m, mask_floor)


def penalty_operator(widths: np.ndarray, centers: np.ndarray, sigma: np.ndarray,
                     mask: np.ndarray, k: int) -> sparse.csr_matrix:
    """The normalized masked difference operator D of the penalty.

    ``(D mu)_r = m_jc * (L_k rho)_r / sigma_d,r`` with ``rho = mu / w``
    and ``sigma_d,r = sqrt(|L_k row|^2 @ sigma_rho^2)`` the exact
    statistical noise of the r-th difference of the density (independent
    bins, including the 1/h energy-spacing factors of L_k).  ``D mu`` is
    dimensionless and unit-variance under the null, so the regularization strength
    ``alpha`` is a dimensionless, problem-independent parameter.
    """
    w = np.asarray(widths, dtype=float)
    s = np.asarray(sigma, dtype=float)
    l = diff_operator(centers, k)
    n_rows = l.shape[0]
    jc = np.arange(n_rows) + k // 2
    sigma_rho = s / w
    sd = np.sqrt(l.multiply(l) @ (sigma_rho ** 2))
    row_w = np.asarray(mask, dtype=float)[jc] / sd
    return sparse.diags(row_w) @ l @ sparse.diags(1.0 / w)
