"""SWR (significance-weighted robust) regularization for unfolding.

Design and sources
------------------

The unfolded density rho = mu / w (counts per keV, w the energy bin
width) is regularized by the penalty

    Omega = sum_j rho_delta( m_j * (L_k rho)_j / sigma_d,j ),

where L_k is the k-th difference operator on the energy grid, sigma_d,j
is the statistical noise of that difference, m_j a peak mask, and
rho_delta the Huber loss.  The three ingredients and their origins:

- Significance normalization (sigma_d): penalizing absolute counts
  under-regularizes low-count continua (where the noise is small) and
  over-regularizes peaks (where it is large).  Measuring differences in
  units of their exact local noise makes both alpha and delta
  dimensionless significance parameters that transfer across spectra
  and datasets.

- SNIP baseline and peak mask: SNIP (Ryan, Clayton, Griffin, Sie and
  Cousens, Nucl. Instrum. Methods B 34 (1988) 396-402) extracts a
  smooth background in log space by iterative peak clipping.  The mask
  m_j = 1 / (1 + max(0, p_j)^2 / p0^2), p_j = (y_j - b_j) / sigma_j,
  suppresses the penalty on significant peaks, so the smoothing budget
  is spent on the continuum instead of rounding peaks; the floor gmin
  keeps high-SNR peaks from being sharpened below the resolution.

- Huber loss (Huber, Ann. Math. Statist. 35 (1964) 73-101): quadratic
  for |d| <= delta, linear beyond.  An L2 penalty grows super-linearly
  with the peak slopes and rounds them; the Huber linear branch caps
  that cost, so noise-level differences in the continuum are still
  suppressed quadratically while peak flanks are treated gently.

The three components are not found combined under this name in the
unfolding literature; the combination and the name are this toolkit's
design, motivated by the artifacts observed on KC761 spectra (spurious
peaks in continua under plain L2, peak rounding under strong L2).
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
    n = v.size
    for p in range(int(n_iter), 0, -1):
        lo = np.concatenate([v[:p], v[:n - p]])
        hi = np.concatenate([v[p:], v[n - p:]])
        v = np.minimum(v, 0.5 * (lo + hi))
    return np.exp(v)


def peak_mask(counts: np.ndarray, sigma: np.ndarray, baseline: np.ndarray,
              p0: float, gmin: float) -> np.ndarray:
    """Regularization mask from the local peak significance.

    ``m = 1 / (1 + max(0, p)^2 / p0^2)`` with p = (y - baseline) / sigma,
    floored at ``gmin`` so high-significance peaks keep a minimum of
    regularization (prevents sharpening below the detector resolution).
    """
    p = (np.asarray(counts, dtype=float) - np.asarray(baseline, dtype=float)
         ) / np.asarray(sigma, dtype=float)
    m = 1.0 / (1.0 + np.maximum(p, 0.0) ** 2 / p0 ** 2)
    return np.maximum(m, gmin)


def swr_operator(widths: np.ndarray, centers: np.ndarray, sigma: np.ndarray,
                 mask: np.ndarray, k: int) -> sparse.csr_matrix:
    """The normalized difference operator D of the SWR penalty.

    (D mu)_r = m_jc * (L_k rho)_r / sigma_d,r, with rho = mu / w and
    sigma_d,r = sqrt(|L_k row|^2 @ sigma_rho^2) the exact statistical
    noise of the r-th difference of the density (independent bins,
    including the 1/h energy-spacing factors of L_k).  D mu is
    dimensionless and unit-variance under the null, so the SWR strength
    alpha and the Huber threshold delta are dimensionless, problem-
    independent parameters.
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


# --------------------------------------------------------------------------
# Huber loss and its derivatives (elementwise)


def huber_value(d: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(d)
    return np.where(a <= delta, 0.5 * d ** 2, delta * (a - 0.5 * delta))


def huber_grad(d: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(d, -delta, delta)


def huber_hess(d: np.ndarray, delta: float) -> np.ndarray:
    return (np.abs(d) <= delta).astype(float)
