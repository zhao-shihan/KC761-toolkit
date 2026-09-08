"""Hybrid regularized unfolding: penalty operator for unfolding.

The density ``rho = mu / w`` (w the energy bin width) is regularized by
the quadratic penalty ``Omega = ||D mu||^2`` with
``D = diag(m/sigma_d) L_k diag(1/w)``: the k-th difference of rho in
units of its exact statistical noise ``sigma_d``, weighted by the SNIP
peak mask ``m`` (see Ryan et al., Nucl. Instrum. Methods B 34 (1988)
396-402).  The significance normalization makes the strength ``alpha`` a
dimensionless parameter that transfers across spectra.

The operator and its calibration-parameter derivatives share one
coefficient source: :func:`_diff_coefficients` builds the ``L_k``
coefficients and (optionally) their exact derivatives from the same
spacing quantities, and :func:`_penalty_geometry` the ``L``, ``sigma_d``
and mask pieces of ``D``, so the value and the derivative can never
drift apart.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _diff_coefficients(centers: np.ndarray, k: int,
                       d_centers: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray | None]:
    """Coefficients of the k-th difference operator ``L_k`` per row.

    Returns ``(coef, cols, dcoef)`` with ``coef``/``cols`` shaped
    ``(n - k, k + 1)`` holding the value and column index of each
    non-zero entry, and ``dcoef`` the exact derivative coefficients when
    ``d_centers`` (the bin-center derivative ``dc/dq``) is given, else
    None.  Both the value and the derivative come from the same local
    spacing quantities:

    k = 1:  h = c[i+1] - c[i],  d(+-1/h) = -+ dh / h^2;
    k = 2:  coef_l = 2/(denom h_l), coef_r = 2/(denom h_r) with
            h_l = c[j]-c[j-1], h_r = c[j+1]-c[j],
            denom = (c[j+1]-c[j-1])/2, and
            dcoef_l = -coef_l (ddenom/denom + dh_l/h_l) (same for r).
    """
    n = len(centers)
    c = centers
    if k == 1:
        rows = np.arange(n - 1)
        h = c[1:] - c[:-1]
        coef = np.stack([-1.0 / h, 1.0 / h], axis=1)
        cols = np.stack([rows, rows + 1], axis=1)
        if d_centers is None:
            return coef, cols, None
        dh = np.asarray(d_centers)[1:] - np.asarray(d_centers)[:-1]
        dcoef = np.stack([dh / h ** 2, -dh / h ** 2], axis=1)
        return coef, cols, dcoef
    if k == 2:
        j = np.arange(1, n - 1)
        rows = j - 1
        h_l = c[j] - c[j - 1]
        h_r = c[j + 1] - c[j]
        denom = 0.5 * (c[j + 1] - c[j - 1])
        coef_l = 2.0 / (denom * h_l)
        coef_r = 2.0 / (denom * h_r)
        coef = np.stack([coef_l, -(coef_l + coef_r), coef_r], axis=1)
        cols = np.stack([j - 1, j, j + 1], axis=1)
        if d_centers is None:
            return coef, cols, None
        dc = np.asarray(d_centers)
        dh_l = dc[j] - dc[j - 1]
        dh_r = dc[j + 1] - dc[j]
        ddenom = 0.5 * (dc[j + 1] - dc[j - 1])
        dcoef_l = -coef_l * (ddenom / denom + dh_l / h_l)
        dcoef_r = -coef_r * (ddenom / denom + dh_r / h_r)
        dcoef = np.stack([dcoef_l, -(dcoef_l + dcoef_r), dcoef_r], axis=1)
        return coef, cols, dcoef
    raise ValueError("k must be 1 or 2")


def _diff_matrix(coef: np.ndarray, cols: np.ndarray, n: int, k: int
                 ) -> sparse.csr_matrix:
    """Assemble the (n - k) x n CSR matrix of a coefficient triplet."""
    rows = np.repeat(np.arange(n - k), k + 1)
    return sparse.csr_matrix((coef.ravel(), (rows, cols.ravel())),
                             shape=(n - k, n))


def diff_operator(centers: np.ndarray, k: int) -> sparse.csr_matrix:
    """k-th difference operator on the non-uniform energy grid.

    Row r of the returned (n - k) x n matrix carries the k-th finite
    difference centered on bin r + k // 2, divided by the local energy
    spacing so the result is a derivative per keV.
    """
    coef, cols, _ = _diff_coefficients(centers, k)
    return _diff_matrix(coef, cols, len(centers), k)


def diff_operator_grad(centers: np.ndarray, k: int,
                       d_centers: np.ndarray) -> sparse.csr_matrix:
    """Exact derivative ``dL_k/dc_p`` of :func:`diff_operator`.

    ``d_centers`` is the derivative of the bin centers with respect to
    one calibration parameter (same length as ``centers``).  The
    coefficients come from the same source as ``L_k`` itself
    (:func:`_diff_coefficients`), so the returned matrix has the same
    sparsity pattern and is consistent with the operator by construction.
    """
    _, cols, dcoef = _diff_coefficients(centers, k, d_centers)
    return _diff_matrix(dcoef, cols, len(centers), k)


def snip_baseline(counts: np.ndarray, n_iter: int) -> np.ndarray:
    """SNIP background estimate: log-domain iterative peak clipping.

    The log spectrum is clipped with a shrinking window
    (v[i] = min(v[i], (v[i-p] + v[i+p]) / 2) for p = n_iter..1) and
    exp() recovers a smooth background running under the peaks through
    the continuum.  Non-positive bins are clamped to 1 before the log.
    The update is vectorized with clamped shifts: bin i's neighbours are
    v[max(i-p, 0)] and v[min(i+p, n-1)], evaluated on the previous
    iteration's v (the same non-causal semantics as the scalar loop).
    """
    y = np.maximum(np.asarray(counts, dtype=float), 1.0)
    v = np.log(y)
    n = len(y)
    if n == 0:
        return np.exp(v)
    for p in range(n_iter, 0, -1):
        left = np.concatenate([np.full(p, v[0]), v])[:n]
        right = np.concatenate([v, np.full(p, v[-1])])[p:]
        v = np.minimum(v, 0.5 * (left + right))
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


def _penalty_geometry(widths: np.ndarray, centers: np.ndarray,
                      sigma: np.ndarray, mask: np.ndarray, k: int
                      ) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray,
                                 np.ndarray, np.ndarray]:
    """Shared pieces of ``D = diag(m_jc/sigma_d) L_k diag(1/w)``.

    Returns ``(w, l, sd, m_jc, sigma_rho2)``: the widths, the difference
    operator, the significance normalization ``sigma_d``, the mask
    entries at the difference rows, and the squared density noise
    ``sigma_rho2 = (sigma/w)^2`` that both the operator (inside ``sd``)
    and the derivative (inside ``d(sigma_d^2)``) consume.  Both
    :func:`penalty_operator` and :func:`penalty_operator_grad` assemble
    ``D`` and ``dD`` from these same pieces.
    """
    w = np.asarray(widths, dtype=float)
    s = np.asarray(sigma, dtype=float)
    l = diff_operator(centers, k)
    jc = np.arange(l.shape[0]) + k // 2
    sigma_rho2 = (s / w) ** 2
    sd = np.sqrt(l.multiply(l) @ sigma_rho2)
    m_jc = np.asarray(mask, dtype=float)[jc]
    return w, l, sd, m_jc, sigma_rho2


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
    w, l, sd, m_jc, _ = _penalty_geometry(widths, centers, sigma, mask, k)
    return sparse.diags(m_jc / sd) @ l @ sparse.diags(1.0 / w)


def penalty_operator_grad(widths: np.ndarray, centers: np.ndarray,
                          sigma: np.ndarray, mask: np.ndarray, k: int,
                          d_centers: np.ndarray, d_widths: np.ndarray
                          ) -> sparse.csr_matrix:
    """Exact derivative ``dD/dc_p`` of :func:`penalty_operator`.

    ``d_centers``/``d_widths`` are the derivatives of the energy bin
    centers/widths with respect to one calibration parameter ``c_p``
    (``sigma`` and ``mask`` are data-side and do not depend on it).
    With ``D = S1 L S2``, ``S1 = diag(m_jc / sigma_d)`` and
    ``S2 = diag(1/w)``, the chain rule gives

        dD = dS1 L S2 + S1 dL S2 + S1 L dS2,

    using :func:`diff_operator_grad` for ``dL`` and

        d(sigma_d^2) = d(L^2) @ (s/w)^2 + L^2 @ ((s/w)^2 (-2 dw/w)),
        d(L^2) = dL .* L + L .* dL   (elementwise),

    the exact differential of the significance normalization.  The
    returned matrix has the same sparsity pattern as ``D`` and shares
    the operator's ``L``/``sigma_d``/mask pieces via
    :func:`_penalty_geometry`.
    """
    w, l, sd, m_jc, sigma_rho2 = _penalty_geometry(widths, centers, sigma,
                                                   mask, k)
    dw = np.asarray(d_widths, dtype=float)
    dl = diff_operator_grad(centers, k, d_centers)
    dl2 = dl.multiply(l) + l.multiply(dl)
    dsd2 = dl2 @ sigma_rho2 + l.multiply(l) @ (sigma_rho2 * (-2.0 * dw / w))
    dsd = 0.5 * dsd2 / sd

    row_w = m_jc / sd
    return (sparse.diags(-m_jc * dsd / sd ** 2) @ l @ sparse.diags(1.0 / w)
            + sparse.diags(row_w) @ dl @ sparse.diags(1.0 / w)
            + sparse.diags(row_w) @ l @ sparse.diags(-dw / w ** 2))
