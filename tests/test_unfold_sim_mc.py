"""Regression for the exact F-UNC-2 simulation-MC propagation (D-119).

The reference is the direct multinomial linearization
``Cov(mu) = sum_js,kt (dmu/dG)_js (dmu/dG)_kt Cov(G_js, G_kt)`` with the FD-free
analytic derivative ``dmu/dG_js = -H^-1 v_js / N_s`` (the same derivative the
core formula implements). The refused rank-one form and the non-square shape
bug are both covered.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kc761tool.core.response import ResponseMatrix
from kc761tool.core.uncertainty import simulation_mc_variance


def _direct_reference(
    matrix: np.ndarray,
    counts: np.ndarray,
    totals: np.ndarray,
    spectrum: np.ndarray,
    mu: np.ndarray,
    sigma_fit: np.ndarray,
    hessian: np.ndarray,
) -> np.ndarray:
    """Direct first-order multinomial propagation of Cov(mu)."""
    n_primary = mu.size
    composed = matrix @ (counts / totals[None, :])
    residual = composed @ mu - spectrum
    weights = 1.0 / sigma_fit**2
    column_term = matrix.T @ (weights * residual)
    mixed = composed.T @ np.diag(weights) @ matrix
    inverse = np.linalg.inv(hessian)
    probabilities = counts / totals[None, :]
    variance = np.zeros(n_primary, dtype=np.float64)
    for column in range(n_primary):
        derivatives = np.empty((counts.shape[0], n_primary), dtype=np.float64)
        basis = np.zeros(n_primary, dtype=np.float64)
        basis[column] = 1.0
        for deposition in range(counts.shape[0]):
            vector = column_term[deposition] * basis + mixed[:, deposition] * mu[column]
            derivatives[deposition] = -inverse @ vector / totals[column]
        mean = probabilities[:, column] @ derivatives
        second = probabilities[:, column] @ (derivatives * derivatives)
        variance += totals[column] * (second - mean * mean)
    return variance


def _problem(*, seed: int, zero_category: bool) -> tuple:
    rng = np.random.default_rng(seed)
    n_rows, n_deposition, n_primary = 14, 7, 6
    matrix = rng.random((n_rows, n_deposition))
    matrix /= matrix.sum(axis=0, keepdims=True)
    counts = rng.random((n_deposition, n_primary)) * 20.0 + 1.0
    if zero_category:
        counts[:, :] *= 0.6  # N_j - sum(counts) > 0
    totals = counts.sum(axis=0) + 5.0
    sigma_fit = rng.random(n_rows) + 0.5
    mu_true = rng.random(n_primary) * 3.0 + 0.5
    composed = matrix @ (counts / totals[None, :])
    spectrum = composed @ mu_true
    hessian = composed.T @ (composed / (sigma_fit**2)[:, None]) + 1e-3 * np.eye(n_primary)
    mu = np.linalg.solve(hessian, composed.T @ (spectrum / sigma_fit**2))
    assert np.all(mu > 0.0)
    response = ResponseMatrix(
        matrix=sparse.csr_matrix(matrix),
        column_sums=matrix.sum(axis=0),
        deposition_edges_kev=np.arange(n_deposition + 1.0),
    )
    return response, counts, totals, spectrum, mu, sigma_fit, hessian


def test_simulation_mc_variance_matches_direct_linearization() -> None:
    response, counts, totals, spectrum, mu, sigma_fit, hessian = _problem(
        seed=41, zero_category=False
    )
    reference = _direct_reference(
        response.matrix.toarray(), counts, totals, spectrum, mu, sigma_fit, hessian
    )
    variance = simulation_mc_variance(
        response,
        counts,
        totals,
        spectrum,
        mu,
        sigma_fit=sigma_fit,
        fisher=sparse.csr_matrix(hessian),
    )
    assert np.all(variance >= 0.0)
    assert np.allclose(variance, reference, rtol=1e-9, atol=1e-12)


def test_simulation_mc_variance_includes_zero_deposition_category() -> None:
    # counts sum to less than N_j: the unrecorded zero-deposition category must
    # contribute through the multinomial second moment (the fixed rank-one bug).
    response, counts, totals, spectrum, mu, sigma_fit, hessian = _problem(
        seed=7, zero_category=True
    )
    reference = _direct_reference(
        response.matrix.toarray(), counts, totals, spectrum, mu, sigma_fit, hessian
    )
    variance = simulation_mc_variance(
        response,
        counts,
        totals,
        spectrum,
        mu,
        sigma_fit=sigma_fit,
        fisher=sparse.csr_matrix(hessian),
    )
    assert np.allclose(variance, reference, rtol=1e-9, atol=1e-12)


def test_simulation_mc_variance_accepts_non_square_shapes() -> None:
    # n_primary != n_deposition used to raise on the old broadcasting.
    response, counts, totals, spectrum, mu, sigma_fit, hessian = _problem(
        seed=3, zero_category=False
    )
    assert counts.shape[0] != mu.size
    variance = simulation_mc_variance(
        response,
        counts,
        totals,
        spectrum,
        mu,
        sigma_fit=sigma_fit,
        fisher=sparse.csr_matrix(hessian),
    )
    assert variance.shape == mu.shape
