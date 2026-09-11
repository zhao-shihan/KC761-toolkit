"""Auxiliary checks for uncertainty propagation (F-UNC-1..3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from kc761tool.core import solver, uncertainty
from kc761tool.core.response import ResponseMatrix
from kc761tool.errors import CertificateError


def test_statistical_band_for_identity_response() -> None:
    spectrum = np.array([2.0, 4.0, 9.0, 16.0])
    sigma = np.sqrt(spectrum)
    band, components = uncertainty.propagate_statistical(
        sparse.eye(4, format="csr"),
        spectrum,
        sigma,
        np.ones(4),
        stat_variance=spectrum,
    )
    assert np.allclose(band, np.sqrt(spectrum))
    assert components[0].kind == "stat"
    assert components[0].formula_id == "F-UNC-1"


def test_data_side_systematic_band() -> None:
    spectrum = np.array([10.0, 20.0])
    sigma = np.ones(2)
    band, components = uncertainty.propagate_systematic(
        sparse.eye(2, format="csr"),
        np.ones(2),
        spectrum=spectrum,
        sigma_fit=sigma,
        calib_jacobian=[],
        calib_covariance=np.zeros((0, 0)),
        syst_frac=0.1,
    )
    assert np.allclose(band, 0.1 * spectrum)
    assert components[0].name == "data_syst_frac"
    assert components[0].kind == "syst"


def test_calibration_systematic_analytic_case() -> None:
    band, components = uncertainty.propagate_systematic(
        sparse.csr_matrix(np.array([[1.0]])),
        np.array([2.0]),
        spectrum=np.array([2.0]),
        sigma_fit=np.array([1.0]),
        calib_jacobian=[sparse.csr_matrix(np.array([[0.5]]))],
        calib_covariance=np.array([[0.04]]),
    )
    assert np.allclose(band, [0.2])
    assert components[0].name == "calibration"


def test_band_decomposition_and_certificate() -> None:
    stat = np.array([1.0, 2.0, 0.0])
    syst = np.array([2.0, 0.0, 3.0])
    bands = uncertainty.combine_bands(stat, syst)
    assert np.allclose(bands.sigma_total, np.hypot(stat, syst))
    assert uncertainty.verify_band_decomposition(bands, strict=True) <= 1e-12
    broken = uncertainty.UncertaintyBands(
        sigma_stat=stat,
        sigma_syst=syst,
        sigma_total=np.zeros(3),
        components=(),
    )
    with pytest.raises(CertificateError, match="F-UNC-3"):
        uncertainty.verify_band_decomposition(broken, strict=True)


def test_statistical_band_uses_reduced_system_at_boundary() -> None:
    response = sparse.csr_matrix(
        np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0]])
    )
    spectrum = np.array([5.0, 1.0, 4.0, 1.0])
    sigma = np.ones(4)
    stat = np.maximum(spectrum, 1.0)
    spec = solver.RegularizationSpec(alpha=0.01)
    solution = solver.solve_nonnegative(response, spectrum, sigma, spec, strict=True)
    assert solution.mu[1] == 0.0
    hessian, _, _ = solver.normal_equations(response, spectrum, sigma, spec)
    band, _ = uncertainty.propagate_statistical(
        response, spectrum, sigma, solution.mu, stat_variance=stat, fisher=hessian
    )
    assert band[1] == 0.0
    free = solution.mu > 0.0
    reduced = hessian.toarray()[free][:, free]
    sensitivity = (response.T @ sparse.diags(1.0 / sigma**2)).toarray()[free]
    reference_free = np.linalg.solve(reduced, sensitivity)
    reference = np.zeros((3, 4))
    reference[free] = reference_free
    assert np.allclose(band, np.sqrt((reference**2) @ stat))
    full_inverse = np.linalg.solve(
        hessian.toarray(), (response.T @ sparse.diags(1.0 / sigma**2)).toarray()
    )
    full_band = np.sqrt((full_inverse**2) @ stat)
    assert full_band[1] > 0.0  # the full inverse would violate the boundary
    assert not np.allclose(band, full_band)


def test_simulation_mc_variance_matches_finite_difference() -> None:
    matrix = sparse.eye(3, format="csr")
    response = ResponseMatrix(
        matrix=matrix,
        column_sums=np.ones(3),
        deposition_edges_kev=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    counts = np.array([[20.0], [30.0], [40.0]])
    totals = np.array([100.0])
    spectrum = np.array([5.0, 6.0, 7.0])
    sigma = np.array([1.0, 1.0, 1.0])
    spec = solver.RegularizationSpec(alpha=0.01)
    hessian, _, _ = solver.normal_equations(
        sparse.csr_matrix(np.array([[0.2], [0.3], [0.4]])), spectrum, sigma, spec
    )
    solution = solver.solve_nonnegative(
        sparse.csr_matrix(np.array([[0.2], [0.3], [0.4]])), spectrum, sigma, spec, strict=True
    )
    assert np.all(solution.mu > 0.0)

    def solve_for(g_values: np.ndarray) -> np.ndarray:
        composed = sparse.csr_matrix((g_values / totals[0]).reshape(3, 1))
        return solver.solve_nonnegative(composed, spectrum, sigma, spec).mu

    step = 0.5
    derivatives = np.zeros((1, 3))
    for index in range(3):
        plus = counts[:, 0].copy()
        minus = counts[:, 0].copy()
        plus[index] += step
        minus[index] -= step
        derivatives[:, index] = (solve_for(plus) - solve_for(minus)) / (2.0 * step)
    q = counts[:, 0] / totals[0]
    covariance = totals[0] * (np.diag(q) - np.outer(q, q))
    reference = np.einsum("ij,jk,ik->i", derivatives, covariance, derivatives)

    variance = uncertainty.simulation_mc_variance(
        response,
        counts,
        totals,
        spectrum,
        solution.mu,
        sigma_fit=sigma,
        fisher=hessian,
    )
    assert np.allclose(variance, reference, rtol=0.15, atol=1e-6)


def test_default_syst_frac_is_five_percent() -> None:
    """D-169: the single-sourced default data-side systematic is 5%."""
    assert uncertainty.DEFAULT_SYST_FRAC == 0.05
