"""Auxiliary checks for Fisher covariance, scaling and profiling (F-COV-1..3)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761.core import covariance
from kc761.errors import CertificateError, SolverError, ValidationError


def test_fisher_information_matches_weighted_product() -> None:
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(9, 4))
    sigma = rng.random(9) + 0.5
    fisher = covariance.fisher_information(jacobian, sigma)
    reference = jacobian.T @ (jacobian / sigma[:, None] ** 2)
    assert np.allclose(fisher, reference, rtol=1e-12)


def test_scaled_covariance_and_certificate() -> None:
    fisher = np.array([[4.0, 1.0], [1.0, 3.0]])
    estimate = covariance.scaled_covariance(fisher, chi2=9.0, dof=3)
    assert estimate.scale == pytest.approx(3.0)
    assert np.allclose(estimate.matrix, 3.0 * np.linalg.inv(fisher))
    smallest = covariance.verify_covariance_psd(estimate, strict=True)
    assert smallest > 0.0


def test_scaled_covariance_rejects_bad_fisher_and_dof() -> None:
    with pytest.raises(ValidationError):
        covariance.scaled_covariance(np.eye(2), chi2=1.0, dof=0)
    with pytest.raises(SolverError):
        covariance.scaled_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), chi2=1.0, dof=1)


def test_psd_certificate_fails_on_negative_spectrum() -> None:
    estimate = covariance.CovarianceEstimate(
        matrix=np.array([[1.0, 2.0], [2.0, 1.0]]),
        scale=1.0,
        chi2=1.0,
        dof=1,
        estimator="test",
    )
    with pytest.raises(CertificateError, match="F-COV-2"):
        covariance.verify_covariance_psd(estimate, strict=True)


def test_profile_covariance_recovers_quadratic_covariance() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    center = np.array([1.0, -2.0])

    def objective(parameters: np.ndarray) -> float:
        delta = parameters - center
        return 0.5 * float(delta @ matrix @ delta)

    estimate = covariance.profile_covariance(objective, center, profile_points=25)
    assert estimate.estimator == "profile"
    # For chi2(q) = 1/2 d^T A d the covariance is 2 A^-1 (profile crossing
    # dchi2 = 1 and the covariance convention dchi2 = d^T F d with F = A/2).
    expected = 2.0 * np.linalg.inv(matrix)
    assert np.allclose(estimate.matrix, expected, rtol=3e-2, atol=1e-3)


def test_profile_covariance_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationError):
        covariance.profile_covariance(lambda q: float(q @ q), np.zeros(2), profile_points=3)
