"""Auxiliary checks for the non-negative Tikhonov solver (F-SOLVE-1..3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, sparse

from kc761tool.core import solver
from kc761tool.errors import ValidationError


def _problem(seed: int = 0, m: int = 12, n: int = 6):
    rng = np.random.default_rng(seed)
    response = sparse.csr_matrix(rng.random((m, n)))
    spectrum = rng.random(m) * 10.0 + 1.0
    sigma = np.sqrt(spectrum) + 0.5
    return response, spectrum, sigma


def test_solution_is_nonnegative_with_valid_kkt() -> None:
    response, spectrum, sigma = _problem()
    solution = solver.solve_nonnegative(
        response, spectrum, sigma, solver.RegularizationSpec(alpha=0.05), strict=True
    )
    assert np.all(solution.mu >= 0.0)
    assert solution.certificate.ok
    certificate = solver.verify_kkt(
        response, spectrum, sigma, solver.RegularizationSpec(alpha=0.05), solution.mu
    )
    assert certificate.ok
    assert np.isfinite(solution.objective)


def test_injected_normal_equations_match_internal_build() -> None:
    """D-150: injecting (H, b, penalty_scale) must not change the solution."""
    response, spectrum, sigma = _problem(seed=11)
    spec = solver.RegularizationSpec(alpha=0.05)
    hessian, gradient, penalty_scale = solver.normal_equations(
        response, spectrum, sigma, spec
    )
    internal = solver.solve_nonnegative(response, spectrum, sigma, spec, strict=True)
    injected = solver.solve_nonnegative(
        response,
        spectrum,
        sigma,
        spec,
        strict=True,
        normal=(hessian, gradient, penalty_scale),
    )
    assert np.array_equal(internal.mu, injected.mu)
    assert internal.objective == injected.objective


def test_matches_reference_minimizer() -> None:
    response, spectrum, sigma = _problem(seed=4)
    spec = solver.RegularizationSpec(alpha=0.2)
    solution = solver.solve_nonnegative(response, spectrum, sigma, spec, strict=True)
    hessian, gradient, _ = solver.normal_equations(response, spectrum, sigma, spec)

    def objective(mu: np.ndarray) -> float:
        return float(0.5 * mu @ np.asarray(hessian @ mu) - gradient @ mu)

    reference = optimize.minimize(
        objective,
        np.zeros(gradient.size),
        method="L-BFGS-B",
        bounds=[(0.0, None)] * gradient.size,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000},
    )
    assert objective(solution.mu) <= reference.fun + 1e-9
    assert np.allclose(solution.mu, reference.x, atol=1e-5)


def test_zero_column_is_fixed_at_zero() -> None:
    rng = np.random.default_rng(2)
    dense = rng.random((10, 4))
    dense[:, 2] = 0.0
    response = sparse.csr_matrix(dense)
    spectrum = rng.random(10) + 1.0
    sigma = np.ones(10)
    solution = solver.solve_nonnegative(
        response, spectrum, sigma, solver.RegularizationSpec(alpha=0.1), strict=True
    )
    assert solution.mu[2] == 0.0
    assert solution.certificate.ok


def test_random_problems_satisfy_kkt() -> None:
    for seed in range(5):
        response, spectrum, sigma = _problem(seed=seed, m=10, n=5)
        spec = solver.RegularizationSpec(alpha=0.01 * (seed + 1))
        solution = solver.solve_nonnegative(response, spectrum, sigma, spec, strict=True)
        assert solution.certificate.ok
        assert solution.certificate.max_negative_reduced_gradient <= solver.KKT_TOL
        assert solution.certificate.complementarity <= solver.KKT_TOL


def test_invalid_inputs_raise() -> None:
    response, spectrum, sigma = _problem()
    spec = solver.RegularizationSpec(alpha=0.1)
    with pytest.raises(ValidationError):
        solver.solve_nonnegative(response, spectrum, np.zeros_like(sigma), spec)
    with pytest.raises(ValidationError):
        solver.solve_nonnegative(response, spectrum[:-1], sigma, spec)
    with pytest.raises(ValidationError):
        solver.RegularizationSpec(alpha=0.0)
    with pytest.raises(ValidationError):
        solver.RegularizationSpec(alpha=1.0, difference_order=3)


def test_difference_operator_shapes() -> None:
    first = solver.difference_operator(5, 1)
    second = solver.difference_operator(5, 2)
    assert first.shape == (4, 5)
    assert second.shape == (3, 5)
    assert np.array_equal(first.toarray()[0], [-1.0, 1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(second.toarray()[0], [1.0, -2.0, 1.0, 0.0, 0.0])
