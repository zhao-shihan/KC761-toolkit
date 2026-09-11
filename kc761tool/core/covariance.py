"""Calibration parameter covariance: Fisher information, scaling, profiling.

Formula IDs (docs/derivations.md): F-COV-1 .. F-COV-3.

* F-COV-1: the covariance is the analytic Fisher information
  ``F = J^T W J`` from the sympy-generated Jacobian ``J`` of the model with
  respect to the fit parameters.
* F-COV-2: ``cov = s**2 * F**-1`` with ``s**2 = chi2/dof`` (PDG convention),
  enabled by default, and both ``chi2``/``dof`` and the scale are recorded in
  the product (D-49). A singular or non-positive-definite Fisher matrix is a
  hard failure: no pseudo-inverse fallback.
* F-COV-3: profile covariance is an optional diagnostic only; it never
  replaces the analytic estimate.

A mixed estimator (marginal slice widths times a Gaussian correlation
matrix) is not used: it has no statistical definition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize

from kc761tool.core._checks import as_float_array
from kc761tool.errors import CertificateError, SolverError, ValidationError

PSD_TOL = 1e-8
"""Relative tolerance of the F-COV-1/F-COV-2 PSD certificate."""


@dataclass(frozen=True)
class CovarianceEstimate:
    """One covariance estimate with its provenance (F-COV-1/F-COV-2)."""

    matrix: NDArray[np.float64]
    scale: float
    chi2: float
    dof: int
    estimator: str


def fisher_information(
    jacobian: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Analytic Fisher information ``J^T W J`` (F-COV-1).

    ``jacobian`` has shape ``(n_bins, n_params)``; ``sigma`` are the fit
    uncertainties (F-CAL-1) of the same length as the bin axis.
    """
    matrix = as_float_array("jacobian", jacobian, ndim=2)
    errors = as_float_array("sigma", sigma, ndim=1)
    if matrix.shape[0] != errors.size:
        raise ValidationError(
            f"jacobian has {matrix.shape[0]} rows but sigma has {errors.size} entries"
        )
    if np.any(errors <= 0.0):
        raise ValidationError("sigma must be strictly positive")
    weighted = matrix / errors[:, None]
    fisher = weighted.T @ weighted
    return np.asarray(fisher, dtype=np.float64)


def scaled_covariance(
    fisher: NDArray[np.float64],
    *,
    chi2: float,
    dof: int,
) -> CovarianceEstimate:
    """``s**2 = chi2/dof`` scaled covariance with the scale recorded (F-COV-2).

    A non-positive ``dof`` is a :class:`kc761tool.errors.ValidationError`; a
    singular or non-positive-definite Fisher matrix is a
    :class:`kc761tool.errors.SolverError`. No pseudo-inverse fallback exists.
    """
    matrix = as_float_array("fisher", fisher, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"fisher must be square, got shape {matrix.shape}")
    if not np.isfinite(chi2) or chi2 < 0.0:
        raise ValidationError(
            f"chi2 must be finite and non-negative, got {chi2!r}")
    if not isinstance(dof, int) or dof < 1:
        raise ValidationError(f"dof must be a positive int, got {dof!r}")
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        factor = linalg.cho_factor(symmetric, lower=True, check_finite=False)
        inverse = linalg.cho_solve(factor, np.eye(
            matrix.shape[0]), check_finite=False)
    except linalg.LinAlgError as exc:
        raise SolverError(f"fisher is not positive definite: {exc}") from exc
    scale = float(chi2) / float(dof)
    covariance = scale * np.asarray(inverse, dtype=np.float64)
    return CovarianceEstimate(
        matrix=0.5 * (covariance + covariance.T),
        scale=scale,
        chi2=float(chi2),
        dof=int(dof),
        estimator="fisher-x2/dof",
    )


def profile_covariance(
    objective: Callable[[NDArray[np.float64]], float],
    parameters: NDArray[np.float64],
    *,
    profile_points: int,
) -> CovarianceEstimate:
    """Optional profile-covariance diagnostic (F-COV-3).

    For every parameter the objective is profiled at
    ``parameters[i] +- 4 * sigma_i`` with ``profile_points`` points; the
    remaining parameters are re-optimized (deterministic Nelder-Mead) and the
    ``chi2_min + 1`` crossing is bracketed and solved with ``brentq``. The
    widths form the diagonal; the correlations come from the numerical
    Hessian of the objective at the optimum. The result never replaces
    :func:`scaled_covariance`.

    ``dof`` is reported as 0 because the profile diagnostic does not use a
    reduced-chi2 scale (``scale`` is 1.0).
    """
    start = as_float_array("parameters", parameters, ndim=1)
    if start.size < 1:
        raise ValidationError("parameters must not be empty")
    if not isinstance(profile_points, int) or profile_points < 5:
        raise ValidationError(
            f"profile_points must be an int >= 5, got {profile_points!r}")
    chi2_min = _scalar_objective(objective, start)
    hessian = _numerical_hessian(objective, start)
    if np.any(np.diag(hessian) <= 0.0):
        raise SolverError(
            "profile covariance requires a positive definite Hessian")
    widths = np.empty(start.size, dtype=np.float64)
    for index in range(start.size):
        sigma0 = float(np.sqrt(2.0 / hessian[index, index]))
        offsets = np.linspace(-4.0 * sigma0, 4.0 * sigma0, profile_points)
        values = np.array(
            [
                _profiled_objective(objective, start, index,
                                    float(start[index] + offset))
                - (chi2_min + 1.0)
                for offset in offsets
            ]
        )
        widths[index] = 0.5 * (
            _first_crossing(objective, start, index, offsets,
                            values, +1.0, chi2_min + 1.0)
            - _first_crossing(objective, start, index, offsets,
                              values, -1.0, chi2_min + 1.0)
        )
    scale = np.outer(widths, widths)
    fisher = 0.5 * hessian
    try:
        factor = linalg.cho_factor(fisher, lower=True, check_finite=False)
        fisher_inverse = linalg.cho_solve(
            factor, np.eye(fisher.shape[0]), check_finite=False)
    except linalg.LinAlgError as exc:
        raise SolverError(
            f"profile Hessian is not positive definite: {exc}") from exc
    correlation = fisher_inverse / np.sqrt(
        np.outer(np.diag(fisher_inverse), np.diag(fisher_inverse))
    )
    covariance = scale * correlation
    covariance = 0.5 * (covariance + covariance.T)
    return CovarianceEstimate(
        matrix=covariance,
        scale=1.0,
        chi2=chi2_min,
        dof=0,
        estimator="profile",
    )


def _scalar_objective(
    objective: Callable[[NDArray[np.float64]], float], parameters: NDArray[np.float64]
) -> float:
    value = float(objective(parameters))
    if not np.isfinite(value):
        raise SolverError("objective returned a non-finite value")
    return value


def _numerical_hessian(
    objective: Callable[[NDArray[np.float64]], float], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    n_params = parameters.size
    steps = np.maximum(1e-4 * np.maximum(np.abs(parameters), 1.0), 1e-6)
    base = _scalar_objective(objective, parameters)
    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    for i in range(n_params):
        plus_i = parameters.copy()
        minus_i = parameters.copy()
        plus_i[i] += steps[i]
        minus_i[i] -= steps[i]
        hessian[i, i] = (
            _scalar_objective(objective, plus_i)
            - 2.0 * base
            + _scalar_objective(objective, minus_i)
        ) / steps[i] ** 2
        for j in range(i + 1, n_params):
            total = 0.0
            for di, dj, sign in (
                (steps[i], steps[j], 1.0),
                (steps[i], -steps[j], -1.0),
                (-steps[i], steps[j], -1.0),
                (-steps[i], -steps[j], 1.0),
            ):
                point = parameters.copy()
                point[i] += di
                point[j] += dj
                total += sign * _scalar_objective(objective, point)
            hessian[i, j] = total / (4.0 * steps[i] * steps[j])
            hessian[j, i] = hessian[i, j]
    return hessian


def _profiled_objective(
    objective: Callable[[NDArray[np.float64]], float],
    parameters: NDArray[np.float64],
    index: int,
    value: float,
) -> float:
    free = np.array([k for k in range(parameters.size)
                    if k != index], dtype=np.int64)
    if free.size == 0:
        point = parameters.copy()
        point[index] = value
        return _scalar_objective(objective, point)

    def inner(free_values: NDArray[np.float64]) -> float:
        point = parameters.copy()
        point[index] = value
        point[free] = free_values
        return _scalar_objective(objective, point)

    result = optimize.minimize(
        inner,
        parameters[free],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12,
                 "maxiter": 2000, "maxfev": 5000},
    )
    if not result.success and not np.isfinite(result.fun):
        raise SolverError(f"profile re-optimization failed: {result.message}")
    return float(result.fun)


def _first_crossing(
    objective: Callable[[NDArray[np.float64]], float],
    parameters: NDArray[np.float64],
    index: int,
    offsets: NDArray[np.float64],
    values: NDArray[np.float64],
    direction: float,
    target: float,
) -> float:
    """Return the parameter value where the profile reaches ``target``.

    The scan points are visited from the optimum outward on the requested
    side; ``values`` are the already evaluated ``profile - target`` offsets.
    """
    if direction > 0.0:
        order = np.argsort(offsets)
        order = order[offsets[order] > 0.0]
    else:
        order = np.argsort(-offsets)
        order = order[offsets[order] < 0.0]
    previous = 0.0
    for position in order:
        offset = float(offsets[position])
        if values[position] >= 0.0:
            def profile(tau: float) -> float:
                return (
                    _profiled_objective(
                        objective, parameters, index, float(
                            parameters[index] + tau)
                    )
                    - target
                )

            root = optimize.brentq(
                profile,
                previous,
                offset,
                xtol=1e-12,
                rtol=1e-12,
                maxiter=200,
            )
            return float(parameters[index] + root)
        previous = offset
    raise SolverError(
        "profile crossing chi2_min + 1 not found within four sigma")


def verify_covariance_psd(
    estimate: CovarianceEstimate,
    *,
    strict: bool,
    formula_id: str = "F-COV-2",
    atol: float = PSD_TOL,
) -> float:
    """F-COV-1/F-COV-2 certificate: symmetric with a non-negative spectrum."""
    matrix = as_float_array("covariance", estimate.matrix, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"covariance must be square, got shape {matrix.shape}")
    asymmetry = float(np.max(np.abs(matrix - matrix.T))
                      ) if matrix.size else 0.0
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (matrix + matrix.T)) if matrix.size else np.zeros(1)
    smallest = float(np.min(eigenvalues))
    scale = max(1.0, float(np.max(np.abs(eigenvalues)))
                if eigenvalues.size else 1.0)
    if strict and (asymmetry > atol * scale or smallest < -atol * scale):
        raise CertificateError(
            formula_id,
            f"covariance PSD certificate failed: min eigenvalue {smallest:.3g}, "
            f"asymmetry {asymmetry:.3g}",
        )
    return smallest
