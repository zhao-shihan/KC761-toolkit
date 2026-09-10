"""Non-negative regularized unfolding: objective, solver and KKT certificate.

Formula IDs (docs/derivations.md): F-SOLVE-1 .. F-SOLVE-3.

* F-SOLVE-1: Tikhonov objective ``chi2 + alpha * ||D mu||**2`` with ``mu >= 0``.
  ``alpha`` is mandatory at the CLI (D-45) and dimensionless (D-46): the
  difference operator is scaled as ``D_tilde = D . diag(sqrt(diag(R^T W R)))``,
  so it acts on the signal-to-noise normalized solution and the penalty
  carries no counts (see docs/derivations.md F-SOLVE-1).
* F-SOLVE-2: self-implemented Cholesky active-set solver (banded when the
  normal matrix is banded, dense otherwise).
* F-SOLVE-3: the KKT certificate is measured in units of the data-gradient
  scale (relative tolerance ``KKT_TOL``), so it does not depend on the count
  normalization of the problem.

The SNIP peak mask is removed (D-74); ``D`` is only the finite-difference
operator below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.core._checks import as_float_array, check_response_matrix, require_same_length
from kc761.core._linalg import solve_spd
from kc761.errors import CertificateError, SolverError, ValidationError

DEFAULT_DIFFERENCE_ORDER = 2
KKT_TOL = 1e-6
"""F-SOLVE-3 relative tolerance for the reduced gradient and complementarity."""

MAX_ITERATIONS_FACTOR = 10
"""Maximum active-set iterations: ``factor * n + 100``."""


@dataclass(frozen=True)
class RegularizationSpec:
    """Dimensionless Tikhonov configuration (F-SOLVE-1)."""

    alpha: float
    difference_order: int = DEFAULT_DIFFERENCE_ORDER

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValidationError(f"alpha must be positive and finite, got {self.alpha!r}")
        if self.difference_order not in (1, 2):
            raise ValidationError(
                f"difference_order must be 1 or 2, got {self.difference_order!r}"
            )


@dataclass(frozen=True)
class KktCertificate:
    """F-SOLVE-3 certificate for one solution (relative metrics)."""

    max_negative_reduced_gradient: float
    complementarity: float
    n_active: int
    iterations: int
    converged: bool

    @property
    def ok(self) -> bool:
        return (
            self.converged
            and self.max_negative_reduced_gradient <= KKT_TOL
            and self.complementarity <= KKT_TOL
        )


@dataclass(frozen=True)
class UnfoldSolution:
    """Solution vector, refolded prediction and certificate (F-SOLVE-2/3)."""

    mu: NDArray[np.float64]
    refolded: NDArray[np.float64]
    objective: float
    certificate: KktCertificate


def difference_operator(n_bins: int, order: int) -> sparse.csr_matrix:
    """Finite-difference operator ``D`` of the requested order (F-SOLVE-1)."""
    if order not in (1, 2):
        raise ValidationError(f"difference_order must be 1 or 2, got {order!r}")
    if n_bins < 1:
        raise ValidationError(f"n_bins must be >= 1, got {n_bins!r}")
    n_rows = max(n_bins - order, 0)
    if n_rows == 0:
        return sparse.csr_matrix((0, n_bins), dtype=np.float64)
    rows = np.repeat(np.arange(n_rows), order + 1)
    cols = np.concatenate([np.arange(start, start + order + 1) for start in range(n_rows)])
    coefficients = {
        1: np.array([-1.0, 1.0]),
        2: np.array([1.0, -2.0, 1.0]),
    }[order]
    values = np.tile(coefficients, n_rows)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n_rows, n_bins))


def normal_equations(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    regularization: RegularizationSpec,
) -> tuple[sparse.csr_matrix, NDArray[np.float64], NDArray[np.float64]]:
    """Half Hessian ``Hc`` and gradient offset ``b`` of the objective (F-SOLVE-1).

    The objective is ``chi2 + alpha * ||D_tilde mu||**2`` with
    ``chi2 = ||(R mu - y) / sigma||**2``; its half gradient is ``Hc mu - b``.
    Returns ``(Hc, b, penalty_scale)`` where ``penalty_scale = sqrt(diag(A))``
    and ``A = R^T W R``.
    """
    matrix = check_response_matrix(response)
    y = as_float_array("spectrum", spectrum, ndim=1)
    errors = as_float_array("sigma", sigma, ndim=1)
    require_same_length("spectrum/sigma", y, errors)
    if np.any(errors <= 0.0):
        raise ValidationError("sigma must be strictly positive")
    n_bins = matrix.shape[1]
    if y.size != matrix.shape[0]:
        raise ValidationError(
            f"spectrum has {y.size} bins, response has {matrix.shape[0]} rows"
        )
    weights = 1.0 / errors**2
    weighted = matrix.T @ sparse.diags(weights)  # n x m
    hessian = (weighted @ matrix).tocsr()
    gradient = np.asarray(weighted @ y, dtype=np.float64)
    diagonal = hessian.diagonal()
    if np.any(diagonal < 0.0):
        raise SolverError("R^T W R has a negative diagonal entry")
    penalty_scale = np.sqrt(diagonal)
    difference = difference_operator(n_bins, regularization.difference_order)
    scaled = difference @ sparse.diags(penalty_scale)
    hessian = (hessian + regularization.alpha * (scaled.T @ scaled)).tocsr()
    return hessian, gradient, penalty_scale


def solve_nonnegative(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    regularization: RegularizationSpec,
    *,
    strict: bool = False,
    normal: tuple[sparse.csr_matrix, NDArray[np.float64], NDArray[np.float64]]
    | None = None,
) -> UnfoldSolution:
    """Solve the non-negative Tikhonov problem (F-SOLVE-2).

    Strict mode verifies the F-SOLVE-3 certificate and raises
    :class:`kc761.errors.CertificateError` on failure; basic shape and
    finiteness validation runs in every mode. A non-converged active set is a
    :class:`kc761.errors.SolverError` in every mode.

    ``normal`` optionally injects the tuple returned by
    :func:`normal_equations` (D-150): the unfolding layer already needs that
    half-Hessian for F-UNC, so passing it here avoids a second identical
    factorization. When omitted, the equations are built internally.
    """
    if normal is None:
        hessian, gradient, penalty_scale = normal_equations(
            response, spectrum, sigma, regularization
        )
    else:
        hessian, gradient, penalty_scale = normal
        checked = check_response_matrix(response)
        y = as_float_array("spectrum", spectrum, ndim=1)
        if y.size != checked.shape[0] or gradient.size != checked.shape[1]:
            raise ValidationError(
                "injected normal equations do not match the response/spectrum"
            )
    n_bins = gradient.size
    mu, iterations, converged = _active_set(hessian, gradient, n_bins)
    certificate = _certificate(hessian, gradient, mu, iterations, converged)
    if not converged:
        raise SolverError(
            f"active set did not converge after {iterations} iterations "
            f"(complementarity {certificate.complementarity:.3g})"
        )
    if strict and not certificate.ok:
        raise CertificateError(
            "F-SOLVE-3",
            "KKT certificate failed: "
            f"max_neg_reduced_gradient={certificate.max_negative_reduced_gradient:.3g}, "
            f"complementarity={certificate.complementarity:.3g}",
        )
    matrix = response.tocsr().astype(np.float64)
    refolded = np.asarray(matrix @ mu, dtype=np.float64)
    return UnfoldSolution(
        mu=mu,
        refolded=refolded,
        objective=_objective(matrix, spectrum, sigma, mu, regularization, penalty_scale),
        certificate=certificate,
    )


def verify_kkt(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    regularization: RegularizationSpec,
    mu: NDArray[np.float64],
) -> KktCertificate:
    """Evaluate the KKT certificate for a given ``mu`` (F-SOLVE-3)."""
    hessian, gradient, _ = normal_equations(response, spectrum, sigma, regularization)
    solution = as_float_array("mu", mu, ndim=1)
    if solution.size != gradient.size:
        raise ValidationError(
            f"mu has {solution.size} bins, response has {gradient.size} columns"
        )
    return _certificate(hessian, gradient, solution, 0, True)


def _certificate(
    hessian: sparse.csr_matrix,
    gradient: NDArray[np.float64],
    mu: NDArray[np.float64],
    iterations: int,
    converged: bool,
) -> KktCertificate:
    residual = np.asarray(hessian @ mu, dtype=np.float64) - gradient
    gradient_scale = max(1.0, float(np.max(np.abs(gradient))) if gradient.size else 1.0)
    active = mu <= 0.0
    negative = np.where(active, -residual, 0.0)
    max_negative = float(np.max(negative)) if negative.size else 0.0
    complementarity = float(np.max(np.abs(mu * residual))) if mu.size else 0.0
    mu_scale = max(1.0, float(np.max(np.abs(mu))) if mu.size else 1.0)
    finite = bool(np.isfinite(mu).all() and np.isfinite(residual).all())
    return KktCertificate(
        max_negative_reduced_gradient=max_negative / gradient_scale,
        complementarity=complementarity / (gradient_scale * mu_scale),
        n_active=int(np.count_nonzero(active)),
        iterations=int(iterations),
        converged=bool(converged and finite),
    )


def _active_set(
    hessian: sparse.csr_matrix,
    gradient: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.float64], int, bool]:
    """Lawson-Hanson active set for ``min 1/2 mu^T H mu - b^T mu``, ``mu >= 0``."""
    mu = np.zeros(n_bins, dtype=np.float64)
    diagonal = np.asarray(hessian.diagonal(), dtype=np.float64)
    unbounded = (diagonal <= 0.0) & (np.abs(gradient) > 0.0)
    if np.any(unbounded):
        worst = int(np.argmax(np.abs(gradient)))
        raise SolverError(
            f"objective is unbounded below: bin {worst} has no curvature and a "
            "non-zero gradient"
        )
    free = diagonal > 0.0
    max_iterations = MAX_ITERATIONS_FACTOR * n_bins + 100
    threshold = KKT_TOL * _gradient_scale(gradient)
    for iteration in range(1, max_iterations + 1):
        if np.any(free):
            proposal = _solve_reduced(hessian, gradient, free)
            candidate = np.zeros(n_bins, dtype=np.float64)
            candidate[free] = proposal
            if np.all(proposal > 0.0):
                mu = candidate
            else:
                # Step from the feasible mu towards the candidate until a
                # variable reaches the boundary, then move exactly the
                # blocking variables into the active set.
                free_indices = np.flatnonzero(free)
                direction = candidate[free_indices] - mu[free_indices]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.where(
                        direction < 0.0, mu[free_indices] / (-direction), np.inf
                    )
                limiting = float(np.min(ratios)) if ratios.size else np.inf
                step = min(1.0, max(limiting, 0.0))
                mu = mu + step * (candidate - mu)
                if limiting <= 1.0 and np.any(direction < 0.0):
                    negative = free_indices[direction < 0.0]
                    boundary = negative[ratios[direction < 0.0] <= limiting + 1e-12]
                else:
                    boundary = np.flatnonzero(free & (mu <= 0.0))
                if boundary.size:
                    free[boundary] = False
                    mu[boundary] = 0.0
                continue
        residual = np.asarray(hessian @ mu, dtype=np.float64) - gradient
        blocked = np.flatnonzero(~free & (residual < -threshold))
        if blocked.size == 0:
            return mu, iteration, True
        free[blocked[int(np.argmin(residual[blocked]))]] = True
    return mu, max_iterations, False


def _gradient_scale(gradient: NDArray[np.float64]) -> float:
    return max(1.0, float(np.max(np.abs(gradient))) if gradient.size else 1.0)


def _solve_reduced(
    hessian: sparse.csr_matrix, gradient: NDArray[np.float64], free: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Solve ``H[free, free] x = b[free]`` with the shared SPD policy."""
    sub = hessian[free][:, free]
    if sub.shape[0] == 0:
        raise SolverError("no free variables to solve for")
    return solve_spd(sub, gradient[free])


def _objective(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    mu: NDArray[np.float64],
    regularization: RegularizationSpec,
    penalty_scale: NDArray[np.float64],
) -> float:
    residual = (np.asarray(response @ mu, dtype=np.float64) - spectrum) / sigma
    chi2 = float(residual @ residual)
    difference = difference_operator(mu.size, regularization.difference_order)
    if difference.shape[0] == 0:
        penalty = 0.0
    else:
        curvature = difference @ (penalty_scale * mu)
        penalty = float(curvature @ curvature)
    return chi2 + regularization.alpha * penalty
