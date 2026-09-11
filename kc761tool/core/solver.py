"""Non-negative regularized unfolding: objective, solver and KKT certificate.

Formula IDs (docs/derivations.md): F-SOLVE-1 .. F-SOLVE-6.

* F-SOLVE-1: Tikhonov objective ``chi2 + alpha * ||D_tilde mu||**2`` with
  ``mu >= 0``. ``alpha`` is mandatory at the CLI (D-45) and dimensionless
  (D-80): the difference operator is scaled on the right,
  ``D_tilde = D . diag(sqrt(diag(R^T W R)))``.
* F-SOLVE-2: self-implemented Cholesky active-set solver (banded when the
  normal matrix is banded, dense otherwise).
* F-SOLVE-3: the KKT certificate is measured in units of the data-gradient
  scale (relative tolerance ``KKT_TOL``), so it does not depend on the count
  normalization of the problem.
* F-SOLVE-4/5/6 (D-154..D-161): an optional, default-on SNIP peak mask. The
  baseline (F-SOLVE-4) and the resolution-matched significance (F-SOLVE-5)
  build a fixed diagonal weight ``W``; the penalty operator becomes
  ``D_tilde' = W**0.5 D W**0.5 . diag(sqrt(diag(A)))`` (F-SOLVE-6). The mask is
  a function of the measured spectrum only, so the problem stays convex and the
  F-SOLVE-3 certificate is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761tool.core._checks import as_float_array, check_response_matrix, require_same_length
from kc761tool.core._linalg import (
    factor_dense_spd,
    prefer_dense_factor,
    solve_spd,
    weighted_normal_and_rhs,
)
from kc761tool.errors import CertificateError, SolverError, ValidationError

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


DEFAULT_SNIP_THRESHOLD_SIGMA = 5.0
DEFAULT_SNIP_PROTECT_SIGMA = 2.0
DEFAULT_SNIP_FLOOR = 0.1
DEFAULT_SNIP_MAX_ITERATIONS = 8
SNIP_FILTER_SIGMA = 3.0
"""Matched-filter support in resolution widths (F-SOLVE-5)."""


@dataclass(frozen=True)
class SnipSettings:
    """SNIP peak-mask configuration (F-SOLVE-4/5, D-156/D-157).

    The defaults are provisional and are refined by the synthetic parameter
    study recorded under F-SOLVE-6 in docs/derivations.md; the realized values are
    always written into the product meta (D-160).
    """

    enabled: bool = True
    threshold_sigma: float = DEFAULT_SNIP_THRESHOLD_SIGMA
    protect_sigma: float = DEFAULT_SNIP_PROTECT_SIGMA
    floor: float = DEFAULT_SNIP_FLOOR
    iterations: int | None = None
    max_iterations: int = DEFAULT_SNIP_MAX_ITERATIONS

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold_sigma) or self.threshold_sigma <= 0.0:
            raise ValidationError(
                f"snip threshold_sigma must be positive and finite, got {self.threshold_sigma!r}"
            )
        if not np.isfinite(self.protect_sigma) or self.protect_sigma < 0.0:
            raise ValidationError(
                f"snip protect_sigma must be non-negative and finite, got {self.protect_sigma!r}"
            )
        if not np.isfinite(self.floor) or not 0.0 <= self.floor <= 1.0:
            raise ValidationError(f"snip floor must lie in [0, 1], got {self.floor!r}")
        if self.iterations is not None and (
            not isinstance(self.iterations, int) or self.iterations < 1
        ):
            raise ValidationError(
                f"snip iterations must be a positive integer or None, got {self.iterations!r}"
            )
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValidationError(
                f"snip max_iterations must be a positive integer, got {self.max_iterations!r}"
            )

    def resolved_iterations(self, fwhm_bins: float) -> int:
        """F-SOLVE-4 iteration count: resolution-derived unless overridden."""
        if self.iterations is not None:
            return int(self.iterations)
        if not np.isfinite(fwhm_bins) or fwhm_bins <= 0.0:
            return 1
        half = int(round(0.5 * float(fwhm_bins)))
        return int(np.clip(half, 1, self.max_iterations))


@dataclass(frozen=True)
class SnipMask:
    """Fixed peak mask and its construction statistics (F-SOLVE-5)."""

    weights: NDArray[np.float64]
    baseline: NDArray[np.float64]
    iterations: int
    clipped_count: int
    clipped_first: int
    clipped_last: int
    n_candidates: int
    n_protected: int


def snip_baseline(
    values: NDArray[np.float64], iterations: int
) -> NDArray[np.float64]:
    """SNIP LLS baseline of a non-negative spectrum (F-SOLVE-4)."""
    y = as_float_array("snip values", values, ndim=1)
    if np.any(y < 0.0):
        raise ValidationError("SNIP baseline requires non-negative values")
    count = int(iterations)
    if count < 1:
        raise ValidationError(f"SNIP iterations must be >= 1, got {iterations!r}")
    transformed = np.log(np.log(np.sqrt(y + 1.0) + 1.0) + 1.0)
    for offset in range(1, count + 1):
        left = transformed.copy()
        left[offset:] = transformed[:-offset]
        right = transformed.copy()
        right[:-offset] = transformed[offset:]
        transformed = np.minimum(transformed, 0.5 * (left + right))
    baseline = (np.exp(np.exp(transformed) - 1.0) - 1.0) ** 2 - 1.0
    return np.maximum(baseline, 0.0)


def snip_peak_mask(
    values: NDArray[np.float64],
    sigma: NDArray[np.float64],
    resolution_sigma_kev: NDArray[np.float64],
    bin_width_kev: NDArray[np.float64],
    settings: SnipSettings,
) -> SnipMask:
    """Resolution-matched peak mask on the measured spectrum (F-SOLVE-5).

    ``values`` is the (channel-domain) measured spectrum, ``sigma`` its
    per-bin uncertainty, ``resolution_sigma_kev`` the detector width and
    ``bin_width_kev`` the per-bin energy width. All arrays share the length of
    the mapped primary axis (D-121 makes the mapping 1:1; the unfold layer
    enforces that before calling this function).
    """
    y_raw = as_float_array("snip spectrum", values, ndim=1)
    errors = as_float_array("snip sigma", sigma, ndim=1)
    resol = as_float_array("snip resolution", resolution_sigma_kev, ndim=1)
    widths = as_float_array("snip bin widths", bin_width_kev, ndim=1)
    require_same_length("snip spectrum/sigma", y_raw, errors)
    require_same_length("snip spectrum/resolution", y_raw, resol)
    require_same_length("snip spectrum/bin widths", y_raw, widths)
    if np.any(errors <= 0.0) or np.any(widths <= 0.0) or np.any(resol <= 0.0):
        raise ValidationError("SNIP inputs require positive sigma, widths and resolution")

    negative = y_raw < 0.0
    clipped = np.maximum(y_raw, 0.0)
    clipped_count = int(np.count_nonzero(negative))
    if clipped_count:
        indices = np.flatnonzero(negative)
        clipped_first = int(indices[0])
        clipped_last = int(indices[-1])
    else:
        clipped_first = -1
        clipped_last = -1

    width_in_bins = resol / widths
    middle = y_raw.size // 2
    fwhm_bins = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(width_in_bins[middle])
    iterations = settings.resolved_iterations(fwhm_bins)
    baseline = snip_baseline(clipped, iterations)
    residual = clipped - baseline

    size = clipped.size
    significance = np.zeros(size, dtype=np.float64)
    for index in range(size):
        half = max(1, int(np.ceil(SNIP_FILTER_SIGMA * float(width_in_bins[index]))))
        low = max(0, index - half)
        high = min(size, index + half + 1)
        offsets = np.arange(low, high, dtype=np.float64) - float(index)
        kernel = np.exp(-0.5 * (offsets / float(width_in_bins[index])) ** 2)
        total = float(kernel.sum())
        if total <= 0.0:
            continue
        kernel /= total
        matched = float(kernel @ residual[low:high])
        variance = float(kernel @ (kernel * errors[low:high] ** 2))
        significance[index] = matched / np.sqrt(variance) if variance > 0.0 else 0.0

    candidate = significance >= settings.threshold_sigma
    local = np.zeros(size, dtype=bool)
    if size == 1:
        local[0] = bool(candidate[0])
    elif size > 1:
        local[0] = bool(candidate[0] and significance[0] >= significance[1])
        local[-1] = bool(candidate[-1] and significance[-1] >= significance[-2])
        local[1:-1] = candidate[1:-1] & (significance[1:-1] >= significance[:-2]) & (
            significance[1:-1] >= significance[2:]
        )
    protected = np.zeros(size, dtype=bool)
    centers = np.flatnonzero(local)
    for index in centers:
        half = int(np.ceil(settings.protect_sigma * float(width_in_bins[index])))
        protected[max(0, index - half) : min(size, index + half + 1)] = True
    weights = np.where(protected, settings.floor, 1.0).astype(np.float64)
    return SnipMask(
        weights=weights,
        baseline=baseline,
        iterations=iterations,
        clipped_count=clipped_count,
        clipped_first=clipped_first,
        clipped_last=clipped_last,
        n_candidates=int(centers.size),
        n_protected=int(np.count_nonzero(protected)),
    )


def verify_snip_mask(
    settings: SnipSettings,
    mask: NDArray[np.float64],
    *,
    values: NDArray[np.float64],
    sigma: NDArray[np.float64],
    resolution_sigma_kev: NDArray[np.float64],
    bin_width_kev: NDArray[np.float64],
) -> SnipMask:
    """F-SOLVE-6 certificate: the mask equals the recomputed construction."""
    expected = snip_peak_mask(values, sigma, resolution_sigma_kev, bin_width_kev, settings)
    provided = as_float_array("snip mask", mask, ndim=1)
    if provided.shape != expected.weights.shape or not np.array_equal(
        provided, expected.weights
    ):
        raise CertificateError(
            "F-SOLVE-6",
            "the SNIP mask does not match the mask recomputed from the recorded "
            "spectrum and settings",
        )
    return expected


def _check_mask(mask: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    weights = as_float_array("snip mask", mask, ndim=1)
    if weights.size != n_bins:
        raise ValidationError(
            f"snip mask has {weights.size} entries, expected {n_bins}"
        )
    if not np.isfinite(weights).all():
        raise ValidationError("snip mask contains non-finite weights")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValidationError("snip mask weights must lie in [0, 1]")
    return weights


def _masked_difference(
    difference: sparse.csr_matrix, mask: NDArray[np.float64]
) -> sparse.csr_matrix:
    """``D' = diag(rho**0.5) D`` with per-row stencil weights (F-SOLVE-6).

    ``rho_r`` is the product of the mask weights over the finite-difference
    stencil of row ``r`` (``order + 1`` columns), so a protected peak bin
    relaxes every difference row that touches it. ``D'^T D'`` stays symmetric
    positive semidefinite and banded, and no rectangular ``W D W`` product is
    needed.
    """
    weights = _check_mask(mask, difference.shape[1])
    n_rows = difference.shape[0]
    if n_rows == 0:
        return difference
    order = difference.shape[1] - n_rows
    row_weight = np.ones(n_rows, dtype=np.float64)
    for offset in range(order + 1):
        row_weight = row_weight * weights[offset : offset + n_rows]
    row_weight = np.maximum(row_weight, 0.0)
    return (sparse.diags(np.sqrt(row_weight)) @ difference).tocsr()


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


def normal_equations(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    regularization: RegularizationSpec,
    *,
    mask: NDArray[np.float64] | None = None,
) -> tuple[sparse.csr_matrix, NDArray[np.float64], NDArray[np.float64]]:
    """Half Hessian ``Hc`` and gradient offset ``b`` of the objective (F-SOLVE-1).

    The objective is ``chi2 + alpha * ||D_tilde' mu||**2`` with
    ``chi2 = ||(R mu - y) / sigma||**2``; its half gradient is ``Hc mu - b``.
    Returns ``(Hc, b, penalty_scale)`` where ``penalty_scale = sqrt(diag(A))``
    and ``A = R^T W R``. With ``mask`` the penalty operator is
    ``D_tilde' = W**0.5 D W**0.5 . diag(penalty_scale)`` (F-SOLVE-6).
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
    hessian, gradient = weighted_normal_and_rhs(matrix, weights, y)
    assert gradient is not None
    diagonal = hessian.diagonal()
    if np.any(diagonal < 0.0):
        raise SolverError("R^T W R has a negative diagonal entry")
    penalty_scale = np.sqrt(diagonal)
    difference = difference_operator(n_bins, regularization.difference_order)
    if mask is not None:
        difference = _masked_difference(difference, mask)
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
    mask: NDArray[np.float64] | None = None,
    normal: tuple[sparse.csr_matrix, NDArray[np.float64], NDArray[np.float64]]
    | None = None,
) -> UnfoldSolution:
    """Solve the non-negative Tikhonov problem (F-SOLVE-2).

    Strict mode verifies the F-SOLVE-3 certificate and raises
    :class:`kc761tool.errors.CertificateError` on failure; basic shape and
    finiteness validation runs in every mode. A non-converged active set is a
    :class:`kc761tool.errors.SolverError` in every mode.

    ``normal`` optionally injects the tuple returned by
    :func:`normal_equations` (D-150): the unfolding layer already needs that
    half-Hessian for F-UNC, so passing it here avoids a second identical
    factorization. When omitted, the equations are built internally.
    """
    if normal is None:
        hessian, gradient, penalty_scale = normal_equations(
            response, spectrum, sigma, regularization, mask=mask
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
        objective=_objective(
            matrix, spectrum, sigma, mu, regularization, penalty_scale, mask=mask
        ),
        certificate=certificate,
    )


def verify_kkt(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    regularization: RegularizationSpec,
    mu: NDArray[np.float64],
    *,
    mask: NDArray[np.float64] | None = None,
) -> KktCertificate:
    """Evaluate the KKT certificate for a given ``mu`` (F-SOLVE-3)."""
    hessian, gradient, _ = normal_equations(
        response, spectrum, sigma, regularization, mask=mask
    )
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
    # Densify a dense-stored Hessian once: the per-iteration reduced system is
    # then a ``numpy.ix_`` slice instead of sparse fancy-indexing, which
    # dominated the active set on the 2048 problem (D-172). The factorization
    # policy itself stays in ``kc761tool.core._linalg``.
    dense: NDArray[np.float64] | None = None
    if prefer_dense_factor(hessian):
        dense = np.asarray(hessian.toarray(), dtype=np.float64)
        dense = 0.5 * (dense + dense.T)
        diagonal = np.diag(dense).copy()
    else:
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
            proposal = _solve_reduced(hessian, dense, gradient, free)
            candidate = np.zeros(n_bins, dtype=np.float64)
            candidate[free] = proposal
            if np.all(proposal > 0.0):
                mu = candidate
            else:
                # Step from the feasible mu toward the candidate until a
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
        residual = (
            dense @ mu if dense is not None else np.asarray(hessian @ mu)
        ) - gradient
        blocked = np.flatnonzero(~free & (residual < -threshold))
        if blocked.size == 0:
            return mu, iteration, True
        free[blocked[int(np.argmin(residual[blocked]))]] = True
    return mu, max_iterations, False


def _gradient_scale(gradient: NDArray[np.float64]) -> float:
    return max(1.0, float(np.max(np.abs(gradient))) if gradient.size else 1.0)


def _solve_reduced(
    hessian: sparse.csr_matrix,
    dense: NDArray[np.float64] | None,
    gradient: NDArray[np.float64],
    free: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Solve ``H[free, free] x = b[free]`` with the shared SPD policy."""
    if dense is not None:
        sub = dense[np.ix_(free, free)]
        if sub.shape[0] == 0:
            raise SolverError("no free variables to solve for")
        return factor_dense_spd(sub).solve(gradient[free])
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
    *,
    mask: NDArray[np.float64] | None = None,
) -> float:
    residual = (np.asarray(response @ mu, dtype=np.float64) - spectrum) / sigma
    chi2 = float(residual @ residual)
    difference = difference_operator(mu.size, regularization.difference_order)
    if mask is not None:
        difference = _masked_difference(difference, mask)
    if difference.shape[0] == 0:
        penalty = 0.0
    else:
        curvature = difference @ (penalty_scale * mu)
        penalty = float(curvature @ curvature)
    return chi2 + regularization.alpha * penalty
