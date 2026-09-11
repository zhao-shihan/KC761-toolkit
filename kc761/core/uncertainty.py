"""Uncertainty propagation with the strict statistical/systematic split.

Formula IDs (docs/derivations.md): F-UNC-1 .. F-UNC-3.

* F-UNC-1: statistical propagation of the data covariance through the same
  normal matrix that solved the problem (``H`` from F-SOLVE-1), so value and
  error cannot drift apart:
  ``Cov(mu) = H**-1 R^T W Sigma_stat W R H**-1`` with ``Sigma_stat`` the pure
  data-statistical variance (``max(stat, 1)``, F-CAL-1).
* F-UNC-2: systematic propagation from the calibration covariance
  (via the F-RESP-4 Jacobian), the simulation MC multinomial term and the
  data-side ``syst_frac`` term.
* F-UNC-3: strict decomposition ``total**2 = stat**2 + syst**2``; ``stat`` is
  pure data statistics and ``syst`` carries every systematic contribution
  (D-50). The identity is a strict-mode certificate.

Every band component records its source name and formula ID so a product can
be audited without re-deriving the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.core._checks import as_float_array, check_response_matrix, require_same_length
from kc761.core._linalg import factor_spd, solve_spd, weighted_normal
from kc761.core.response import ResponseMatrix
from kc761.errors import CertificateError, SolverError, ValidationError

BandKind = Literal["stat", "syst"]

DEFAULT_SYST_FRAC = 0.05
"""Default data-side fractional systematic (F-CAL-1/F-UNF-2, D-48).

Single source for both the calibration and the unfolding weight defaults.
"""

DECOMPOSITION_TOL = 1e-9
"""Relative tolerance of the F-UNC-3 identity."""

MC_BLOCK_COLUMNS = 128
"""Free-set columns solved per block in the streaming F-UNC-2 MC term (D-151)."""


@dataclass(frozen=True)
class BandComponent:
    """One additive-in-quadrature contribution to an uncertainty band."""

    name: str
    kind: BandKind
    formula_id: str
    sigma: NDArray[np.float64]


@dataclass(frozen=True)
class UncertaintyBands:
    """Strictly decomposed 1-sigma bands (F-UNC-3)."""

    sigma_stat: NDArray[np.float64]
    sigma_syst: NDArray[np.float64]
    sigma_total: NDArray[np.float64]
    components: tuple[BandComponent, ...]


def propagate_statistical(
    response: sparse.csr_matrix,
    spectrum: NDArray[np.float64],
    sigma: NDArray[np.float64],
    mu: NDArray[np.float64],
    *,
    stat_variance: NDArray[np.float64],
    fisher: NDArray[np.float64] | sparse.spmatrix | None = None,
) -> tuple[NDArray[np.float64], tuple[BandComponent, ...]]:
    """Statistical band and its components for one solution (F-UNC-1).

    ``sigma`` are the fit uncertainties (F-CAL-1) that define the weights
    ``W``; ``stat_variance`` is the pure data-statistical variance per channel.
    ``fisher`` is the full half-Hessian ``H`` of the solved problem (F-SOLVE-1,
    including the regularization term); when it is ``None`` the data-only
    ``R^T W R`` is used and the caller accepts the documented difference.
    Variables that are active in ``mu`` are kept at zero: the sensitivity is
    the reduced free-set system ``H_FF**-1`` (D-86).
    """
    matrix = check_response_matrix(response)
    y = as_float_array("spectrum", spectrum, ndim=1)
    errors = as_float_array("sigma", sigma, ndim=1)
    solution = as_float_array("mu", mu, ndim=1)
    stat_var = as_float_array("stat_variance", stat_variance, ndim=1)
    require_same_length("spectrum/sigma", y, errors)
    require_same_length("spectrum/stat_variance", y, stat_var)
    if y.size != matrix.shape[0]:
        raise ValidationError("spectrum length does not match the response rows")
    if solution.size != matrix.shape[1]:
        raise ValidationError("mu length does not match the response columns")
    if np.any(errors <= 0.0):
        raise ValidationError("sigma must be strictly positive")
    if np.any(stat_var < 0.0):
        raise ValidationError("stat_variance must be non-negative")
    hessian = _hessian(matrix, errors, fisher)
    weighted_response = matrix.T @ sparse.diags(1.0 / errors**2)
    sensitivity = _reduced_columns(hessian, solution, weighted_response.toarray())
    variance = (sensitivity**2) @ stat_var
    variance = np.maximum(variance, 0.0)
    band = np.sqrt(variance)
    component = BandComponent(
        name="data_statistical",
        kind="stat",
        formula_id="F-UNC-1",
        sigma=band,
    )
    return band, (component,)


def propagate_systematic(
    response: sparse.csr_matrix,
    mu: NDArray[np.float64],
    *,
    spectrum: NDArray[np.float64],
    sigma_fit: NDArray[np.float64],
    calib_jacobian: list[sparse.csr_matrix],
    calib_covariance: NDArray[np.float64],
    mc_variance: NDArray[np.float64] | None = None,
    syst_frac: float = 0.0,
    fisher: NDArray[np.float64] | sparse.spmatrix | None = None,
) -> tuple[NDArray[np.float64], tuple[BandComponent, ...]]:
    """Systematic band and its components for one solution (F-UNC-2).

    ``calib_jacobian`` is the list of ``dR/dq_k`` matrices from F-RESP-4 and
    ``calib_covariance`` their covariance. ``mc_variance`` is the
    per-primary-bin variance contribution from the simulation MC term
    (see :func:`simulation_mc_variance`); it is a variance, not a sigma.
    ``syst_frac`` is the data-side fractional systematic (F-CAL-1). Active
    variables are kept at zero through the reduced free-set system (D-86).
    """
    matrix = check_response_matrix(response)
    solution = as_float_array("mu", mu, ndim=1)
    y = as_float_array("spectrum", spectrum, ndim=1)
    errors = as_float_array("sigma_fit", sigma_fit, ndim=1)
    require_same_length("spectrum/sigma_fit", y, errors)
    if not np.isfinite(syst_frac) or syst_frac < 0.0:
        raise ValidationError(f"syst_frac must be finite and >= 0, got {syst_frac!r}")
    if y.size != matrix.shape[0] or solution.size != matrix.shape[1]:
        raise ValidationError("spectrum or mu shape does not match the response")
    if np.any(errors <= 0.0):
        raise ValidationError("sigma_fit must be strictly positive")
    covariance = as_float_array("calib_covariance", calib_covariance, ndim=2)
    n_params = covariance.shape[0]
    if covariance.shape[1] != n_params:
        raise ValidationError("calib_covariance must be square")
    if len(calib_jacobian) != n_params:
        raise ValidationError(
            f"calib_jacobian has {len(calib_jacobian)} matrices, "
            f"calib_covariance is {n_params}x{n_params}"
        )
    for jacobian in calib_jacobian:
        if jacobian.shape != matrix.shape:
            raise ValidationError(
                f"calib_jacobian entry has shape {jacobian.shape}, expected {matrix.shape}"
            )

    hessian = _hessian(matrix, errors, fisher)
    weights = 1.0 / errors**2
    weighted_response = matrix.T @ sparse.diags(weights)
    residual = np.asarray(matrix @ solution, dtype=np.float64) - y
    sensitivity = _reduced_columns(hessian, solution, weighted_response.toarray())

    components: list[BandComponent] = []
    variance = np.zeros_like(solution)

    if syst_frac > 0.0:
        data_side = syst_frac * y
        data_variance = (sensitivity**2) @ (data_side**2)
        variance += data_variance
        components.append(
            BandComponent(
                name="data_syst_frac",
                kind="syst",
                formula_id="F-UNC-2",
                sigma=np.sqrt(np.maximum(data_variance, 0.0)),
            )
        )

    if n_params:
        weighted_residual = weights * residual
        columns = np.empty((solution.size, n_params), dtype=np.float64)
        for index, jacobian in enumerate(calib_jacobian):
            gradient = np.asarray(jacobian.T @ weighted_residual, dtype=np.float64) + np.asarray(
                weighted_response @ (jacobian @ solution), dtype=np.float64
            )
            columns[:, index] = _reduced_columns(
                hessian, solution, -gradient[:, None]
            )[:, 0]
        calibration_variance = np.einsum(
            "ik,kl,il->i", columns, covariance, columns, optimize=True
        )
        scale = np.maximum(1.0, np.abs(calibration_variance))
        if np.any(calibration_variance < -1e-9 * scale):
            worst = float(np.min(calibration_variance))
            raise ValidationError(
                f"calibration covariance produced a negative variance ({worst:.3g}); "
                "the input covariance is not positive semi-definite"
            )
        calibration_variance = np.maximum(calibration_variance, 0.0)
        variance += calibration_variance
        components.append(
            BandComponent(
                name="calibration",
                kind="syst",
                formula_id="F-UNC-2",
                sigma=np.sqrt(calibration_variance),
            )
        )

    if mc_variance is not None:
        mc_var = as_float_array("mc_variance", mc_variance, ndim=1)
        if mc_var.shape != solution.shape:
            raise ValidationError("mc_variance must match the primary axis of mu")
        if np.any(mc_var < 0.0):
            raise ValidationError("mc_variance must be non-negative")
        variance += mc_var
        components.append(
            BandComponent(
                name="mc_variance",
                kind="syst",
                formula_id="F-UNC-2",
                sigma=np.sqrt(mc_var),
            )
        )

    return np.sqrt(np.maximum(variance, 0.0)), tuple(components)


def combine_bands(
    sigma_stat: NDArray[np.float64],
    sigma_syst: NDArray[np.float64],
    *,
    components: tuple[BandComponent, ...] = (),
) -> UncertaintyBands:
    """Combine the two bands into the strict decomposition (F-UNC-3).

    ``sigma_total`` is computed as ``hypot(sigma_stat, sigma_syst)`` so the
    identity holds exactly under the certificate check.
    """
    stat = as_float_array("sigma_stat", sigma_stat, ndim=1)
    syst = as_float_array("sigma_syst", sigma_syst, ndim=1)
    if stat.shape != syst.shape:
        raise ValidationError(
            f"sigma_stat and sigma_syst must match, got {stat.shape} and {syst.shape}"
        )
    if np.any(stat < 0.0) or np.any(syst < 0.0):
        raise ValidationError("band sigmas must be non-negative")
    for component in components:
        if component.sigma.shape != stat.shape:
            raise ValidationError(
                f"component {component.name!r} does not match the band shape"
            )
    return UncertaintyBands(
        sigma_stat=stat,
        sigma_syst=syst,
        sigma_total=np.hypot(stat, syst),
        components=components,
    )


def verify_band_decomposition(bands: UncertaintyBands, *, strict: bool) -> float:
    """F-UNC-3 certificate: ``total**2 = stat**2 + syst**2``.

    Returns the largest relative deviation; strict mode raises when it exceeds
    :data:`DECOMPOSITION_TOL`.
    """
    residual = bands.sigma_total**2 - bands.sigma_stat**2 - bands.sigma_syst**2
    scale = np.maximum(1.0, bands.sigma_total**2)
    worst = float(np.max(np.abs(residual) / scale)) if residual.size else 0.0
    if strict and worst > DECOMPOSITION_TOL:
        raise CertificateError(
            "F-UNC-3",
            f"band decomposition violates total^2 = stat^2 + syst^2 (max rel. {worst:.3g})",
        )
    return worst


def simulation_mc_variance(
    response: ResponseMatrix,
    deposition_counts: NDArray[np.float64],
    column_totals: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    mu: NDArray[np.float64],
    *,
    sigma_fit: NDArray[np.float64],
    fisher: NDArray[np.float64] | sparse.spmatrix | None = None,
) -> NDArray[np.float64]:
    """Per-primary-bin variance from the multinomial simulation MC (F-UNC-2).

    Exact first-order propagation (D-119). With ``R = C G diag(1/N)`` the
    half-gradient derivative of ``g = H mu - b`` with respect to ``G_js`` is the
    **full** vector

        ``dg/dG_js = [ (C_j^T W r) e_s + (R^T W C_j) mu_s ] / N_s``,

    not the rank-one form ``d_js e_s^T``: the second term couples every primary
    row and cannot be dropped. With the multinomial covariance
    ``Cov(G_s) = N_s (diag(p_s) - p_s p_s^T)`` and reduced free-set inverse
    ``U = H_FF**-1``, the diagonal of the propagated covariance is

        ``Var_i = sum_s (1/N_s) [ sum_j p_js X_js_i^2 - Xbar_s_i^2 ]``

    with ``X_js = U v_js``, ``v_js = (C_j^T W r) e_s + (R^T W C_j) mu_s`` and
    ``Xbar_s = sum_j p_js X_js``. The unrecorded zero-deposition category has
    ``v = 0``, so it cancels from ``A_s`` and the sums run over the recorded
    deposition bins only. ``fisher`` is the same half-Hessian used by the
    solver.
    """
    matrix = check_response_matrix(response.matrix)
    counts = as_float_array("deposition_counts", deposition_counts, ndim=2)
    totals = as_float_array("column_totals", column_totals, ndim=1)
    y = as_float_array("spectrum", spectrum, ndim=1)
    solution = as_float_array("mu", mu, ndim=1)
    errors = as_float_array("sigma_fit", sigma_fit, ndim=1)
    if counts.shape[0] != matrix.shape[1]:
        raise ValidationError("deposition_counts rows do not match the response deposition axis")
    if counts.shape[1] != totals.size or totals.size != solution.size:
        raise ValidationError("deposition_counts columns, column_totals and mu must agree")
    if y.size != matrix.shape[0] or errors.size != matrix.shape[0]:
        raise ValidationError("spectrum and sigma_fit must match the response rows")
    if np.any(totals <= 0.0):
        raise ValidationError("column_totals must be strictly positive")
    if np.any(errors <= 0.0):
        raise ValidationError("sigma_fit must be strictly positive")

    normalized = counts / totals[None, :]
    composed = np.asarray(matrix @ normalized, dtype=np.float64)
    composed_sparse = sparse.csr_matrix(composed)
    residual = composed @ solution - y
    weights = 1.0 / errors**2
    hessian = _hessian(composed_sparse, errors, fisher)
    weighted = matrix.T @ sparse.diags(weights)
    weighted_composed = composed_sparse.T @ sparse.diags(weights)
    # (R^T W C): shape (n_primary, n_deposition).
    mixed = (weighted_composed @ matrix).toarray()
    column_term = np.asarray(weighted @ residual, dtype=np.float64)

    if solution.size == 0:
        return np.zeros(0, dtype=np.float64)

    # Streaming F-UNC-2 (D-151): solve the reduced system for free-set columns
    # in blocks and contract block-local quantities. Neither ``H_FF**-1`` nor
    # ``U (R^T W C)`` is ever materialised as a whole; the only O(n^2) dense
    # object is the data-side ``mixed = R^T W C``, which carries no inverse.
    free = solution > 0.0
    if not np.any(free):
        return np.zeros(solution.size, dtype=np.float64)
    free_index = np.flatnonzero(free)
    reduced = hessian[free][:, free]
    factor = factor_spd(reduced)

    probabilities = normalized  # (n_deposition, n_primary)
    totals_safe = totals
    a = column_term
    abar = probabilities.T @ a
    energy = probabilities.T @ (a * a)
    dA = energy - abar * abar
    w1 = dA / totals_safe
    w2 = solution / totals_safe
    w3 = solution * solution / totals_safe

    variance = np.zeros(solution.size, dtype=np.float64)
    n_free = free_index.size
    block = max(1, min(MC_BLOCK_COLUMNS, n_free))
    for start in range(0, n_free, block):
        stop = min(start + block, n_free)
        width = stop - start
        rhs = np.zeros((n_free, width), dtype=np.float64)
        rhs[np.arange(start, stop), np.arange(width)] = 1.0
        solved = factor.solve(rhs)
        velocities = np.zeros((solution.size, width), dtype=np.float64)
        velocities[free_index, :] = solved
        g_block = mixed.T @ velocities  # (n_deposition, width): g_i
        t2 = probabilities.T @ g_block  # (n_primary, width)
        t1 = probabilities.T @ (a[:, None] * g_block)
        wv = probabilities.T @ (g_block * g_block)
        for local in range(width):
            u = velocities[:, local]
            t1_local = t1[:, local]
            t2_local = t2[:, local]
            wv_local = wv[:, local]
            variance[free_index[start + local]] = (
                float(w1 @ (u * u))
                + 2.0 * float(w2 @ (u * (t1_local - abar * t2_local)))
                + float(w3 @ (wv_local - t2_local * t2_local))
            )
    scale = np.maximum(1.0, np.abs(variance))
    if np.any(variance < -1e-9 * scale):
        worst = float(np.min(variance))
        raise SolverError(f"simulation MC variance is negative: {worst:.3g}")
    return np.maximum(variance, 0.0)


def _hessian(
    response: sparse.csr_matrix,
    sigma: NDArray[np.float64],
    fisher: NDArray[np.float64] | sparse.spmatrix | None,
) -> sparse.csr_matrix:
    # The injected ``fisher`` is the half-Hessian that already solved the
    # problem (F-SOLVE-1/D-150). Building ``R^T W R`` first and discarding it
    # wastes a full sparse product; validate the injected shape against the
    # response instead and only assemble the data Hessian when none is given.
    shape = (response.shape[1], response.shape[1])
    if fisher is None:
        weights = 1.0 / sigma**2
        return weighted_normal(response, weights)
    if sparse.issparse(fisher):
        matrix = fisher.tocsr().astype(np.float64)
        if matrix.shape != shape:
            raise ValidationError(
                f"fisher has shape {matrix.shape}, expected {shape}"
            )
        if not np.isfinite(matrix.data).all():
            raise ValidationError("fisher contains non-finite values")
        return matrix
    dense = as_float_array("fisher", fisher, ndim=2)
    if dense.shape != shape:
        raise ValidationError(f"fisher has shape {dense.shape}, expected {shape}")
    return sparse.csr_matrix(dense)


def _reduced_columns(
    hessian: sparse.csr_matrix,
    mu: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the reduced system on the free set, zeroing active rows (F-UNC-1).

    At a boundary solution the first-order sensitivity is
    ``H_FF**-1 rhs_F`` with the active variables fixed at zero, not the full
    ``H**-1 rhs``. Using the full inverse would overstate the free directions.
    """
    active = mu <= 0.0
    free = ~active
    result = np.zeros_like(rhs, dtype=np.float64)
    if not np.any(free):
        return result
    reduced = hessian[free][:, free]
    solution = _solve_matrix(reduced, rhs[free])
    result[free] = solution
    return result


def _solve_matrix(hessian: sparse.csr_matrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the normal system with the shared SPD policy (see _linalg.py)."""
    return solve_spd(hessian, np.asarray(rhs, dtype=np.float64))
