"""Non-negative solve and strict uncertainty propagation (F-UNF-3/F-UNF-4, W4).

The solver is :func:`kc761.core.solver.solve_nonnegative`; the uncertainty
bands come from :mod:`kc761.core.uncertainty` with the **same** half-Hessian
``H`` that defines the solved objective (requirement C of the W4 brief), so the
values and the bands cannot drift. Exactly-zero response columns are removed
(F-UNF-3) and re-inserted as zeros in the full-axis solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.core.binning import ChannelGrid
from kc761.core.model import InternalCalibration
from kc761.core.response import (
    ComposedResponse,
    ResponseMatrix,
    compose_parameter_jacobian,
    response_parameter_jacobian,
    slice_response,
)
from kc761.core.solver import (
    KktCertificate,
    RegularizationSpec,
    UnfoldSolution,
    normal_equations,
    solve_nonnegative,
)
from kc761.core.uncertainty import (
    UncertaintyBands,
    combine_bands,
    propagate_statistical,
    propagate_systematic,
    simulation_mc_variance,
    verify_band_decomposition,
)
from kc761.errors import ValidationError
from kc761.unfold.selection import fit_sigma
from kc761.unfold.types import WindowSelection


@dataclass(frozen=True)
class SolveOutcome:
    """Full-axis solution and band arrays plus F-UNF-4 diagnostics."""

    mu_full: NDArray[np.float64]
    stat_full: NDArray[np.float64]
    syst_full: NDArray[np.float64]
    total_full: NDArray[np.float64]
    chi2: float
    dof: int
    n_active: int
    n_fit_rows: int
    kept_columns: NDArray[np.int64]
    pruned_columns: NDArray[np.int64]
    reduced_matrix: sparse.csr_matrix
    sigma_fit: NDArray[np.float64]
    solution: UnfoldSolution
    bands: UncertaintyBands

    @property
    def certificate(self) -> KktCertificate:
        return self.solution.certificate

    @property
    def n_kept_columns(self) -> int:
        return int(self.kept_columns.size)

    @property
    def n_pruned_columns(self) -> int:
        return int(self.pruned_columns.size)


def exact_zero_columns(
    matrix: sparse.spmatrix,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split columns into exactly-zero and kept, by exact comparison (F-UNF-3).

    ``R`` is non-negative, so a column is exactly zero if and only if its sum is
    exactly zero. No tolerance is applied (D-110).
    """
    csr = matrix.tocsr().astype(np.float64)
    sums = np.asarray(csr.sum(axis=0), dtype=np.float64).ravel()
    zero = np.flatnonzero(sums == 0.0)
    kept = np.flatnonzero(sums != 0.0)
    return kept.astype(np.int64), zero.astype(np.int64)


def solve_window(
    *,
    composed: ComposedResponse,
    response: ResponseMatrix,
    selection: WindowSelection,
    data_values: NDArray[np.float64],
    data_variances: NDArray[np.float64],
    deposition_counts: NDArray[np.float64],
    column_totals: NDArray[np.float64],
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    channel_grid: ChannelGrid,
    channel_max: float,
    param_cov: NDArray[np.float64],
    regularization: RegularizationSpec,
    syst_frac: float,
    strict: bool = False,
) -> SolveOutcome:
    """Solve the padded window and propagate both bands (F-UNF-3/F-UNC-*)."""
    sliced = slice_response(composed, selection.solve_low, selection.solve_high)
    sliced_matrix = sliced.matrix.tocsr().astype(np.float64)
    kept, pruned = exact_zero_columns(sliced_matrix)
    if kept.size == 0:
        raise ValidationError(
            "no primary column has a non-zero response inside the requested window"
        )
    reduced = sliced_matrix[:, kept].tocsr()

    rows = slice(selection.solve_low, selection.solve_high + 1)
    y = np.asarray(data_values, dtype=np.float64)[rows]
    stat = np.asarray(data_variances, dtype=np.float64)[rows]
    if np.any(stat < 0.0):
        raise ValidationError("data fSumw2 must be non-negative")
    stat_variance = np.maximum(stat, 1.0)
    sigma_fit = fit_sigma(stat, y, syst_frac)

    # The half-Hessian fed to F-UNC-1/F-UNC-2 is the same construction the
    # solver uses; it is built once here and injected into the solver (D-150),
    # so the value and the bands share one definition with no double
    # factorization (requirement C, D-84).
    hessian, gradient, penalty_scale = normal_equations(
        reduced, y, sigma_fit, regularization
    )
    solution = solve_nonnegative(
        reduced,
        y,
        sigma_fit,
        regularization,
        strict=strict,
        normal=(hessian, gradient, penalty_scale),
    )
    mu = solution.mu
    residual = (np.asarray(reduced @ mu, dtype=np.float64) - y) / sigma_fit
    chi2 = float(residual @ residual)
    n_active = int(np.count_nonzero(mu > 0.0))
    dof = int(y.size - n_active)

    stat_band, stat_components = propagate_statistical(
        reduced,
        y,
        sigma_fit,
        mu,
        stat_variance=stat_variance,
        fisher=hessian,
    )

    calib_jacobian_full = response_parameter_jacobian(
        response.deposition_edges_kev,
        calibration,
        resol_params,
        channel_grid=channel_grid,
        channel_max=channel_max,
        strict=strict,
    )
    composed_jacobian_full = compose_parameter_jacobian(
        calib_jacobian_full, deposition_counts, column_totals
    )
    composed_jacobian = [
        jacobian[rows][:, kept].tocsr() for jacobian in composed_jacobian_full
    ]

    windowed_response = ResponseMatrix(
        matrix=response.matrix[rows].tocsr(),
        column_sums=response.column_sums,
        deposition_edges_kev=response.deposition_edges_kev,
    )
    sim_variance = simulation_mc_variance(
        windowed_response,
        np.asarray(deposition_counts, dtype=np.float64)[:, kept],
        np.asarray(column_totals, dtype=np.float64)[kept],
        y,
        mu,
        sigma_fit=sigma_fit,
        fisher=hessian,
    )

    syst_band, syst_components = propagate_systematic(
        reduced,
        mu,
        spectrum=y,
        sigma_fit=sigma_fit,
        calib_jacobian=composed_jacobian,
        calib_covariance=np.asarray(param_cov, dtype=np.float64),
        sim_variance=sim_variance,
        syst_frac=syst_frac,
        fisher=hessian,
    )
    bands = combine_bands(
        stat_band, syst_band, components=stat_components + syst_components
    )
    verify_band_decomposition(bands, strict=strict)

    n_primary = composed.matrix.shape[1]
    mu_full = np.zeros(n_primary, dtype=np.float64)
    stat_full = np.zeros(n_primary, dtype=np.float64)
    syst_full = np.zeros(n_primary, dtype=np.float64)
    total_full = np.zeros(n_primary, dtype=np.float64)
    mu_full[kept] = mu
    stat_full[kept] = bands.sigma_stat
    syst_full[kept] = bands.sigma_syst
    total_full[kept] = bands.sigma_total

    return SolveOutcome(
        mu_full=mu_full,
        stat_full=stat_full,
        syst_full=syst_full,
        total_full=total_full,
        chi2=chi2,
        dof=dof,
        n_active=n_active,
        n_fit_rows=int(y.size),
        kept_columns=kept,
        pruned_columns=pruned,
        reduced_matrix=reduced,
        sigma_fit=sigma_fit,
        solution=solution,
        bands=bands,
    )


__all__ = ["SolveOutcome", "exact_zero_columns", "solve_window"]
