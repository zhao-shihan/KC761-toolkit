"""Deposition-to-channel response C and composed primary-to-channel R.

Formula IDs (docs/derivations.md): F-RESP-1 .. F-RESP-4.

* F-RESP-1: ``C[i, j]`` is the probability that a gamma depositing energy in
  deposition bin ``j`` lands in channel bin ``i``; every non-empty column
  sums to exactly 1 after taper renormalization, empty columns are zero.
* F-RESP-2: ``R = C . p_tilde . diag(eta) = C . G . diag(1 / N)`` (the two
  forms are algebraically identical because ``p_tilde . eta = G / N``). The
  implementation uses the second form, which has no intermediate 0/0.
* F-RESP-3: composition happens over the full primary axis and only then is
  the working window sliced (D-43/D-44). After slicing, ``column_sums`` and
  ``efficiency`` keep their full-axis meaning; the new ``channel_low`` /
  ``channel_high`` fields describe exactly which rows the matrix holds, and
  :func:`verify_window_slice` checks that slicing only removed rows.
* F-RESP-4: ``dC/dq`` is assembled by chaining the F-KERN-4 kernel
  derivatives through the calibration model; ``dR/dq = dC/dq . P``.

Matrix orientation follows D-20: ``x`` is the output side, ``y`` the input
side. The composed matrix therefore has x = channel, y = primary energy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.core import _gen
from kc761.core._checks import as_float_array, require_positive
from kc761.core.binning import ChannelGrid, EnergyGrid
from kc761.core.kernel import (
    N_SIGMA,
    SparseTriples,
    response_triples,
    response_triples_grad,
)
from kc761.core.model import (
    InternalCalibration,
    energy_kev,
    resolution_sigma_grad,
    resolution_sigma_kev,
)
from kc761.errors import CertificateError, ValidationError

N_REPORTED_PARAMS = 7
"""Reported fit parameters ``(c0, c1, c2, c3, b0, b1, b2)`` (F-RESP-4)."""

RESPONSE_COLUMN_TOL = 1e-10
"""F-RESP-1/F-RESP-2 certificate tolerance."""


@dataclass(frozen=True)
class ResponseMatrix:
    """Sparse deposition-to-channel matrix with its geometry (F-RESP-1)."""

    matrix: sparse.csr_matrix
    column_sums: NDArray[np.float64]
    deposition_edges_kev: NDArray[np.float64]


@dataclass(frozen=True)
class ComposedResponse:
    """Composed primary-to-channel matrix and derived efficiency (F-RESP-2).

    ``column_sums`` and ``efficiency`` always describe the **full** channel
    axis. ``channel_low``/``channel_high`` record which rows ``matrix`` holds:
    the full axis for a freshly composed response, the requested window after
    :func:`slice_response` (F-RESP-3).
    """

    matrix: sparse.csr_matrix
    column_sums: NDArray[np.float64]
    efficiency: NDArray[np.float64]
    primary_edges_kev: NDArray[np.float64]
    channel_low: int = 0
    channel_high: int | None = None


def _check_edges(name: str, edges_kev: NDArray[np.float64]) -> NDArray[np.float64]:
    return EnergyGrid(edges_kev=as_float_array(name, edges_kev, ndim=1)).edges_kev


def build_response_matrix(
    deposition_edges_kev: NDArray[np.float64],
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    *,
    channel_grid: ChannelGrid,
    channel_max: float,
    n_sigma: float = N_SIGMA,
    strict: bool = False,
) -> ResponseMatrix:
    """Assemble C from the calibrated model (F-RESP-1).

    The deposition axis is the full product axis (D-79): there is no
    parameter-dependent bin selection, so the fit objective is continuous in
    the parameters by construction. ``channel_max`` is the acquisition
    constant of the internal calibration basis (F-MODEL-1).
    """
    edges = _check_edges("deposition_edges_kev", deposition_edges_kev)
    maximum = require_positive("channel_max", channel_max)
    channel_edges_kev = energy_kev(channel_grid.edges(), calibration, channel_max=maximum)
    if not np.all(np.diff(channel_edges_kev) > 0.0):
        raise ValidationError(
            "calibrated channel edges are not strictly increasing; "
            "F-MODEL-3 requires a monotone calibration"
        )
    deposition_centers = 0.5 * (edges[:-1] + edges[1:])
    sigma = resolution_sigma_kev(deposition_centers, resol_params, strict=strict)
    if strict and np.any(sigma <= 0.0):
        raise CertificateError(
            "F-MODEL-5",
            "zero resolution width on the deposition grid; strict mode refuses to build C",
        )
    triples = response_triples(channel_edges_kev, deposition_centers, sigma, n_sigma=n_sigma)
    matrix = sparse.csr_matrix(
        (triples.values, (triples.rows, triples.cols)),
        shape=(channel_grid.n_channels, deposition_centers.size),
    )
    column_sums = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
    response = ResponseMatrix(
        matrix=matrix,
        column_sums=column_sums,
        deposition_edges_kev=edges,
    )
    verify_response_columns(response, strict=strict)
    return response


def compose_response(
    response: ResponseMatrix,
    deposition_counts: NDArray[np.float64],
    column_totals: NDArray[np.float64],
    *,
    primary_edges_kev: NDArray[np.float64],
) -> ComposedResponse:
    """Compose ``R = C . p_tilde . diag(eta) = C . G . diag(1 / N)`` (F-RESP-2).

    ``deposition_counts`` is the simulation matrix G (deposition x primary)
    and ``column_totals`` the per-primary-column event count ``N_j``. The
    composition runs over the full primary axis (D-43); ``primary_edges_kev``
    is required so the result is self-describing and validated.
    """
    counts = as_float_array("deposition_counts", deposition_counts, ndim=2)
    totals = as_float_array("column_totals", column_totals, ndim=1)
    primary_edges = _check_edges("primary_edges_kev", primary_edges_kev)
    n_deposition = response.matrix.shape[1]
    if counts.shape[0] != n_deposition:
        raise ValidationError(
            f"deposition_counts has {counts.shape[0]} deposition bins, "
            f"response has {n_deposition}"
        )
    if totals.size != counts.shape[1] or primary_edges.size != counts.shape[1] + 1:
        raise ValidationError(
            "column_totals and primary_edges_kev must match deposition_counts columns"
        )
    if np.any(counts < 0.0):
        raise ValidationError("deposition_counts must be non-negative")
    if np.any(totals < 0.0):
        raise ValidationError("column_totals must be non-negative")
    column_sums = counts.sum(axis=0)
    if np.any(column_sums > totals + RESPONSE_COLUMN_TOL):
        worst = int(np.argmax(column_sums - totals))
        raise ValidationError(
            f"column {worst}: detected counts exceed the column total "
            f"({column_sums[worst]:.12g} > {totals[worst]:.12g})"
        )

    normalized = np.zeros_like(counts)
    positive = totals > 0.0
    np.divide(
        counts,
        totals[None, :],
        out=normalized,
        where=positive[None, :],
    )
    composed = np.asarray(response.matrix @ normalized, dtype=np.float64)
    efficiency = np.zeros_like(totals)
    np.divide(column_sums, totals, out=efficiency, where=positive)
    if np.any((efficiency < -RESPONSE_COLUMN_TOL) | (efficiency > 1.0 + RESPONSE_COLUMN_TOL)):
        raise ValidationError("derived efficiency outside [0, 1] (F-SIM-3)")
    return ComposedResponse(
        matrix=sparse.csr_matrix(composed),
        column_sums=np.asarray(composed.sum(axis=0)).ravel(),
        efficiency=efficiency,
        primary_edges_kev=primary_edges,
        channel_low=0,
        channel_high=response.matrix.shape[0] - 1,
    )


def slice_response(
    composed: ComposedResponse,
    channel_low: int,
    channel_high: int,
) -> ComposedResponse:
    """Slice rows to ``[channel_low, channel_high]`` after composition (F-RESP-3).

    Only the matrix rows are selected: ``column_sums`` and ``efficiency`` keep
    their full-axis meaning and ``channel_low``/``channel_high`` record the
    window, so downstream certificates can distinguish the full-axis column
    sums from the sliced row sums.
    """
    n_channels = composed.matrix.shape[0]
    if not 0 <= channel_low <= channel_high < n_channels:
        raise ValidationError(
            f"window [{channel_low}, {channel_high}] outside [0, {n_channels - 1}]"
        )
    return replace(
        composed,
        matrix=composed.matrix[channel_low : channel_high + 1],
        channel_low=channel_low,
        channel_high=channel_high,
    )


def response_triples_for(
    deposition_edges_kev: NDArray[np.float64],
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    *,
    channel_grid: ChannelGrid,
    channel_max: float,
    n_sigma: float = N_SIGMA,
) -> SparseTriples:
    """Kernel triples of C for inspection and certificates (F-RESP-1)."""
    edges = _check_edges("deposition_edges_kev", deposition_edges_kev)
    channel_edges_kev = energy_kev(
        channel_grid.edges(), calibration, channel_max=require_positive("channel_max", channel_max)
    )
    deposition_centers = 0.5 * (edges[:-1] + edges[1:])
    sigma = resolution_sigma_kev(deposition_centers, resol_params)
    return response_triples(channel_edges_kev, deposition_centers, sigma, n_sigma=n_sigma)


def response_parameter_jacobian(
    deposition_edges_kev: NDArray[np.float64],
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    *,
    channel_grid: ChannelGrid,
    channel_max: float,
    n_sigma: float = N_SIGMA,
    strict: bool = False,
) -> list[sparse.csr_matrix]:
    """``dC/dq`` for ``q = (c0..c3, b0..b2)`` in the reported basis (F-RESP-4).

    Chain: a channel bin edge contributes through its energy ``E(ch)`` and a
    deposition column through its resolution width ``sigma(E)``. The kernel
    derivatives ``d p / d e_lo``, ``d p / d e_hi`` and ``d p / d sigma`` come
    from F-KERN-4; the model derivatives come from F-MODEL-1/F-MODEL-4.
    """
    edges = _check_edges("deposition_edges_kev", deposition_edges_kev)
    maximum = require_positive("channel_max", channel_max)
    channel_edges_kev = energy_kev(channel_grid.edges(), calibration, channel_max=maximum)
    if not np.all(np.diff(channel_edges_kev) > 0.0):
        raise ValidationError("calibrated channel edges are not strictly increasing")
    deposition_centers = 0.5 * (edges[:-1] + edges[1:])
    sigma, sigma_grad = resolution_sigma_grad(deposition_centers, resol_params, strict=strict)
    if strict and np.any(sigma <= 0.0):
        raise CertificateError(
            "F-MODEL-5",
            "zero resolution width on the deposition grid; strict mode refuses to build dC/dq",
        )
    gradients = response_triples_grad(
        channel_edges_kev, deposition_centers, sigma, n_sigma=n_sigma
    )
    n_channels = channel_grid.n_channels
    shape = (n_channels, deposition_centers.size)

    # Channel-edge coordinate l corresponds to channel coordinate l - 0.5, so
    # dE_l/dq = energy_grad_reported(l - 0.5) for the reported basis.
    edge_coordinates = np.arange(n_channels + 1, dtype=np.float64) - 0.5
    edge_gradient = _gen.model_expr.energy_grad_reported(
        edge_coordinates
    )  # shape (4, n_channels + 1)

    rows = gradients.triples.rows
    cols = gradients.triples.cols
    unnormalized = gradients.values
    denominator = gradients.column_denominator[cols]
    n_primary = deposition_centers.size

    def quotient(local_derivative: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalized derivative with the full-column renormalization sum."""
        column_sums = np.bincount(cols, weights=local_derivative, minlength=n_primary)
        return (
            local_derivative / denominator
            - unnormalized * column_sums[cols] / denominator**2
        )

    jacobians: list[sparse.csr_matrix] = []
    for parameter in range(4):
        local = (
            gradients.dn_de_lo * edge_gradient[parameter][rows]
            + gradients.dn_de_hi * edge_gradient[parameter][rows + 1]
        )
        jacobians.append(
            sparse.csr_matrix((quotient(local), (rows, cols)), shape=shape)
        )
    for resolution_index in range(3):
        local = gradients.dn_dsigma * sigma_grad[resolution_index][cols]
        jacobians.append(
            sparse.csr_matrix((quotient(local), (rows, cols)), shape=shape)
        )
    return jacobians


def compose_parameter_jacobian(
    response_jacobian: list[sparse.csr_matrix],
    deposition_counts: NDArray[np.float64],
    column_totals: NDArray[np.float64],
) -> list[sparse.csr_matrix]:
    """``dR/dq = dC/dq . P`` with ``P = G / N`` (F-RESP-4)."""
    counts = as_float_array("deposition_counts", deposition_counts, ndim=2)
    totals = as_float_array("column_totals", column_totals, ndim=1)
    if totals.shape != (counts.shape[1],):
        raise ValidationError("column_totals must match deposition_counts columns")
    normalized = np.zeros_like(counts)
    np.divide(counts, totals[None, :], out=normalized, where=(totals > 0.0)[None, :])
    return [sparse.csr_matrix(jacobian @ normalized) for jacobian in response_jacobian]


def verify_response_columns(response: ResponseMatrix, *, strict: bool) -> NDArray[np.float64]:
    """F-RESP-1 certificate: C columns sum to 1 or are exactly zero."""
    matrix = response.matrix
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0.0):
        raise ValidationError("response matrix contains negative or non-finite values")
    sums = response.column_sums
    expected = (sums > 0.5).astype(np.float64)
    bad = np.abs(sums - expected) > RESPONSE_COLUMN_TOL
    if strict and np.any(bad):
        worst = int(np.argmax(np.abs(sums - expected)))
        raise CertificateError(
            "F-RESP-1",
            f"column {worst} sums to {sums[worst]:.12g} instead of {expected[worst]:.0f}",
        )
    return sums


def verify_composed_columns(
    composed: ComposedResponse,
    response: ResponseMatrix,
    deposition_counts: NDArray[np.float64],
    column_totals: NDArray[np.float64],
    *,
    strict: bool,
) -> NDArray[np.float64]:
    """F-RESP-2 certificate: R equals ``C . G . diag(1 / N)`` and stays bounded.

    The expected column sum is the detected mass reaching non-empty C columns
    divided by ``N_j``; it is recomputed from the inputs, not from R itself.
    """
    counts = as_float_array("deposition_counts", deposition_counts, ndim=2)
    totals = as_float_array("column_totals", column_totals, ndim=1)
    reachable = response.column_sums > 0.5
    detected = counts[reachable].sum(axis=0) if np.any(reachable) else np.zeros(counts.shape[1])
    expected = np.zeros(counts.shape[1])
    np.divide(detected, totals, out=expected, where=totals > 0.0)
    bad = np.abs(composed.column_sums - expected) > RESPONSE_COLUMN_TOL
    if np.any(composed.efficiency < -RESPONSE_COLUMN_TOL) or np.any(
        composed.efficiency > 1.0 + RESPONSE_COLUMN_TOL
    ):
        raise ValidationError("efficiency outside [0, 1] (F-SIM-3)")
    if strict and np.any(bad):
        worst = int(np.argmax(np.abs(composed.column_sums - expected)))
        raise CertificateError(
            "F-RESP-2",
            f"column {worst}: composed sum {composed.column_sums[worst]:.12g} "
            f"!= expected {expected[worst]:.12g}",
        )
    return expected


def verify_window_slice(
    composed: ComposedResponse,
    *,
    strict: bool,
) -> None:
    """F-RESP-3 certificate: slicing only removed rows from the full matrix.

    ``column_sums``/``efficiency`` retain full-axis semantics, so the sliced
    row sums may be smaller but never larger; the removed mass is non-negative.
    """
    if composed.channel_high is None:
        return
    row_sums = np.asarray(composed.matrix.sum(axis=0)).ravel()
    removed = composed.column_sums - row_sums
    if np.any(row_sums < -RESPONSE_COLUMN_TOL) or np.any(removed < -RESPONSE_COLUMN_TOL):
        raise ValidationError("window slicing produced negative row sums")
    if strict:
        exceeded = row_sums > composed.column_sums + RESPONSE_COLUMN_TOL
        if np.any(exceeded):
            worst = int(np.argmax(row_sums - composed.column_sums))
            raise CertificateError(
                "F-RESP-3",
                f"column {worst}: sliced row sum exceeds the full column sum",
            )
