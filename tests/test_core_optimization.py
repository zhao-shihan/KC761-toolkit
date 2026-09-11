"""Equivalence regressions and benchmarks for the D-172 performance paths.

These are auxiliary: the numeric definitions live in docs/derivations.md and
the runtime certificates, not here. The tests pin that an optimised path is
algebraically the same as the path it replaced, and the ``bench`` cases only
record wall-clock numbers (never a correctness gate).
"""

from __future__ import annotations

import os
import time

import numba
import numpy as np
import pytest
from scipy import sparse

from kc761.core import kernel, model, response
from kc761.core._gen import kernel_expr, model_expr
from kc761.core._linalg import weighted_normal_and_rhs
from kc761.core.binning import ChannelGrid
from kc761.core.response import ResponseMatrix
from kc761.core.uncertainty import simulation_mc_variance


def _per_column_triples(
    edges: np.ndarray,
    centers: np.ndarray,
    sigma: np.ndarray,
    n_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation: exactly one Python loop per source column."""
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for column, (center, width) in enumerate(zip(centers, sigma, strict=True)):
        radius = n_sigma * float(width)
        start = int(np.searchsorted(bin_centers, center - radius, side="right"))
        stop = int(np.searchsorted(bin_centers, center + radius, side="left"))
        if start >= stop:
            continue
        window = np.arange(start, stop)
        raw = kernel_expr.tapered_bin(
            edges[window],
            edges[window + 1],
            float(center),
            float(width),
            bin_centers[window],
            n_sigma,
        )
        denominator = float(raw.sum())
        if denominator <= 0.0:
            continue
        probability = raw / denominator
        keep = probability > 0.0
        rows.extend(window[keep].tolist())
        cols.extend([column] * int(keep.sum()))
        values.extend(probability[keep].tolist())
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(values)


def test_vectorised_kernel_matches_per_column_reference() -> None:
    rng = np.random.default_rng(11)
    edges = np.linspace(0.0, 400.0, 401)
    centers = rng.uniform(5.0, 395.0, size=120)
    sigma = rng.uniform(0.3, 6.0, size=120)
    triple = kernel.response_triples(edges, centers, sigma, n_sigma=5.0)
    ref_rows, ref_cols, ref_values = _per_column_triples(edges, centers, sigma, 5.0)
    dense = sparse.csr_matrix(
        (ref_values, (ref_rows, ref_cols)), shape=(edges.size - 1, centers.size)
    ).toarray()
    dense_opt = sparse.csr_matrix(
        (triple.values, (triple.rows, triple.cols)), shape=(edges.size - 1, centers.size)
    ).toarray()
    assert np.allclose(dense_opt, dense, rtol=1e-12, atol=1e-15)


def test_weighted_normal_dense_and_sparse_paths_agree() -> None:
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.5, 2.0, size=40)
    rhs = rng.standard_normal(40)
    # Dense enough to select the BLAS path.
    dense_matrix = sparse.csr_matrix(rng.random((40, 25)))
    hessian_dense, gradient_dense = weighted_normal_and_rhs(dense_matrix, weights, rhs)
    dense_array = dense_matrix.toarray()
    assert np.allclose(
        hessian_dense.toarray(),
        (dense_array.T * weights) @ dense_array,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(gradient_dense, dense_array.T @ (weights * rhs), rtol=1e-12, atol=1e-12)
    # A banded matrix with nnz < 5% selects the sparse path.
    sparse_matrix = sparse.diags(
        [rng.random(25), rng.random(25)], offsets=[0, 1], shape=(40, 25)
    ).tocsr()
    hessian_sparse, gradient_sparse = weighted_normal_and_rhs(sparse_matrix, weights, rhs)
    sparse_array = sparse_matrix.toarray()
    assert np.allclose(
        hessian_sparse.toarray(),
        (sparse_array.T * weights) @ sparse_array,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(gradient_sparse, sparse_array.T @ (weights * rhs), rtol=1e-12, atol=1e-12)


def test_streaming_mc_variance_with_active_bins_matches_direct() -> None:
    rng = np.random.default_rng(23)
    n_rows, n_deposition, n_primary = 11, 6, 6
    matrix = rng.random((n_rows, n_deposition))
    matrix /= matrix.sum(axis=0, keepdims=True)
    counts = rng.random((n_deposition, n_primary)) * 10.0 + 1.0
    totals = counts.sum(axis=0) + 3.0
    sigma_fit = rng.random(n_rows) + 0.5
    composed = matrix @ (counts / totals[None, :])
    hessian = composed.T @ (composed / (sigma_fit**2)[:, None]) + 1e-3 * np.eye(n_primary)
    spectrum = composed @ np.ones(n_primary)
    mu = np.linalg.solve(hessian, composed.T @ (spectrum / sigma_fit**2))
    mu[:2] = 0.0  # exercise the active set / block embedding
    response = ResponseMatrix(
        matrix=sparse.csr_matrix(matrix),
        column_sums=matrix.sum(axis=0),
        deposition_edges_kev=np.arange(n_deposition + 1.0),
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
    assert variance.shape == mu.shape
    assert np.all(variance >= 0.0)
    assert variance[0] == 0.0 and variance[1] == 0.0
    # Active coordinates stay exactly zero, free coordinates are produced by the
    # block path; compare against the direct linearization with the reduced
    # free-set inverse (the boundary convention the solver and F-UNC use).
    free = mu > 0.0
    inverse = np.zeros((n_primary, n_primary))
    inverse[np.ix_(free, free)] = np.linalg.inv(hessian[np.ix_(free, free)])
    weights = 1.0 / sigma_fit**2
    column_term = matrix.T @ (weights * (composed @ mu - spectrum))
    mixed = composed.T @ np.diag(weights) @ matrix
    probabilities = counts / totals[None, :]
    reference = np.zeros(n_primary)
    for column in range(n_primary):
        derivatives = np.stack(
            [
                -inverse
                @ (column_term[d] * np.eye(n_primary)[column] + mixed[:, d] * mu[column])
                / totals[column]
                for d in range(n_deposition)
            ]
        )
        mean = probabilities[:, column] @ derivatives
        second = probabilities[:, column] @ (derivatives * derivatives)
        reference += totals[column] * (second - mean * mean)
    assert np.allclose(variance, reference, rtol=1e-9, atol=1e-12)


@pytest.mark.bench
@pytest.mark.skipif(
    os.environ.get("KC761_RUN_BENCH") != "1",
    reason="benchmark case; set KC761_RUN_BENCH=1 and select -m bench",
)
def test_bench_vectorised_kernel_small() -> None:
    """Record a small kernel timing; excluded from the correctness CI."""
    edges = np.linspace(0.0, 1024.0, 1025)
    centers = np.linspace(2.0, 1022.0, 256)
    sigma = np.full_like(centers, 4.0)
    sizes: list[int] = []
    for _ in range(3):
        start = time.perf_counter()
        triples = kernel.response_triples(edges, centers, sigma, n_sigma=5.0)
        sizes.append(int(triples.values.size))
        assert time.perf_counter() - start >= 0.0
    assert all(size > 0 for size in sizes)


def _scale_parameter_jacobian_problem():  # noqa: ANN202
    """A production-scale problem for the fused Jacobian.

    The calibration is linear (10 keV per channel) and the wide resolution makes
    every deposition column span about 30 channel bins, so the flat pattern has
    hundreds of thousands of entries (the 2048 production products are ~1.8M).
    """
    n_channels = 3000
    grid = ChannelGrid(n_channels)
    deposition_edges = np.linspace(-5.0, 10.0 * n_channels - 5.0, n_channels + 1)
    calibration = model.InternalCalibration(c0=0.0, k1=10.0, k2=10.0, k3=10.0)
    resol = np.array([30.0, 30.0, 30.0])
    return grid, deposition_edges, calibration, resol


def _reference_parameter_jacobians(grid, edges, calibration, resol):  # noqa: ANN202
    """Independent reference: vectorised fill plus per-parameter quotient."""
    n_sigma = kernel.N_SIGMA
    channel_edges = model.energy_kev(
        grid.edges(), calibration, channel_max=float(grid.n_channels)
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma, sigma_grad = model.resolution_sigma_grad(centers, resol)
    gradients = kernel.response_triples_grad(
        channel_edges, centers, sigma, n_sigma=n_sigma
    )
    edge_gradient = model_expr.energy_grad_reported(
        np.arange(grid.n_channels + 1, dtype=np.float64) - 0.5
    )
    rows = gradients.triples.rows
    cols = gradients.triples.cols
    unnormalized = gradients.values
    denominator = gradients.column_denominator[cols]
    matrices = []
    for parameter in range(4):
        local = (
            gradients.dn_de_lo * edge_gradient[parameter][rows]
            + gradients.dn_de_hi * edge_gradient[parameter][rows + 1]
        )
        sums = np.bincount(cols, weights=local, minlength=centers.size)
        values = kernel_expr.normalize_derivative(
            unnormalized, local, denominator, sums[cols]
        )
        matrices.append(sparse.csr_matrix((values, (rows, cols))))
    for resolution_index in range(3):
        local = gradients.dn_dsigma * sigma_grad[resolution_index][cols]
        sums = np.bincount(cols, weights=local, minlength=centers.size)
        values = kernel_expr.normalize_derivative(
            unnormalized, local, denominator, sums[cols]
        )
        matrices.append(sparse.csr_matrix((values, (rows, cols))))
    return matrices


def test_fused_parameter_jacobian_matches_small_path_at_scale() -> None:
    grid, edges, calibration, resol = _scale_parameter_jacobian_problem()
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma = model.resolution_sigma_kev(centers, resol)
    channel_edges = model.energy_kev(
        grid.edges(), calibration, channel_max=float(grid.n_channels)
    )
    bin_centers = 0.5 * (channel_edges[:-1] + channel_edges[1:])
    starts = np.searchsorted(bin_centers, centers - kernel.N_SIGMA * sigma, side="right")
    stops = np.searchsorted(bin_centers, centers + kernel.N_SIGMA * sigma, side="left")
    total = int(np.maximum(stops - starts, 0).sum())
    assert total > 50_000, "test must be production-scale, not a toy problem"

    fused = response.response_parameter_jacobian(
        edges, calibration, resol, channel_grid=grid, channel_max=float(grid.n_channels)
    )
    reference = _reference_parameter_jacobians(grid, edges, calibration, resol)
    assert len(fused) == len(reference) == 7
    for got, expected in zip(fused, reference, strict=True):
        assert got.shape == expected.shape
        assert np.allclose(got.toarray(), expected.toarray(), rtol=1e-9, atol=1e-12)


def test_fused_parameter_jacobian_is_thread_deterministic() -> None:
    grid, edges, calibration, resol = _scale_parameter_jacobian_problem()
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = response.response_parameter_jacobian(
            edges, calibration, resol, channel_grid=grid, channel_max=float(grid.n_channels)
        )
        numba.set_num_threads(min(8, original))
        many = response.response_parameter_jacobian(
            edges, calibration, resol, channel_grid=grid, channel_max=float(grid.n_channels)
        )
    finally:
        numba.set_num_threads(original)
    for first, second in zip(single, many, strict=True):
        assert np.array_equal(first.data, second.data)
        assert np.array_equal(first.indices, second.indices)
        assert np.array_equal(first.indptr, second.indptr)
