"""Auxiliary checks for C, R, slicing and the response Jacobian (F-RESP-1..4)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761tool.core import model, response
from kc761tool.core.binning import ChannelGrid
from kc761tool.errors import ValidationError


def _calibration() -> model.InternalCalibration:
    # E(channel) = 10 keV per channel.
    return model.InternalCalibration(c0=0.0, k1=10.0, k2=10.0, k3=10.0)


def _setup(n_channels: int = 24):
    grid = ChannelGrid(n_channels)
    deposition_edges = np.linspace(-5.0, 10.0 * n_channels - 5.0, n_channels + 1)
    resol = np.array([1.5, 1.5, 1.5])
    return grid, deposition_edges, resol


def _counts(n_deposition: int, n_primary: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    totals = np.full(n_primary, 500.0)
    counts = np.zeros((n_deposition, n_primary))
    for column in range(n_primary):
        counts[:, column] = rng.multinomial(500, rng.dirichlet(np.full(n_deposition, 0.3)))
    return counts, totals


def _primary_edges(n_primary: int, n_channels: int) -> np.ndarray:
    return np.linspace(-5.0, 10.0 * n_channels - 5.0, n_primary + 1)


def test_response_columns_and_certificate() -> None:
    grid, edges, resol = _setup()
    matrix = response.build_response_matrix(
        edges,
        _calibration(),
        resol,
        channel_grid=grid,
        channel_max=float(grid.n_channels - 1),
        strict=True,
    )
    assert matrix.matrix.shape == (grid.n_channels, grid.n_channels)
    assert np.allclose(matrix.column_sums, 1.0, atol=1e-10)
    response.verify_response_columns(matrix, strict=True)


def test_compose_matches_explicit_normalization() -> None:
    grid, edges, resol = _setup()
    matrix = response.build_response_matrix(
        edges, _calibration(), resol, channel_grid=grid, channel_max=23.0
    )
    counts, totals = _counts(grid.n_channels, 3)
    composed = response.compose_response(
        matrix, counts, totals, primary_edges_kev=_primary_edges(3, grid.n_channels)
    )
    detected = counts.sum(axis=0)
    expected = np.asarray(matrix.matrix @ (counts / totals[None, :]))
    assert np.allclose(composed.matrix.toarray(), expected, rtol=1e-12, atol=1e-15)
    assert np.allclose(composed.efficiency, detected / totals)
    assert np.allclose(composed.column_sums, expected.sum(axis=0), rtol=1e-12)
    assert np.allclose(composed.efficiency, composed.column_sums, atol=1e-12)
    response.verify_composed_columns(composed, matrix, counts, totals, strict=True)


def test_slice_keeps_full_axis_semantics() -> None:
    grid, edges, resol = _setup()
    matrix = response.build_response_matrix(
        edges, _calibration(), resol, channel_grid=grid, channel_max=23.0
    )
    counts, totals = _counts(grid.n_channels, 2)
    composed = response.compose_response(
        matrix, counts, totals, primary_edges_kev=_primary_edges(2, grid.n_channels)
    )
    sliced = response.slice_response(composed, 5, 15)
    assert sliced.matrix.shape == (11, composed.matrix.shape[1])
    assert (sliced.channel_low, sliced.channel_high) == (5, 15)
    assert np.array_equal(sliced.column_sums, composed.column_sums)
    assert np.array_equal(sliced.efficiency, composed.efficiency)
    response.verify_window_slice(sliced, strict=True)
    row_sums = np.asarray(sliced.matrix.sum(axis=0)).ravel()
    assert np.all(row_sums <= sliced.column_sums + 1e-12)


def test_compose_rejects_inconsistent_totals() -> None:
    grid, edges, resol = _setup()
    matrix = response.build_response_matrix(
        edges, _calibration(), resol, channel_grid=grid, channel_max=23.0
    )
    counts, totals = _counts(grid.n_channels, 2)
    totals = totals * 0.0
    with pytest.raises(ValidationError):
        response.compose_response(
            matrix, counts, totals, primary_edges_kev=_primary_edges(2, grid.n_channels)
        )


def test_parameter_jacobian_column_sums_vanish() -> None:
    grid, edges, resol = _setup(12)
    jacobians = response.response_parameter_jacobian(
        edges,
        _calibration(),
        resol,
        channel_grid=grid,
        channel_max=11.0,
        n_sigma=3.0,
    )
    assert len(jacobians) == model.N_REPORTED_PARAMS
    for jacobian in jacobians:
        column_sums = np.asarray(jacobian.sum(axis=0)).ravel()
        assert np.max(np.abs(column_sums)) < 1e-10


def test_parameter_jacobian_matches_finite_difference() -> None:
    grid, edges, resol = _setup(10)
    base_reported = model.internal_to_reported(
        _calibration(), channel_max=9.0
    ).as_array()
    jacobians = response.response_parameter_jacobian(
        edges,
        _calibration(),
        resol,
        channel_grid=grid,
        channel_max=9.0,
        n_sigma=3.0,
    )

    def build(reported: np.ndarray, resolution: np.ndarray):
        internal = model.reported_to_internal(
            model.ReportedCalibration(*reported), channel_max=9.0
        )
        return response.build_response_matrix(
            edges, internal, resolution, channel_grid=grid, channel_max=9.0, n_sigma=3.0
        ).matrix

    for parameter in (1, 2):
        steps = (1e-6, 1e-7)
        differences = []
        for step in steps:
            plus = base_reported.copy()
            minus = base_reported.copy()
            plus[parameter] += step
            minus[parameter] -= step
            differences.append((build(plus, resol) - build(minus, resol)) / (2.0 * step))
        richardson = (4.0 * differences[1] - differences[0]) / 3.0
        assert np.allclose(
            jacobians[parameter].toarray(), richardson.toarray(), rtol=1e-4, atol=1e-6
        )
    for index in range(3):
        step = 1e-7
        plus = resol.copy()
        minus = resol.copy()
        plus[index] += step
        minus[index] -= step
        finite = (build(base_reported, plus) - build(base_reported, minus)) / (2.0 * step)
        assert np.allclose(
            jacobians[4 + index].toarray(), finite.toarray(), rtol=1e-5, atol=1e-8
        )


def test_compose_parameter_jacobian_matches_finite_difference() -> None:
    grid, edges, resol = _setup(8)
    counts, totals = _counts(8, 2, seed=11)
    d_c = response.response_parameter_jacobian(
        edges,
        _calibration(),
        resol,
        channel_grid=grid,
        channel_max=7.0,
        n_sigma=3.0,
    )
    d_r = response.compose_parameter_jacobian(d_c, counts, totals)
    step = 1e-7
    base_internal = _calibration()
    plus_internal = model.reported_to_internal(
        model.ReportedCalibration(base_internal.c0 + step, 10.0, 10.0, 10.0),
        channel_max=7.0,
    )
    minus_internal = model.reported_to_internal(
        model.ReportedCalibration(base_internal.c0 - step, 10.0, 10.0, 10.0),
        channel_max=7.0,
    )
    plus = response.build_response_matrix(
        edges, plus_internal, resol, channel_grid=grid, channel_max=7.0, n_sigma=3.0
    )
    minus = response.build_response_matrix(
        edges, minus_internal, resol, channel_grid=grid, channel_max=7.0, n_sigma=3.0
    )
    composed_plus = response.compose_response(
        plus, counts, totals, primary_edges_kev=_primary_edges(2, grid.n_channels)
    )
    composed_minus = response.compose_response(
        minus, counts, totals, primary_edges_kev=_primary_edges(2, grid.n_channels)
    )
    finite = (composed_plus.matrix - composed_minus.matrix) / (2.0 * step)
    assert np.allclose(d_r[0].toarray(), finite.toarray(), rtol=1e-4, atol=1e-7)


def test_empty_deposition_columns_are_zero() -> None:
    grid = ChannelGrid(8)
    deposition_edges = np.linspace(1000.0, 1080.0, 9)
    matrix = response.build_response_matrix(
        deposition_edges, _calibration(), np.array([1.0, 1.0, 1.0]),
        channel_grid=grid,
        channel_max=7.0,
    )
    assert np.all(matrix.column_sums == 0.0)
    response.verify_response_columns(matrix, strict=True)
    counts, totals = _counts(8, 1, seed=5)
    composed = response.compose_response(
        matrix, counts, totals, primary_edges_kev=_primary_edges(1, grid.n_channels)
    )
    assert np.all(composed.column_sums == 0.0)
    assert np.all(composed.efficiency > 0.0)
    response.verify_composed_columns(composed, matrix, counts, totals, strict=True)
    response.verify_window_slice(composed, strict=True)
