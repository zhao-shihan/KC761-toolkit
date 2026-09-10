"""W5 no-Geant4 tests: sampling formulas, seed derivation, worker independence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kc761 import errors
from kc761.sim import sources
from kc761.sim.detector import build_plane_gamma_source, build_sphere_gamma_source
from kc761.sim.generator import (
    block_seed,
    column_seed,
    lambertian_cos_sin,
    sample_plane_surface,
    sample_sphere_surface,
)


def test_lambertian_cos_sin_is_a_unit_vector() -> None:
    for draw in (0.0, 0.25, 0.5, 0.999999):
        cos_t, sin_t = lambertian_cos_sin(draw)
        assert 0.0 <= cos_t <= 1.0
        assert math.isclose(cos_t**2 + sin_t**2, 1.0, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(cos_t, math.sqrt(draw))
    with pytest.raises(errors.ValidationError):
        lambertian_cos_sin(1.0)
    with pytest.raises(errors.ValidationError):
        lambertian_cos_sin(-0.1)


def test_plane_surface_position_and_direction() -> None:
    source = build_plane_gamma_source()
    position, direction = sample_plane_surface(source, 0.0, 1.0, 0.5, 0.0)
    assert position[0] == -0.5 * source.size_x_mm
    assert position[1] == 0.5 * source.size_y_mm
    assert position[2] == source.z_mm
    assert direction[2] <= 0.0
    assert math.isclose(math.sqrt(sum(component**2 for component in direction)), 1.0)
    # u_cos -> 1 gives normal incidence toward -z
    _, forward = sample_plane_surface(source, 0.5, 0.5, 0.999999, 0.0)
    assert forward[2] == pytest.approx(-math.sqrt(0.999999), abs=1e-6)
    assert forward[0] == pytest.approx(0.0, abs=1e-3)
    # u_cos = 0 gives a grazing direction in the plane
    _, grazing = sample_plane_surface(source, 0.5, 0.5, 0.0, 0.0)
    assert grazing[2] == pytest.approx(0.0)
    assert math.isclose(math.hypot(grazing[0], grazing[1]), 1.0)


def test_sphere_surface_point_is_inward_and_direction_unit() -> None:
    source = build_sphere_gamma_source()
    for u in (0.0, 0.3, 0.5, 0.7, 1.0):
        position, direction = sample_sphere_surface(source, u, 0.25, 0.1, 0.6)
        radius = math.sqrt(sum(component**2 for component in position))
        assert math.isclose(radius, source.radius_mm, rel_tol=1e-12)
        inward_dot = -sum(p * d for p, d in zip(position, direction, strict=True))
        assert inward_dot >= 0.0
        assert math.isclose(
            math.sqrt(sum(component**2 for component in direction)), 1.0, abs_tol=1e-12
        )
    # degenerate north/south pole must fall back to a valid frame
    _, pole_direction = sample_sphere_surface(source, 0.0, 0.0, 0.5, 0.5)
    assert math.isclose(
        math.sqrt(sum(component**2 for component in pole_direction)), 1.0, abs_tol=1e-12
    )


def _draws(seed: int, count: int) -> np.ndarray:
    return np.random.default_rng(seed).random(count)


def _column_draws(seed: int, schedule: sources.ColumnSchedule, workers: int) -> dict:
    """Emulate the per-column reseeded streams over a worker partition."""
    out: dict[int, np.ndarray] = {}
    for chunk in schedule.slices(workers):
        for column, count in zip(chunk.columns, chunk.counts, strict=True):
            out[column] = _draws(column_seed(seed, column), count)
    return out


def test_per_column_streams_are_worker_count_independent() -> None:
    axis = sources.make_primary_axis(np.linspace(0.0, 100.0, 11))
    schedule = sources.ColumnSchedule.fixed_total(axis, 37)
    seed = 908136382
    single = _column_draws(seed, schedule, 1)
    split = _column_draws(seed, schedule, 4)
    assert set(single) == set(split)
    for column in single:
        assert np.array_equal(single[column], split[column])
    # same seed twice reproduces; a different seed changes the stream
    assert np.array_equal(single[3], _column_draws(seed, schedule, 1)[3])
    assert not np.array_equal(single[3], _draws(column_seed(seed + 1, 3), len(single[3])))


def test_block_seeds_partition_independently() -> None:
    event_count = 5000
    block = 1024
    blocks = [event // block for event in range(0, event_count, block)]
    first = {index: block_seed(1, index) for index in blocks}
    second = {index: block_seed(1, index) for index in reversed(blocks)}
    assert first == second
