"""Bezier scale tests (F-CAL-2).

Analytic derivatives are compared to central differences (auxiliary only,
D-65). The rank structure documents the model class: the four-parameter family
is identifiable for a genuine parabola (``s0`` not at the window midpoint) and
degenerates to the quadratic-polynomial (degree-2 Bernstein) subfamily exactly
at ``s0 = midpoint`` or for a constant scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from kc761.calib.scaling import (
    N_SCALE,
    scale_bounds,
    scale_curve,
    scale_curve_grad,
    scale_names,
)
from kc761.errors import ValidationError


def test_bounds_cover_s0_and_values() -> None:
    bounds = scale_bounds(2.0, 100, 300)
    assert len(bounds) == N_SCALE
    assert bounds[0][0] > 100.0
    assert bounds[0][1] < 300.0
    for low, high in bounds[1:]:
        assert low == pytest.approx(0.02)
        assert high == pytest.approx(6.0)


def test_endpoints_and_constant_scale() -> None:
    params = np.array([173.0, 0.9, 1.0, 1.1])
    assert scale_curve(params, np.array([100.0]), 100, 300)[0] == pytest.approx(0.9)
    assert scale_curve(params, np.array([300.0]), 100, 300)[0] == pytest.approx(1.1)
    flat = np.array([173.0, 1.3, 1.3, 1.3])
    values = scale_curve(flat, np.linspace(100, 300, 11), 100, 300)
    assert np.allclose(values, 1.3)


def test_scale_derivatives_match_central_differences() -> None:
    channels = np.linspace(105.0, 295.0, 23)
    params = np.array([120.0, 0.83, 1.04, 1.19])
    value, gradient = scale_curve_grad(params, channels, 100, 300)
    assert value.shape == channels.shape
    assert gradient.shape == (N_SCALE, channels.size)
    step = 1e-6
    for index in range(N_SCALE):
        up = params.copy()
        down = params.copy()
        up[index] += step
        down[index] -= step
        finite_difference = (
            scale_curve(up, channels, 100, 300)
            - scale_curve(down, channels, 100, 300)
        ) / (2.0 * step)
        assert np.allclose(gradient[index], finite_difference, rtol=1e-6, atol=1e-9)


def test_rank_four_for_a_genuine_parabola() -> None:
    """A non-midpoint control channel is identifiable (rank 4, D-103)."""
    channels = np.linspace(0.0, 63.0, 64)
    params = np.array([12.0, 0.9, 1.0, 1.1])
    _, gradient = scale_curve_grad(params, channels, 0, 63)
    matrix = gradient.reshape(N_SCALE, -1).T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert singular_values[-1] / singular_values[0] > 1e-4


def test_rank_three_for_the_polynomial_subfamily() -> None:
    """At the midpoint the family collapses to quadratic polynomials (gauge)."""
    channels = np.linspace(0.0, 63.0, 64)
    params = np.array([31.5, 0.9, 1.0, 1.1])
    _, gradient = scale_curve_grad(params, channels, 0, 63)
    matrix = gradient.reshape(N_SCALE, -1).T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert singular_values[-1] / singular_values[0] < 1e-8


def test_constant_scale_has_zero_s0_derivative() -> None:
    channels = np.linspace(0.0, 63.0, 64)
    _, gradient = scale_curve_grad(np.array([12.0, 1.0, 1.0, 1.0]), channels, 0, 63)
    assert np.max(np.abs(gradient[0])) == 0.0


def test_scale_names_are_labelled() -> None:
    assert scale_names("run-1", 0) == ("s0_run_1", "s1_run_1", "s2_run_1", "s3_run_1")


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValidationError):
        scale_bounds(0.0, 0, 10)
    with pytest.raises(ValidationError):
        scale_curve(np.array([5.0, 1.0, 1.0]), np.array([3.0]), 0, 10)
    with pytest.raises(ValidationError):
        scale_curve(np.array([15.0, 1.0, 1.0, 1.0]), np.array([3.0]), 0, 10)
