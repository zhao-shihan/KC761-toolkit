"""Auxiliary checks for the overlap projection (F-PROJ-1/F-PROJ-2)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761.core import projection
from kc761.errors import ValidationError


def test_identity_projection_is_idempotent() -> None:
    edges = np.linspace(0.0, 10.0, 11)
    plan = projection.build_projection_plan(edges, edges)
    values = np.arange(10.0)
    variances = np.full(10, 4.0)
    projected, propagated = projection.project(values, variances, plan)
    assert np.allclose(projected, values)
    assert np.allclose(propagated, variances)


def test_coarsening_conserves_counts_and_variances() -> None:
    source = np.linspace(0.0, 10.0, 11)
    target = np.array([0.0, 2.0, 5.0, 10.0])
    plan = projection.build_projection_plan(source, target)
    values = np.ones(10)
    projected, propagated = projection.project(values, np.ones(10), plan)
    assert np.allclose(projected, [2.0, 3.0, 5.0])
    assert np.allclose(propagated, [2.0, 3.0, 5.0])


def test_refinement_shares_fractions() -> None:
    source = np.array([0.0, 10.0])
    target = np.linspace(0.0, 10.0, 11)
    plan = projection.build_projection_plan(source, target)
    projected, propagated = projection.project(np.array([10.0]), np.array([4.0]), plan)
    assert np.allclose(projected, 1.0)
    assert np.allclose(propagated, 4.0 * 0.1**2)


def test_incomplete_coverage_fails_loudly() -> None:
    with pytest.raises(ValidationError):
        projection.build_projection_plan(np.array([0.0, 10.0]), np.array([0.0, 5.0]))


def test_projection_validates_inputs() -> None:
    plan = projection.build_projection_plan(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValidationError):
        projection.project(np.zeros(3), None, plan)
    with pytest.raises(ValidationError):
        projection.project(np.zeros(2), np.ones(2) * -1.0, plan)
