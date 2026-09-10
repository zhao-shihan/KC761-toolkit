"""Rebinning and folding projections between energy grids.

Formula IDs (docs/derivations.md): F-PROJ-1 .. F-PROJ-2.

* F-PROJ-1: exact bin-overlap weights between source and target binnings; each
  source bin is fully covered by the target (otherwise the projection would
  silently drop mass and :func:`build_projection_plan` fails loudly).
* F-PROJ-2: values project as ``W @ values``; independent variances project as
  ``W**2 @ variances`` (weights are the squared fractions). Identical grids
  give the identity matrix, so the projection is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.core._checks import as_float_array
from kc761.core.binning import EnergyGrid
from kc761.errors import ValidationError

COVERAGE_TOL = 1e-9
"""F-PROJ-1 tolerance for full coverage of a source bin."""


@dataclass(frozen=True)
class ProjectionPlan:
    """Frozen projection geometry and weights (F-PROJ-1)."""

    source_edges_kev: NDArray[np.float64]
    target_edges_kev: NDArray[np.float64]
    weights: sparse.csr_matrix


def build_projection_plan(
    source_edges_kev: NDArray[np.float64],
    target_edges_kev: NDArray[np.float64],
) -> ProjectionPlan:
    """Build the exact overlap projection between two binnings (F-PROJ-1).

    ``weights[t, s]`` is the fraction of source bin ``s`` that falls into
    target bin ``t``. Target bins must cover every source bin; otherwise a
    :class:`kc761.errors.ValidationError` is raised.
    """
    source_edges = EnergyGrid(
        edges_kev=as_float_array("source_edges_kev", source_edges_kev, ndim=1)
    ).edges_kev
    target_edges = EnergyGrid(
        edges_kev=as_float_array("target_edges_kev", target_edges_kev, ndim=1)
    ).edges_kev

    # Piecewise-constant overlap: split the union of both edge arrays into
    # segments; each segment belongs to exactly one source and one target bin.
    breakpoints = np.union1d(source_edges, target_edges)
    mids = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    widths = np.diff(breakpoints)
    source_index = np.searchsorted(source_edges, mids, side="right") - 1
    target_index = np.searchsorted(target_edges, mids, side="right") - 1
    inside = (
        (source_index >= 0)
        & (source_index < source_edges.size - 1)
        & (target_index >= 0)
        & (target_index < target_edges.size - 1)
    )
    if not np.any(inside):
        raise ValidationError("source and target binnings do not overlap")
    source_bin = source_index[inside]
    target_bin = target_index[inside]
    segment = widths[inside]
    n_source = source_edges.size - 1
    n_target = target_edges.size - 1
    weights = sparse.csr_matrix(
        (
            segment,
            (target_bin, source_bin),
        ),
        shape=(n_target, n_source),
    )
    coverage = np.asarray(weights.sum(axis=0)).ravel() / np.diff(source_edges)
    if np.any(np.abs(coverage - 1.0) > COVERAGE_TOL):
        worst = int(np.argmin(coverage))
        raise ValidationError(
            f"target binning does not fully cover source bin {worst} "
            f"(coverage {coverage[worst]:.12g})"
        )
    weights = sparse.csr_matrix(weights / np.diff(source_edges)[None, :])
    return ProjectionPlan(
        source_edges_kev=source_edges,
        target_edges_kev=target_edges,
        weights=weights,
    )


def project(
    values: NDArray[np.float64],
    variances: NDArray[np.float64] | None,
    plan: ProjectionPlan,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Project values (and variances when given) onto the target grid (F-PROJ-2)."""
    values_array = as_float_array("values", values, ndim=1)
    n_source = plan.source_edges_kev.size - 1
    if values_array.size != n_source:
        raise ValidationError(
            f"values has {values_array.size} bins, projection plan expects {n_source}"
        )
    if plan.weights.shape[1] != n_source:
        raise ValidationError("projection plan weights do not match the source edges")
    projected = np.asarray(plan.weights @ values_array, dtype=np.float64)
    if variances is None:
        return projected, None
    variances_array = as_float_array("variances", variances, ndim=1)
    if variances_array.shape != values_array.shape:
        raise ValidationError("variances and values must have the same shape")
    if np.any(variances_array < 0.0):
        raise ValidationError("variances must be non-negative")
    propagated = np.asarray(plan.weights.power(2) @ variances_array, dtype=np.float64)
    return projected, propagated
