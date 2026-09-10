"""Axis contract: strictly increasing edges plus a unit string.

Every product axis is an :class:`Axis`. Units are values from the frozen
vocabulary below and also appear in key names (docs/formats.md D-14).
Structural helpers (centers, widths, slicing, equality) live here so no other
module reimplements one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kc761.core.binning import MAX_CHANNELS
from kc761.errors import SchemaError

UNIT_CHANNEL = "channel"
UNIT_KEV = "kev"
UNIT_MM = "mm"
UNIT_COUNTS = "counts"
UNIT_DIMENSIONLESS = "dimensionless"

UNITS: tuple[str, ...] = (
    UNIT_CHANNEL,
    UNIT_KEV,
    UNIT_MM,
    UNIT_COUNTS,
    UNIT_DIMENSIONLESS,
)


@dataclass(frozen=True)
class Axis:
    """One histogram axis: name, monotonically increasing edges and unit."""

    name: str
    edges: NDArray[np.float64]
    unit: str

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("axis name must be non-empty")
        if self.unit not in UNITS:
            raise SchemaError(
                f"axis {self.name!r}: unknown unit {self.unit!r}; expected one of {UNITS}"
            )
        edges = np.asarray(self.edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise SchemaError(
                f"axis {self.name!r}: edges must be one-dimensional with >= 2 entries"
            )
        if not np.isfinite(edges).all():
            raise SchemaError(f"axis {self.name!r}: edges must be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise SchemaError(f"axis {self.name!r}: edges must be strictly increasing")
        object.__setattr__(self, "edges", edges)

    @property
    def n_bins(self) -> int:
        return int(self.edges.size) - 1

    def centers(self) -> NDArray[np.float64]:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def widths(self) -> NDArray[np.float64]:
        return np.diff(self.edges)

    def slice(self, low: int, high: int) -> Axis:
        """Return the sub-axis covering bins ``[low, high]`` inclusive."""
        if not 0 <= low <= high < self.n_bins:
            raise SchemaError(
                f"axis {self.name!r}: bin range [{low}, {high}] outside [0, {self.n_bins - 1}]"
            )
        return Axis(name=self.name, edges=self.edges[low : high + 2], unit=self.unit)


def check_same_edges(left: Axis, right: Axis) -> None:
    """Require two axes to share identical edges (exact comparison)."""
    if left.edges.shape != right.edges.shape or not np.array_equal(left.edges, right.edges):
        raise SchemaError(
            f"axes {left.name!r} and {right.name!r} have different edges"
        )


def channel_axis(n_channels: int) -> Axis:
    """Uniform channel axis ``-0.5 .. n_channels - 0.5`` (F-BIN-1)."""
    if n_channels < 1:
        raise SchemaError(f"n_channels must be >= 1, got {n_channels!r}")
    if n_channels > MAX_CHANNELS:
        raise SchemaError(
            f"n_channels={n_channels} exceeds the supported maximum {MAX_CHANNELS} (D-52)"
        )
    return Axis(
        name="channel",
        edges=np.arange(-0.5, n_channels + 0.5, 1.0),
        unit=UNIT_CHANNEL,
    )


def energy_axis(edges_kev: NDArray[np.float64], *, name: str = "energy_kev") -> Axis:
    """Variable energy axis in keV (F-BIN-1)."""
    return Axis(name=name, edges=np.asarray(edges_kev, dtype=np.float64), unit=UNIT_KEV)
