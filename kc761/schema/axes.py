"""Axis contract: strictly increasing edges plus a unit string.

Every product axis is an :class:`Axis`. Units are values from the frozen
vocabulary below and also appear in key names (docs/formats.md D-14).
Structural helpers (centers, widths, slicing, equality) live here so no other
module reimplements one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761.core.binning import MAX_CHANNELS
from kc761.core.model import PARAM_NAMES_REPORTED
from kc761.errors import SchemaError

UNIT_CHANNEL = "channel"
UNIT_KEV = "kev"
UNIT_MM = "mm"
UNIT_COUNTS = "counts"
UNIT_DIMENSIONLESS = "dimensionless"

CHANNEL_AXIS_NAME = "channel"
DEPOSITION_AXIS_NAME = "deposition_energy_kev"
PRIMARY_AXIS_NAME = "primary_energy_kev"
ENERGY_AXIS_NAME = "energy_kev"
"""Canonical axis names (single source; D-14 units live in the unit constants)."""

PARAM_AXIS_NAME = "reported_parameter"
"""Axis name of the ``param_cov`` matrix (D-13)."""

UNIT_DISPLAY: Final[dict[str, str]] = {
    UNIT_CHANNEL: "channel",
    UNIT_KEV: "keV",
    UNIT_MM: "mm",
    UNIT_COUNTS: "counts",
    UNIT_DIMENSIONLESS: "dimensionless",
}
"""Display spelling of each unit in human-readable axis titles (D-170)."""

_UNITLESS_DISPLAY: Final[frozenset[str]] = frozenset(
    {UNIT_CHANNEL, UNIT_COUNTS, UNIT_DIMENSIONLESS}
)

HUMAN_AXIS_LABELS: Final[dict[str, str]] = {
    CHANNEL_AXIS_NAME: "Channel",
    DEPOSITION_AXIS_NAME: "Deposition energy",
    PRIMARY_AXIS_NAME: "Primary energy",
    ENERGY_AXIS_NAME: "Energy",
    PARAM_AXIS_NAME: "Reported parameter",
}
"""Human axis labels; the canonical machine name stays in the axis ``fName``."""

AXIS_UNITS: Final[dict[str, str]] = {
    CHANNEL_AXIS_NAME: UNIT_CHANNEL,
    DEPOSITION_AXIS_NAME: UNIT_KEV,
    PRIMARY_AXIS_NAME: UNIT_KEV,
    ENERGY_AXIS_NAME: UNIT_KEV,
    PARAM_AXIS_NAME: UNIT_DIMENSIONLESS,
}
"""Canonical axis name -> unit; units are no longer parsed from the title."""

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
        name=CHANNEL_AXIS_NAME,
        edges=np.arange(-0.5, n_channels + 0.5, 1.0),
        unit=UNIT_CHANNEL,
    )


def energy_axis(
    edges_kev: NDArray[np.float64], *, name: str = ENERGY_AXIS_NAME
) -> Axis:
    """Variable energy axis in keV (F-BIN-1)."""
    return Axis(name=name, edges=np.asarray(edges_kev, dtype=np.float64), unit=UNIT_KEV)


def reported_parameter_axis() -> Axis:
    """Index axis of ``param_cov``; bin labels are ``PARAM_NAMES_REPORTED`` (D-13)."""
    edges = np.arange(-0.5, len(PARAM_NAMES_REPORTED) + 0.5, 1.0)
    return Axis(name=PARAM_AXIS_NAME, edges=edges, unit=UNIT_DIMENSIONLESS)


def human_axis_title(axis: Axis) -> str:
    """Human-readable axis title: ``Label (unit)``, unitless axes have no unit."""
    label = HUMAN_AXIS_LABELS.get(axis.name, axis.name)
    if axis.unit in _UNITLESS_DISPLAY:
        return label
    return f"{label} ({UNIT_DISPLAY.get(axis.unit, axis.unit)})"


def unit_for_axis_name(name: str) -> str:
    """Return the canonical unit for a canonical axis name (D-170)."""
    try:
        return AXIS_UNITS[name]
    except KeyError:
        raise SchemaError(
            f"unknown axis name {name!r}; expected one of {tuple(AXIS_UNITS)}"
        ) from None
