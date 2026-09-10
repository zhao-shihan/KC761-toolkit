"""Frozen detector geometry values with per-field provenance (D-34).

The numeric values are carried over unchanged from the pre-rewrite detector
construction (D-34); only their organisation changed: they now live in one
frozen dataclass with an explicit provenance note per field. ``assumed`` marks
a value whose source measurement was not available to the rewrite; ``measured``
marks a value that is a physical property of the instrument.

No Geant4 import happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from kc761.errors import ValidationError

_PROVENANCE: Final[dict[str, str]] = {
    "crystal_half_x_mm": "assumed",
    "crystal_half_y_mm": "assumed",
    "crystal_half_z_mm": "assumed",
    "housing_wall_mm": "assumed",
    "detector_gap_mm": "assumed",
    "world_half_mm": "assumed",
}


@dataclass(frozen=True)
class DetectorGeometry:
    """Crystal, housing and world dimensions in mm (D-34).

    Derived quantities are properties so the housing thickness can never drift
    away from the crystal size in a second literal.
    """

    crystal_half_x_mm: float = 5.0
    crystal_half_y_mm: float = 5.0
    crystal_half_z_mm: float = 12.7
    housing_wall_mm: float = 1.0
    detector_gap_mm: float = 1.0
    world_half_mm: float = 150.0

    def __post_init__(self) -> None:
        for name in _PROVENANCE:
            value = float(getattr(self, name))
            if not (value > 0.0):
                raise ValidationError(f"geometry field {name} must be positive, got {value!r}")

    @property
    def housing_half_x_mm(self) -> float:
        return self.crystal_half_x_mm + self.housing_wall_mm

    @property
    def housing_half_y_mm(self) -> float:
        return self.crystal_half_y_mm + self.housing_wall_mm

    @property
    def housing_half_z_mm(self) -> float:
        return self.crystal_half_z_mm + self.housing_wall_mm

    @property
    def detector_front_z_mm(self) -> float:
        """z of the housing front face (the matrix plane source position)."""
        return self.housing_half_z_mm


DEFAULT_GEOMETRY: Final = DetectorGeometry()
"""The unchanged pre-rewrite detector geometry (D-34)."""


def geometry_provenance() -> dict[str, str]:
    """Return the field -> provenance mapping (``measured``/``assumed``)."""
    return dict(_PROVENANCE)
