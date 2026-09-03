"""Source specifications for the KC761 simulation (pure data, no Geant4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangular source with full edge lengths in mm."""

    size_x: float
    size_y: float
    size_z: float

    def volume_cm3(self) -> float:
        return self.size_x / 10.0 * (self.size_y / 10.0) * (self.size_z / 10.0)


@dataclass(frozen=True)
class Cylinder:
    """Cylinder with dimensions in mm; ``axis`` is the symmetry axis."""

    radius: float
    half_length: float
    axis: str = "z"

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (
            2.0 * self.half_length / 10.0
        )


@dataclass(frozen=True)
class Disk:
    """Flat circular slab in mm, symmetric axis along z."""

    radius: float
    thickness: float

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (self.thickness / 10.0)


@dataclass(frozen=True)
class Layer:
    """Single planar layer of a :class:`Sandwich`, thickness in mm."""

    material: str
    thickness: float
    active: bool = False


@dataclass(frozen=True)
class Sandwich:
    """Stack of cylindrical layers in mm along z (e.g. foil sources)."""

    radius: float
    layers: Tuple[Layer, ...]

    @property
    def active_layers(self) -> Tuple[Layer, ...]:
        return tuple(layer for layer in self.layers if layer.active)

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def active_thickness(self) -> float:
        return sum(layer.thickness for layer in self.active_layers)

    @property
    def active_center_offset(self) -> float:
        """z of the active-layer centroid relative to the stack center."""
        active = self.active_layers
        if not active:
            return 0.0
        z = -0.5 * self.total_thickness
        centroid = 0.0
        for layer in self.layers:
            z += 0.5 * layer.thickness
            if layer.active:
                centroid += layer.thickness * z
            z += 0.5 * layer.thickness
        return centroid / self.active_thickness

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (
            self.total_thickness / 10.0
        )


@dataclass(frozen=True)
class Sphere:
    """Full sphere with outer radius in mm."""

    radius: float

    def volume_cm3(self) -> float:
        return 4.0 / 3.0 * math.pi * (self.radius / 10.0) ** 3


@dataclass(frozen=True)
class Tube:
    """Hollow cylinder (shell) surrounding a source, dimensions in mm."""

    inner_radius: float
    outer_radius: float
    half_length: float
    axis: str = "z"


@dataclass(frozen=True)
class SourceSpec:
    """Complete description of one radioactive source.

    Lengths are in mm. Density resolution: an explicit ``density``
    (g/cm^3) takes precedence; otherwise a ``mass_g`` divided by the
    geometry volume is used. Exactly one of the two may be set; leaving
    both undefined is only valid for NIST materials.
    """

    key: str
    name: str
    nuclide: Tuple[int, int]
    geometry: Box | Cylinder | Disk | Sandwich | Sphere
    material: str
    density: Optional[float] = None
    mass_g: Optional[float] = None
    container: Optional[Tube] = None
    container_material: Optional[str] = None
    container_offset: Optional[Tuple[float, float, float]] = None
    nucleus_limits: Optional[Tuple[int, int, int, int]] = None
    threshold_years: float = 1.0e60

    def __post_init__(self) -> None:
        if (self.container is None) != (self.container_material is None):
            raise ValueError(
                f"source {self.key!r}: 'container' and 'container_material' "
                f"must be given together"
            )
        if self.container is None and self.container_offset is not None:
            raise ValueError(
                f"source {self.key!r}: 'container_offset' requires a "
                f"'container'"
            )
        if self.density is not None and self.mass_g is not None:
            raise ValueError(
                f"source {self.key!r}: 'density' and 'mass_g' are mutually "
                f"exclusive"
            )

    @property
    def effective_density(self) -> Optional[float]:
        if self.density is not None:
            return self.density
        if self.mass_g is not None:
            return self.mass_g / self.geometry.volume_cm3()
        return None


SOURCES: dict[str, SourceSpec] = {
    "k40": SourceSpec(
        key="k40",
        name="K-40 in anhydrous potassium carbonate (13x7.5x6 cm, 500 g)",
        nuclide=(19, 40),
        geometry=Box(size_x=130.0, size_y=75.0, size_z=60.0),
        material="K2CO3",
        mass_g=500.0,
    ),
    "lu176": SourceSpec(
        key="lu176",
        name="Lu-176 in lutetium oxide powder (3x3x0.5 cm, 10 g)",
        nuclide=(71, 176),
        geometry=Box(size_x=30.0, size_y=30.0, size_z=5.0),
        material="Lu2O3",
        mass_g=10.0,
    ),
    "am241": SourceSpec(
        key="am241",
        name="Am-241 in gold-foil sandwich (diameter 2 mm, "
             "2 um Au / 1 um source / 2 um Au)",
        nuclide=(95, 241),
        geometry=Sandwich(
            radius=1.0,
            layers=(
                Layer(material="G4_Au", thickness=0.002),
                Layer(material="G4_Au", thickness=0.001, active=True),
                Layer(material="G4_Au", thickness=0.002),
            ),
        ),
        material="G4_Au",
        nucleus_limits=(241, 241, 95, 95),
    ),
    "th232": SourceSpec(
        key="th232",
        name="Th-232 in thorium nitrate pentahydrate cylinder "
             "(r 0.87 cm x 1.5 cm, 10 g) in glass tube",
        nuclide=(90, 232),
        geometry=Cylinder(radius=8.7, half_length=7.5, axis="y"),
        material="Th(NO3)4-5H2O",
        mass_g=10.0,
        container=Tube(inner_radius=10.0, outer_radius=11.0,
                       half_length=25.0, axis="y"),
        container_material="G4_Pyrex_Glass",
        container_offset=(0.0, 0.0, -1.3),
    ),
    "ra226": SourceSpec(
        key="ra226",
        name="Ra-226 in glass ball (diameter 5 mm) in stainless-steel tube",
        nuclide=(88, 226),
        geometry=Sphere(radius=2.5),
        material="G4_GLASS_PLATE",
        container=Tube(inner_radius=2.5, outer_radius=3.0,
                       half_length=2.5, axis="z"),
        container_material="G4_STAINLESS-STEEL",
    ),
}
