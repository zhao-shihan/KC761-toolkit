"""Source configurations for the KC761 simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Box:
    kind: str = "box"
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0

    def volume_cm3(self) -> float:
        return self.size_x / 10.0 * (self.size_y / 10.0) * (self.size_z / 10.0)


@dataclass(frozen=True)
class Cylinder:
    kind: str = "cylinder"
    radius: float = 0.0
    half_length: float = 0.0
    axis: str = "z"

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (2.0 * self.half_length / 10.0)


@dataclass(frozen=True)
class Disk:
    kind: str = "disk"
    radius: float = 0.0
    thickness: float = 0.0

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (self.thickness / 10.0)


@dataclass(frozen=True)
class Layer:
    material: str
    thickness: float
    active: bool = False


@dataclass(frozen=True)
class Sandwich:
    kind: str = "sandwich"
    radius: float = 0.0
    layers: Tuple[Layer, ...] = ()

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
    kind: str = "sphere"
    radius: float = 0.0

    def volume_cm3(self) -> float:
        return 4.0 / 3.0 * math.pi * (self.radius / 10.0) ** 3


@dataclass(frozen=True)
class Tube:
    kind: str = "tube"
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    half_length: float = 0.0
    axis: str = "z"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    nuclide: Tuple[int, int]
    geometry: Box | Cylinder | Disk | Sandwich | Sphere
    material: str
    density: Optional[float] = None
    mass_g: Optional[float] = None
    container: Optional[Tube] = None
    container_offset: Optional[Tuple[float, float, float]] = None
    nucleus_limits: Optional[Tuple[int, int, int, int]] = None
    threshold_years: float = 1.0e60

    @property
    def effective_density(self) -> Optional[float]:
        if self.density is not None:
            return self.density
        if self.mass_g is not None:
            return self.mass_g / self.geometry.volume_cm3()
        return None


def _tube_center_z(tube: Tube, near_z: float) -> float:
    if tube.axis == "y":
        return near_z + tube.outer_radius
    if tube.axis == "z":
        return near_z + tube.half_length
    raise ValueError(f"unsupported tube axis: {tube.axis!r}")


DETECTOR_GAP_MM = 1.0


def container_center_z(spec: SourceSpec, detector_front_z: float) -> Optional[float]:
    if spec.container is None:
        return None
    return _tube_center_z(spec.container, detector_front_z + DETECTOR_GAP_MM)


def source_center_z(spec: SourceSpec, detector_front_z: float) -> float:
    near_z = detector_front_z + DETECTOR_GAP_MM
    g = spec.geometry

    if spec.container is not None:
        z = _tube_center_z(spec.container, near_z)
        offset = spec.container_offset
        if offset is not None:
            z += offset[2]
        return z

    if isinstance(g, Box):
        return near_z + g.size_z / 2.0
    if isinstance(g, Disk):
        return near_z + g.thickness / 2.0
    if isinstance(g, Sandwich):
        return near_z + g.total_thickness / 2.0
    if isinstance(g, Sphere):
        return near_z + g.radius
    if isinstance(g, Cylinder):
        if g.axis != "z":
            raise ValueError(
                f"a {g.axis}-axis source cylinder outside a container is not "
                f"supported (axis must be 'z' to face the detector)"
            )
        return near_z + g.half_length
    raise ValueError(f"unknown geometry {g!r}")


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
    ),
}
