"""Fixed source configurations for the kc761 gamma-spectrometry simulation.

The user selects exactly one of the five sources with a command-line flag.
Each :class:`SourceSpec` describes the nuclide, the physical source geometry,
the material(s), and the radioactive-decay handling.  For simple sources the
decaying nuclide is distributed uniformly throughout the geometry; for a
layered :class:`Sandwich` only the layers marked ``active`` contain the
nuclide, the rest are inert cladding.

All linear dimensions are given in **mm** and volumes/densities in **cm3** /
**g/cm3** as plain numbers so that this module stays free of the Geant4
bindings; conversion to Geant4 internal units happens at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Box:
    """Axis-aligned box (mm, full side lengths)."""

    kind: str = "box"
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0

    def volume_cm3(self) -> float:
        return self.size_x / 10.0 * (self.size_y / 10.0) * (self.size_z / 10.0)


@dataclass(frozen=True)
class Cylinder:
    """Solid cylinder (mm), axis along one of the coordinate axes."""

    kind: str = "cylinder"
    radius: float = 0.0
    half_length: float = 0.0
    axis: str = "z"

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (2.0 * self.half_length / 10.0)


@dataclass(frozen=True)
class Disk:
    """Thin circular disk / foil, axis along z (mm)."""

    kind: str = "disk"
    radius: float = 0.0
    thickness: float = 0.0

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (self.thickness / 10.0)


@dataclass(frozen=True)
class Layer:
    """One layer of a layered (sandwich) source, stacked along z (mm).

    ``active`` marks the layers that carry the decaying nuclide; inactive
    layers are inert cladding (they attenuate the emitted gammas only).
    """

    material: str
    thickness: float
    active: bool = False


@dataclass(frozen=True)
class Sandwich:
    """Stack of coaxial circular layers along z (foil sandwich).

    Layers are ordered front-to-back: the first layer faces the detector
    (smaller z), the last layer is at the back.  Only layers with
    ``active = True`` contain the decaying nuclide.
    """

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
        """z-offset (mm) of the centroid of the active material from the
        sandwich centre; positive = toward the back (larger z)."""
        active = self.active_layers
        if not active:
            return 0.0
        z = -0.5 * self.total_thickness  # front face, in the sandwich frame
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
    """Sphere (mm)."""

    kind: str = "sphere"
    radius: float = 0.0

    def volume_cm3(self) -> float:
        return 4.0 / 3.0 * math.pi * (self.radius / 10.0) ** 3


@dataclass(frozen=True)
class Tube:
    """Cylindrical shell (container), axis along one of the coordinate axes."""

    kind: str = "tube"
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    half_length: float = 0.0
    axis: str = "z"


@dataclass(frozen=True)
class SourceSpec:
    """Everything needed to build, place and populate one source."""

    key: str
    name: str
    nuclide: Tuple[int, int]  # (Z, A) of the decaying isotope
    geometry: Box | Cylinder | Disk | Sandwich | Sphere  # physical source shape
    material: str  # material key of the active region, see kc761sim.materials
    #: density of the source material in g/cm3; ``None`` means either the NIST
    #: material density (no ``mass_g`` set) or mass/volume (``mass_g`` set)
    density: Optional[float] = None
    #: total source mass in g, used to derive the density from the volume
    mass_g: Optional[float] = None
    #: optional encasing tube the source sits in (e.g. Th-232, Ra-226)
    container: Optional[Tube] = None
    #: world-frame offset (mm) of the source-volume centre from the container
    #: centre; ``None`` = the source sits centred in the container (e.g. the
    #: Th-232 cylinder is offset so that it rests against the detector-side
    #: inner wall of the glass tube)
    container_offset: Optional[Tuple[float, float, float]] = None
    #: (AMin, AMax, ZMin, ZMax) window restricting which nuclides RDM decays;
    #: None means "all nuclides" (used to suppress long-lived daughters)
    nucleus_limits: Optional[Tuple[int, int, int, int]] = None
    #: RDM time threshold above which decays at rest are ignored (years).
    #: Geant4 >= 11.2 defaults this to 1 year, which would kill every decay of
    #: our long-lived parents; the value is raised per source (see README of
    #: the Geant4 rdecay01 example and ReleaseNotes.11.2).
    threshold_years: float = 1.0e60

    @property
    def effective_density(self) -> Optional[float]:
        if self.density is not None:
            return self.density
        if self.mass_g is not None:
            return self.mass_g / self.geometry.volume_cm3()
        return None


def _tube_center_z(tube: Tube, near_z: float) -> float:
    """z-coordinate of the centre of a tube whose outer surface (axis ``y``)
    or end face (axis ``z``) sits at ``near_z``.

    ``near_z`` is the z-coordinate of the tube's closest surface to the
    detector; the tube is centred on the z axis (x = y = 0).
    """
    if tube.axis == "y":
        # nearest point of the cylinder wall in +z sits at z = center - R_outer
        return near_z + tube.outer_radius
    if tube.axis == "z":
        # tube end face at z = near_z
        return near_z + tube.half_length
    raise ValueError(f"unsupported tube axis: {tube.axis!r}")


#: Face-to-face gap between the detector housing and every source, along z (mm).
DETECTOR_GAP_MM = 1.0


def container_center_z(spec: SourceSpec, detector_front_z: float) -> Optional[float]:
    """World-frame z of the container centre, or ``None`` if the source has no
    container.  The container faces the detector front face located at
    ``detector_front_z`` (mm)."""
    if spec.container is None:
        return None
    return _tube_center_z(spec.container, detector_front_z + DETECTOR_GAP_MM)


def source_center_z(spec: SourceSpec, detector_front_z: float) -> float:
    """World-frame z of the source volume centre for a source facing the
    detector front face located at ``detector_front_z`` (mm).

    For a source inside a container the position is derived from the
    container centre plus ``container_offset``; otherwise the source volume
    itself faces the detector with a face-to-face gap.
    """
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
    # K-40 uniformly distributed in a 13 cm x 7.5 cm x 6 cm block of anhydrous
    # potassium carbonate (K2CO3), total mass 500 g.  The 13 cm x 7.5 cm face
    # faces the detector, i.e. 6 cm thickness along z.  The density follows
    # from mass and volume (0.8547 g/cm3).
    "k40": SourceSpec(
        key="k40",
        name="K-40 in anhydrous potassium carbonate (13x7.5x6 cm, 500 g)",
        nuclide=(19, 40),
        geometry=Box(size_x=130.0, size_y=75.0, size_z=60.0),
        material="K2CO3",
        mass_g=500.0,
    ),
    # Lu-176 uniformly distributed in a 3 cm x 3 cm x 0.5 cm slab of lutetium
    # oxide (Lu2O3), total mass 10 g, 3 cm x 3 cm face towards the detector
    # (0.5 cm along z).  The density follows from mass and volume
    # (2.222 g/cm3).
    "lu176": SourceSpec(
        key="lu176",
        name="Lu-176 in lutetium oxide powder (3x3x0.5 cm, 10 g)",
        nuclide=(71, 176),
        geometry=Box(size_x=30.0, size_y=30.0, size_z=5.0),
        material="Lu2O3",
        mass_g=10.0,
    ),
    # Am-241 distributed in the central 1 um layer of a 2 um Au / 1 um /
    # 2 um Au gold-foil sandwich (diameter 2 mm) facing the detector.  The
    # decaying nuclide is produced only in the middle layer; the two outer
    # gold layers are pure inert cladding.  The long-lived daughter Np-237
    # (2.14 Myr) is excluded from the decay by restricting RDM to Am-241 only.
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
    # Th-232 uniformly distributed in a thorium-nitrate pentahydrate cylinder
    # (length 1.5 cm, radius 0.87 cm, 10 g) inside a glass tube (outer
    # diameter 2.2 cm, wall 1 mm, length 5 cm) whose axis runs along y.  The
    # cylinder axis is parallel to the tube axis and offset by
    # (10 - 8.7) = 1.3 mm toward the detector (-z), so it is tangent to the
    # inner wall of the tube on the detector side (as close to the detector as
    # physically possible).  The density follows from mass and volume
    # (2.80 g/cm3).  The source is in secular equilibrium, hence the full
    # decay chain is simulated.
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
        # (inner_radius - source_radius) = 10.0 - 8.7 = 1.3 mm toward -z
        container_offset=(0.0, 0.0, -1.3),
    ),
    # Ra-226 uniformly distributed in a glass ball (diameter 5 mm) at the
    # centre of a stainless-steel tube (outer diameter 6 mm, wall 0.5 mm,
    # length 5 mm) whose axis runs along z.  The full decay chain (secular
    # equilibrium, incl. the Bi-214 gammas) is simulated.
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
