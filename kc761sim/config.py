"""Fixed source configurations for the kc761 gamma-spectrometry simulation.

The user selects exactly one of the five sources with a command-line flag.
Each :class:`SourceSpec` describes the nuclide, the geometry the activity is
uniformly distributed in, the material, and the radioactive-decay handling.

All linear dimensions are given in **mm** and volumes/densities in **cm3** /
**g/cm3** as plain numbers so that this module stays free of the Geant4
bindings; conversion to Geant4 internal units happens at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
class Disk:
    """Thin circular disk / foil, axis along z (mm)."""

    kind: str = "disk"
    radius: float = 0.0
    thickness: float = 0.0

    def volume_cm3(self) -> float:
        return math.pi * (self.radius / 10.0) ** 2 * (self.thickness / 10.0)


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
    geometry: Box | Disk | Sphere  # where the activity is distributed
    material: str  # material key, see kc761sim.materials
    #: density of the source material in g/cm3; ``None`` means either the NIST
    #: material density (no ``mass_g`` set) or mass/volume (``mass_g`` set)
    density: Optional[float] = None
    #: total source mass in g, used to derive the density from the volume
    mass_g: Optional[float] = None
    #: optional encasing tube the source sits in (e.g. Th-232, Ra-226)
    container: Optional[Tube] = None
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


def _tube(tube: Tube, near_z: float) -> Tuple[float, float]:
    """Return (tube_center_z, tube_center_along_axis) for a tube whose outer
    surface (for axis ``y``) or end face (for axis ``z``) sits at ``near_z``.

    ``near_z`` is the z-coordinate of the tube's closest surface to the
    detector; the tube is centred on the z axis (x = y = 0).
    """
    if tube.axis == "y":
        # nearest point of the cylinder wall in +z sits at z = center - R_outer
        z = near_z + tube.outer_radius
    elif tube.axis == "z":
        # tube end face at z = near_z
        z = near_z + tube.half_length
    else:
        raise ValueError(f"unsupported tube axis: {tube.axis!r}")
    return z


#: Face-to-face gap between the detector housing and every source, along z (mm).
DETECTOR_GAP_MM = 1.0


def source_center_z(spec: SourceSpec, detector_front_z: float) -> float:
    """World-frame z of the source volume centre for a source facing the
    detector front face located at ``detector_front_z`` (mm)."""
    near_z = detector_front_z + DETECTOR_GAP_MM
    g = spec.geometry
    if isinstance(g, Box):
        return near_z + g.size_z / 2.0
    if isinstance(g, Disk):
        return near_z + g.thickness / 2.0
    if isinstance(g, Sphere):
        if spec.container is not None:
            return _tube(spec.container, near_z)
        return near_z + g.radius
    raise ValueError(f"unknown geometry {g!r}")


SOURCES: dict[str, SourceSpec] = {
    # K-40 uniformly distributed in a 13 cm x 8 cm x 6 cm block of anhydrous
    # potassium carbonate (K2CO3), total mass 500 g.  The 13 cm x 8 cm face
    # faces the detector, i.e. 6 cm thickness along z.  The density follows
    # from mass and volume (0.8013 g/cm3).
    "k40": SourceSpec(
        key="k40",
        name="K-40 in anhydrous potassium carbonate (13x8x6 cm, 500 g)",
        nuclide=(19, 40),
        geometry=Box(size_x=130.0, size_y=80.0, size_z=60.0),
        material="K2CO3",
        mass_g=500.0,
    ),
    # Lu-176 uniformly distributed in a 3 cm x 3 cm x 0.5 cm slab of lutetium
    # oxide (Lu2O3), 3 cm x 3 cm face towards the detector (0.5 cm along z).
    # Density: compact/packed Lu2O3 powder, taken as 5.5 g/cm3 (the crystal
    # density of Lu2O3 is 9.42 g/cm3; a packed powder reaches roughly 55-60 %
    # of the crystal density).  This value is a modelling choice.
    "lu176": SourceSpec(
        key="lu176",
        name="Lu-176 in lutetium oxide powder (3x3x0.5 cm)",
        nuclide=(71, 176),
        geometry=Box(size_x=30.0, size_y=30.0, size_z=5.0),
        material="Lu2O3",
        density=5.5,
    ),
    # Am-241 uniformly distributed in a circular gold foil, diameter 2 mm,
    # thickness 3 um, facing the detector.  The long-lived daughter Np-237
    # (2.14 Myr) is excluded from the decay by restricting RDM to Am-241 only.
    "am241": SourceSpec(
        key="am241",
        name="Am-241 in gold foil (diameter 2 mm, 3 um thick)",
        nuclide=(95, 241),
        geometry=Disk(radius=1.0, thickness=0.003),
        material="G4_Au",
        nucleus_limits=(241, 241, 95, 95),
    ),
    # Th-232 uniformly distributed in a thorium-nitrate pentahydrate sphere
    # (diameter 1.5 cm, 10 g) at the centre of a glass tube (outer diameter
    # 2.2 cm, wall 1 mm, length 5 cm) whose axis runs along y.  The sphere
    # density follows from mass and volume (5.66 g/cm3).  The source is in
    # secular equilibrium, hence the full decay chain is simulated.
    "th232": SourceSpec(
        key="th232",
        name="Th-232 in thorium nitrate pentahydrate sphere (diameter 1.5 cm, 10 g)",
        nuclide=(90, 232),
        geometry=Sphere(radius=7.5),
        material="Th(NO3)4-5H2O",
        mass_g=10.0,
        container=Tube(inner_radius=10.0, outer_radius=11.0,
                       half_length=25.0, axis="y"),
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
