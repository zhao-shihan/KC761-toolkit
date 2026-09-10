"""Source registry, geometry descriptions and matrix-mode sampling surfaces.

The values are the unchanged pre-rewrite data (D-34): the seven source keys
(D-122), the composite beta shield, the gold-foil sandwich, the tube/cup
containers and the matrix modes' plane/sphere surfaces (D-31). Only the
organisation changed: the registry carries an explicit provenance note per key
and the mode/geometry labels are frozen here once.

The matrix-mode primaries are drawn on the fixed source-mode Monte-Carlo axis
``0..4096 keV / 4096 bins`` from :mod:`kc761.core.binning` (D-121); the
deposition axis is the calibration product's ``C.y`` and is supplied by the
runner. No Geant4 import happens in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761.core.binning import source_mode_deposition_edges_kev
from kc761.errors import ValidationError


# --------------------------------------------------------------------------
# Geometry descriptions (millimetres)
# --------------------------------------------------------------------------
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
        return math.pi * (self.radius / 10.0) ** 2 * (2.0 * self.half_length / 10.0)


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
    layers: tuple[Layer, ...]

    @property
    def active_layers(self) -> tuple[Layer, ...]:
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
        return math.pi * (self.radius / 10.0) ** 2 * (self.total_thickness / 10.0)


@dataclass(frozen=True)
class Sphere:
    """Full sphere with outer radius in mm."""

    radius: float

    def volume_cm3(self) -> float:
        return 4.0 / 3.0 * math.pi * (self.radius / 10.0) ** 3


@dataclass(frozen=True)
class Ellipsoid:
    """Triaxial ellipsoid with semi-axis lengths in mm."""

    semi_x: float
    semi_y: float
    semi_z: float

    def volume_cm3(self) -> float:
        return (
            4.0
            / 3.0
            * math.pi
            * (self.semi_x / 10.0)
            * (self.semi_y / 10.0)
            * (self.semi_z / 10.0)
        )


@dataclass(frozen=True)
class Tube:
    """Hollow cylinder (shell) surrounding a source, dimensions in mm."""

    material: str
    inner_radius: float
    outer_radius: float
    half_length: float
    axis: str = "z"


@dataclass(frozen=True)
class Cup:
    """Vertical bottomed cylinder holding a source, axis along z.

    Dimensions in mm; walls and bottom share ``wall_thickness``. The interior
    spans ``inner_radius`` in radius and ``height - bottom_thickness`` above
    the bottom plate.
    """

    material: str
    inner_radius: float
    wall_thickness: float
    height: float
    bottom_thickness: float

    def __post_init__(self) -> None:
        if self.height <= self.bottom_thickness:
            raise ValidationError(
                f"Cup height ({self.height} mm) must exceed the bottom "
                f"thickness ({self.bottom_thickness} mm)"
            )

    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.wall_thickness


@dataclass(frozen=True)
class ShieldPart:
    """One axis-aligned box piece of a composite shield, in mm."""

    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]


@dataclass(frozen=True)
class BetaShield:
    """Composite beta shield the source assembly sits on.

    The assembly (container plus source) rests on the shield's top plane,
    ``top_z`` (derived as the highest part face).
    """

    material: str
    parts: tuple[ShieldPart, ...]

    @property
    def top_z(self) -> float:
        return max(part.center[2] + part.half_size[2] for part in self.parts)

    @property
    def lowest_z(self) -> float:
        return min(part.center[2] - part.half_size[2] for part in self.parts)


SourceGeometry = Box | Cylinder | Disk | Sandwich | Sphere | Ellipsoid


# --------------------------------------------------------------------------
# Source specification and registry (D-122)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSpec:
    """Complete description of one radioactive source.

    Lengths are in mm. Density resolution: an explicit ``density`` (g/cm^3)
    takes precedence; otherwise a ``mass_g`` divided by the geometry volume is
    used. Exactly one of the two may be set; leaving both undefined is only
    valid for NIST materials. ``provenance`` records the origin/assumption of
    the entry (D-122).
    """

    key: str
    name: str
    nuclide: tuple[int, int]
    geometry: SourceGeometry
    material: str
    provenance: str
    density: float | None = None
    mass_g: float | None = None
    container: Tube | Cup | None = None
    container_offset: tuple[float, float, float] | None = None
    shield: BetaShield | None = None
    nucleus_limits: tuple[int, int, int, int] | None = None
    threshold_years: float = 1.0e60

    def __post_init__(self) -> None:
        if not self.key or self.key != self.key.strip():
            raise ValidationError(
                f"source key must be a non-empty trimmed string, got {self.key!r}"
            )
        if not self.provenance:
            raise ValidationError(f"source {self.key!r}: provenance must be non-empty")
        if self.container is None and self.container_offset is not None:
            raise ValidationError(
                f"source {self.key!r}: 'container_offset' requires a 'container'"
            )
        if self.density is not None and self.mass_g is not None:
            raise ValidationError(
                f"source {self.key!r}: 'density' and 'mass_g' are mutually exclusive"
            )
        if isinstance(self.container, Cup):
            if self.shield is None:
                raise ValidationError(
                    f"source {self.key!r}: a Cup container must sit on a beta shield"
                )
            if self.container_offset is not None:
                raise ValidationError(
                    f"source {self.key!r}: 'container_offset' is not supported "
                    "for a Cup container"
                )
        if (
            isinstance(self.container, Tube)
            and self.container.axis != "z"
            and self.shield is not None
        ):
            raise ValidationError(
                f"source {self.key!r}: a {self.container.axis}-axis tube cannot "
                "sit on a beta shield"
            )
        if self.shield is not None and self.container is None:
            raise ValidationError(f"source {self.key!r}: a shielded source requires a container")

    @property
    def effective_density(self) -> float | None:
        if self.density is not None:
            return self.density
        if self.mass_g is not None:
            return self.mass_g / self.geometry.volume_cm3()
        return None

    @property
    def geometry_name(self) -> str:
        return _GEOMETRY_NAMES[type(self.geometry)]

    @property
    def geometry_param_mm(self) -> float:
        """Characteristic source half-extent in mm (D-120 meta field)."""
        return _characteristic_size_mm(self.geometry)


_GEOMETRY_NAMES: Final[dict[type, str]] = {
    Box: "box",
    Cylinder: "cylinder",
    Disk: "disk",
    Sandwich: "sandwich",
    Sphere: "sphere",
    Ellipsoid: "ellipsoid",
}


def _characteristic_size_mm(geometry: SourceGeometry) -> float:
    match geometry:
        case Box():
            return max(geometry.size_x, geometry.size_y, geometry.size_z) / 2.0
        case Cylinder():
            return max(geometry.radius, geometry.half_length)
        case Disk():
            return max(geometry.radius, geometry.thickness / 2.0)
        case Sandwich():
            return max(geometry.radius, geometry.total_thickness / 2.0)
        case Sphere():
            return geometry.radius
        case Ellipsoid():
            return max(geometry.semi_x, geometry.semi_y, geometry.semi_z)


#: Beta shield shared by the shielded source modes: a wide plate carries the
#: source assembly on its far-side top face, and two bumps rise from its
#: detector-side face to rest on the detector front surface.
BETA_SHIELD: Final = BetaShield(
    material="R4600",
    parts=(
        ShieldPart(
            name="BetaShieldPlate",
            center=(13.75, -3.0, 20.2),
            half_size=(40.25, 27.0, 5.0),
        ),
        ShieldPart(
            name="BetaShieldBumpA",
            center=(0.0, 0.0, 14.45),
            half_size=(11.5, 7.0, 0.75),
        ),
        ShieldPart(
            name="BetaShieldBumpB",
            center=(27.5, 0.0, 14.45),
            half_size=(11.5, 7.0, 0.75),
        ),
    ),
)

#: Unshielded base Ra-226 source; the shielded mode reuses its geometry and
#: container verbatim (via ``dataclasses.replace``), so the two can never
#: drift apart.
_RA226_UNSHIELDED: Final = SourceSpec(
    key="ra226-unshielded",
    name="Ra-226 in glass ball (diameter 5 mm) in stainless-steel tube",
    nuclide=(88, 226),
    geometry=Sphere(radius=2.5),
    material="G4_GLASS_PLATE",
    provenance="assumed (glass ball diameter 5 mm; legacy value)",
    container=Tube(
        material="G4_STAINLESS-STEEL",
        inner_radius=2.5,
        outer_radius=3.0,
        half_length=2.5,
        axis="z",
    ),
)

SOURCES: Final[dict[str, SourceSpec]] = {
    "k40": SourceSpec(
        key="k40",
        name="K-40 in anhydrous potassium carbonate (13x7.5x6 cm, 500 g)",
        nuclide=(19, 40),
        geometry=Box(size_x=130.0, size_y=75.0, size_z=60.0),
        material="K2CO3",
        mass_g=500.0,
        provenance="assumed (geometry and 500 g mass; legacy value)",
    ),
    "lu176": SourceSpec(
        key="lu176",
        name="Lu-176 in lutetium oxide powder (3x3x0.5 cm, 10 g)",
        nuclide=(71, 176),
        geometry=Box(size_x=30.0, size_y=30.0, size_z=5.0),
        material="Lu2O3",
        mass_g=10.0,
        provenance="assumed (geometry and 10 g mass; legacy value)",
    ),
    "am241": SourceSpec(
        key="am241",
        name="Am-241 in gold-foil sandwich (diameter 2 mm, 2 um Au / 1 um source / 2 um Au)",
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
        provenance="assumed (foil thicknesses 2/1/2 um; legacy value)",
    ),
    "th232": SourceSpec(
        key="th232",
        name=(
            "Th-232 in thorium nitrate pentahydrate ellipsoid (r 1x1x0.85 cm, "
            "10 g) in a bottomed Pyrex container on the beta shield"
        ),
        nuclide=(90, 232),
        geometry=Ellipsoid(semi_x=10.0, semi_y=10.0, semi_z=8.5),
        material="Th(NO3)4-5H2O",
        mass_g=10.0,
        container=Cup(
            material="G4_Pyrex_Glass",
            inner_radius=10.0,
            wall_thickness=1.0,
            height=50.0,
            bottom_thickness=1.0,
        ),
        shield=BETA_SHIELD,
        provenance="assumed (ellipsoid 1x1x0.85 cm, 10 g, cup and shield; legacy value)",
    ),
    "th232-unshielded": SourceSpec(
        key="th232-unshielded",
        name=(
            "Th-232 in thorium nitrate pentahydrate cylinder "
            "(r 0.87 cm x 1.5 cm, 10 g) in glass tube"
        ),
        nuclide=(90, 232),
        geometry=Cylinder(radius=8.7, half_length=7.5, axis="y"),
        material="Th(NO3)4-5H2O",
        mass_g=10.0,
        container=Tube(
            material="G4_Pyrex_Glass",
            inner_radius=10.0,
            outer_radius=11.0,
            half_length=25.0,
            axis="y",
        ),
        container_offset=(0.0, 0.0, -1.3),
        provenance="assumed (cylinder 0.87x1.5 cm, 10 g, tube offset -1.3 mm; legacy value)",
    ),
    "ra226": replace(
        _RA226_UNSHIELDED,
        key="ra226",
        name="Ra-226 in glass ball (diameter 5 mm) in stainless-steel tube on the beta shield",
        shield=BETA_SHIELD,
        provenance="assumed (glass ball 5 mm, steel tube, beta shield; legacy value)",
    ),
    "ra226-unshielded": _RA226_UNSHIELDED,
}

SOURCE_KEYS: Final[tuple[str, ...]] = tuple(SOURCES)


def get_source(key: str) -> SourceSpec:
    """Return the frozen source specification for ``key``."""
    try:
        return SOURCES[key]
    except KeyError as exc:
        raise ValidationError(
            f"unknown source key {key!r}; expected one of {SOURCE_KEYS}"
        ) from exc


# --------------------------------------------------------------------------
# Matrix modes (D-31/D-121)
# --------------------------------------------------------------------------
MODE_PLANE: Final = 1
MODE_SPHERE: Final = 2

#: ``mode -> (mode_name, geometry_name)`` for the matrix products (D-120-style
#: meta fields shared with the sim product).
MODE_METADATA: Final[dict[int, tuple[str, str]]] = {
    MODE_PLANE: ("plane_front_gamma", "plane"),
    MODE_SPHERE: ("sphere_circumscribed_gamma", "sphere"),
}


@dataclass(frozen=True)
class PrimaryAxis:
    """Primary-energy binning shared by both matrix modes.

    ``edges_kev`` are strictly increasing in keV (the fixed source-mode axis,
    D-121); ``active`` holds the column indices that receive events (upper edge
    > 0, since a gamma cannot carry a negative kinetic energy).
    """

    edges_kev: tuple[float, ...]
    active: tuple[int, ...]

    def __post_init__(self) -> None:
        edges = tuple(float(edge) for edge in self.edges_kev)
        if len(edges) < 2:
            raise ValidationError("primary axis needs at least two edges")
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValidationError("primary axis edges are not strictly increasing")
        active = tuple(
            column for column in range(len(edges) - 1) if edges[column + 1] > 0.0
        )
        if not active:
            raise ValidationError(
                "primary axis has no active column (no positive upper edge)"
            )
        object.__setattr__(self, "edges_kev", edges)
        object.__setattr__(self, "active", active)

    @property
    def n_columns(self) -> int:
        return len(self.edges_kev) - 1

    @property
    def n_active(self) -> int:
        return len(self.active)

    def energy_bounds_kev(self, column: int) -> tuple[float, float]:
        """Return the clamped ``(lo, hi)`` energy range of one active column."""
        if column not in self.active:
            raise ValidationError(f"primary column {column} is not active")
        lo = max(self.edges_kev[column], 0.0)
        hi = self.edges_kev[column + 1]
        if not lo < hi:
            raise ValidationError(
                f"primary column {column} has an empty clamped range [{lo}, {hi}] keV"
            )
        return lo, hi

    def sample_energy_kev(self, column: int, u: float) -> float:
        """Sample ``E_gamma`` uniformly inside an active column (F-SIM-1)."""
        if not 0.0 <= u < 1.0:
            raise ValidationError(f"uniform draw u must lie in [0, 1), got {u!r}")
        lo, hi = self.energy_bounds_kev(column)
        return lo + u * (hi - lo)


def make_primary_axis(edges_kev: NDArray[np.float64] | tuple[float, ...]) -> PrimaryAxis:
    """Build the primary axis from explicit edges (keV, strictly increasing)."""
    return PrimaryAxis(edges_kev=tuple(float(edge) for edge in edges_kev), active=())


def default_primary_axis() -> PrimaryAxis:
    """The fixed source-mode primary axis ``0..4096 keV / 4096 bins`` (D-121)."""
    return make_primary_axis(source_mode_deposition_edges_kev())


@dataclass(frozen=True)
class PlaneGammaSource:
    """Square plane gamma source on the housing front face (mode 1).

    Lengths in mm; ``z_mm`` is the housing front surface; ``angular`` names the
    direction distribution (``"lambertian"``: cosine-weighted toward -z).
    """

    size_x_mm: float
    size_y_mm: float
    z_mm: float
    angular: str = "lambertian"


@dataclass(frozen=True)
class SphereGammaSource:
    """Circumscribed-sphere gamma source wrapping the housing (mode 2).

    The sphere is centered on the detector and tangent to the eight housing
    corners (radius = ``sqrt(hx^2 + hy^2 + hz^2)`` in mm); ``angular`` is
    ``"inward-lambertian"`` (cosine-weighted about the local inward normal).
    """

    radius_mm: float
    angular: str = "inward-lambertian"


MatrixSource = PlaneGammaSource | SphereGammaSource


def _mode_and_geometry(source: MatrixSource) -> tuple[int, str, str, float]:
    if isinstance(source, PlaneGammaSource):
        mode = MODE_PLANE
        mode_name, geometry_name = MODE_METADATA[mode]
        return mode, mode_name, geometry_name, source.z_mm
    mode = MODE_SPHERE
    mode_name, geometry_name = MODE_METADATA[mode]
    return mode, mode_name, geometry_name, source.radius_mm


def mode_metadata(source: MatrixSource) -> tuple[int, str, str, float]:
    """``(mode, mode_name, geometry_name, geometry_param_mm)`` for a source."""
    return _mode_and_geometry(source)


# --------------------------------------------------------------------------
# Fixed-per-column event schedule (F-SIM-1/D-35)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnSchedule:
    """Fixed per-column event counts ``N_j`` with the legacy remainder rule.

    ``n_events`` is split evenly over the active columns; the first
    ``n_events % n_active`` active columns (in ascending index order) receive
    one extra event. This is exactly the legacy round-robin assignment
    ``active[(offset + event) % n_active]`` and keeps ``sum_j N_j = n_events``.
    """

    axis: PrimaryAxis
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != self.axis.n_columns:
            raise ValidationError(
                f"schedule counts length {len(self.counts)} != primary columns "
                f"{self.axis.n_columns}"
            )
        for column, count in enumerate(self.counts):
            if count < 0:
                raise ValidationError(f"column {column} has negative count {count}")
            if column not in self.axis.active and count != 0:
                raise ValidationError(f"inactive column {column} has {count} events")
        if sum(self.counts) < 0:  # pragma: no cover - defensive
            raise ValidationError("schedule total is negative")

    @classmethod
    def fixed_total(cls, axis: PrimaryAxis, n_events: int) -> ColumnSchedule:
        if n_events < 0:
            raise ValidationError(f"n_events must be non-negative, got {n_events!r}")
        active = axis.active
        base, remainder = divmod(n_events, len(active))
        counts = [0] * axis.n_columns
        for rank, column in enumerate(active):
            counts[column] = base + (1 if rank < remainder else 0)
        return cls(axis=axis, counts=tuple(counts))

    @property
    def total_events(self) -> int:
        return sum(self.counts)

    def column_counts(self) -> tuple[tuple[int, int], ...]:
        """Return ``(column, N_j)`` for the active columns in ascending order."""
        return tuple((column, self.counts[column]) for column in self.axis.active)

    def slices(self, n_workers: int) -> tuple[ColumnSlice, ...]:
        """Partition the active columns into at most ``n_workers`` slices.

        The partition is contiguous and balanced by column count; because every
        column has its own RNG stream (F-SIM-7) the partition does not affect
        the merged result.
        """
        if n_workers < 1:
            raise ValidationError(f"n_workers must be >= 1, got {n_workers!r}")
        pairs = self.column_counts()
        if not pairs:
            return ()
        workers = min(n_workers, len(pairs))
        base, remainder = divmod(len(pairs), workers)
        slices: list[ColumnSlice] = []
        start = 0
        for worker in range(workers):
            size = base + (1 if worker < remainder else 0)
            chunk = pairs[start : start + size]
            start += size
            if chunk:
                slices.append(ColumnSlice.from_pairs(chunk))
        return tuple(slices)


@dataclass(frozen=True)
class ColumnSlice:
    """One worker's contiguous run of active columns and their event counts."""

    columns: tuple[int, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.counts):
            raise ValidationError("ColumnSlice columns/counts length mismatch")
        if any(count < 0 for count in self.counts):
            raise ValidationError("ColumnSlice counts must be non-negative")

    @classmethod
    def from_pairs(cls, pairs: tuple[tuple[int, int], ...]) -> ColumnSlice:
        return cls(
            columns=tuple(column for column, _ in pairs),
            counts=tuple(count for _, count in pairs),
        )

    @property
    def total_events(self) -> int:
        return sum(self.counts)

    def cumulative_ends(self) -> tuple[int, ...]:
        """Exclusive cumulative event ends, one per assigned column."""
        ends = []
        running = 0
        for count in self.counts:
            running += count
            ends.append(running)
        return tuple(ends)

    def locate(self, local_event: int) -> tuple[int, int]:
        """Map a zero-based local event index to ``(column, within_column)``."""
        if not 0 <= local_event < self.total_events:
            raise ValidationError(
                f"local event {local_event} outside [0, {self.total_events})"
            )
        start = 0
        for column, count in zip(self.columns, self.counts, strict=True):
            if local_event < start + count:
                return column, local_event - start
            start += count
        raise ValidationError(  # pragma: no cover - guarded by the range check
            f"local event {local_event} not covered by the column slice"
        )


__all__ = [
    "BETA_SHIELD",
    "MODE_METADATA",
    "MODE_PLANE",
    "MODE_SPHERE",
    "SOURCES",
    "SOURCE_KEYS",
    "BetaShield",
    "Box",
    "ColumnSchedule",
    "ColumnSlice",
    "Cup",
    "Cylinder",
    "Disk",
    "Ellipsoid",
    "Layer",
    "MatrixSource",
    "PlaneGammaSource",
    "PrimaryAxis",
    "Sandwich",
    "ShieldPart",
    "SourceGeometry",
    "SourceSpec",
    "Sphere",
    "SphereGammaSource",
    "Tube",
    "default_primary_axis",
    "get_source",
    "make_primary_axis",
    "mode_metadata",
]
