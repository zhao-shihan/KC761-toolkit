"""Detector and source geometry for the KC761 simulation."""

from __future__ import annotations

import math

from geant4_pybind import (
    G4Box,
    G4Ellipsoid,
    G4LogicalVolume,
    G4Material,
    G4PVPlacement,
    G4RotationMatrix,
    G4Sphere,
    G4ThreeVector,
    G4Transform3D,
    G4Tubs,
    G4VUserDetectorConstruction,
    cm,
    deg,
    mm,
    twopi,
)
from .config import (
    BetaShield,
    Box,
    Cup,
    Cylinder,
    Disk,
    Ellipsoid,
    Sandwich,
    SourceSpec,
    Sphere,
    Tube,
)
from .sources import PlaneGammaSource, SphereGammaSource, make_primary_axis

WORLD_HALF_SIZE = 15.0 * cm

CRYSTAL_HALF_X = 5.0 * mm
CRYSTAL_HALF_Y = 5.0 * mm
CRYSTAL_HALF_Z = 12.7 * mm

HOUSING_WALL_THICKNESS = 1.0 * mm
HOUSING_HALF_X = CRYSTAL_HALF_X + HOUSING_WALL_THICKNESS
HOUSING_HALF_Y = CRYSTAL_HALF_Y + HOUSING_WALL_THICKNESS
HOUSING_HALF_Z = CRYSTAL_HALF_Z + HOUSING_WALL_THICKNESS

DETECTOR_FRONT_Z = HOUSING_HALF_Z

# Air gap between the housing front face and the nearest source plane.
DETECTOR_GAP_MM = 1.0


def build_plane_gamma_source(primary_edges) -> PlaneGammaSource:
    """Plane-mode source: crystal-face-sized square at the housing front.

    The plane spans the crystal end face exactly (2*CRYSTAL_HALF_X by
    2*CRYSTAL_HALF_Y, never hardcoded) and sits on the housing front
    surface (DETECTOR_FRONT_Z).  ``primary_edges`` are the input
    calibration file's deposition-energy edges (keV).
    """
    return PlaneGammaSource(
        size_x=2.0 * CRYSTAL_HALF_X / mm,
        size_y=2.0 * CRYSTAL_HALF_Y / mm,
        z_mm=DETECTOR_FRONT_Z / mm,
        axis=make_primary_axis(primary_edges),
    )


def build_sphere_gamma_source(primary_edges) -> SphereGammaSource:
    """Sphere-mode source: the housing's minimal circumscribed sphere.

    Centered on the detector center and tangent to the eight housing
    corners (radius derived from the housing half extents, never
    hardcoded).  ``primary_edges`` are the input calibration file's
    deposition-energy edges (keV).
    """
    radius = math.sqrt(
        HOUSING_HALF_X**2 + HOUSING_HALF_Y**2 + HOUSING_HALF_Z**2
    ) / mm
    return SphereGammaSource(
        radius=radius,
        axis=make_primary_axis(primary_edges),
    )


def _rotate_to_y() -> G4RotationMatrix:
    """Rotation mapping a z-symmetric solid onto a y symmetry axis."""
    rot = G4RotationMatrix()
    rot.rotateX(-90.0 * deg)
    return rot


def _geometry_half_z(geometry) -> float:
    """Half extent of the source geometry along the world z axis, in mm."""
    match geometry:
        case Box():
            return 0.5 * geometry.size_z
        case Disk():
            return 0.5 * geometry.thickness
        case Sandwich():
            return 0.5 * geometry.total_thickness
        case Sphere():
            return geometry.radius
        case Ellipsoid():
            return geometry.semi_z
        case Cylinder():
            if geometry.axis != "z":
                raise ValueError(
                    f"a {geometry.axis}-axis source cylinder is only "
                    f"supported inside a tube container (axis must be 'z' "
                    f"to face the detector)"
                )
            return geometry.half_length
        case _:
            raise ValueError(f"unknown geometry {geometry!r}")


def _bare_source_center_z(geometry, near_z: float) -> float:
    """Center z of a bare (container-less) source given its near face."""
    return near_z + _geometry_half_z(geometry)


def _container_position(
    container: Tube | Cup, anchor_z: float
) -> G4ThreeVector:
    """Container volume center for the assembly anchor plane at ``anchor_z``.

    The assembly is centered on the detector axis (x = y = 0). A Cup sits
    with its bottom face on the shield top plane (``anchor_z``); a tube
    faces the detector with its near face at ``anchor_z``. SourceSpec
    validation guarantees each container only appears with its matching
    mount.
    """
    if isinstance(container, Cup):
        return G4ThreeVector(
            0.0, 0.0, (anchor_z + 0.5 * container.height) * mm
        )
    if container.axis == "z":
        return G4ThreeVector(
            0.0, 0.0, (anchor_z + container.half_length) * mm
        )
    if container.axis == "y":
        return G4ThreeVector(
            0.0, 0.0, (anchor_z + container.outer_radius) * mm
        )
    raise ValueError(f"unsupported tube axis: {container.axis!r}")


def _source_position(
    spec: SourceSpec, container_position: G4ThreeVector
) -> G4ThreeVector:
    """Source center given its container's center, in world mm."""
    container = spec.container
    if isinstance(container, Cup):
        # The source rests on the cup's inner bottom surface.
        source_z = (
            container_position.z
            - 0.5 * container.height * mm
            + (container.bottom_thickness + _geometry_half_z(spec.geometry))
            * mm
        )
        return G4ThreeVector(
            container_position.x, container_position.y, source_z
        )
    offset = spec.container_offset
    if offset is None:
        return container_position
    return G4ThreeVector(
        container_position.x + offset[0] * mm,
        container_position.y + offset[1] * mm,
        container_position.z + offset[2] * mm,
    )


class DetectorConstruction(G4VUserDetectorConstruction):
    def __init__(
        self,
        source: SourceSpec | None,
        mats: dict[str, G4Material],
        check_overlaps: bool = True,
    ):
        super().__init__()
        self.source = source
        self.materials = mats
        self.check_overlaps = check_overlaps
        self.crystal_lv: G4LogicalVolume | None = None
        self.source_center: G4ThreeVector | None = None

    def Construct(self):
        air = self.materials["G4_AIR"]

        world_solid = G4Box("World", WORLD_HALF_SIZE,
                            WORLD_HALF_SIZE, WORLD_HALF_SIZE)
        world_lv = G4LogicalVolume(world_solid, air, "World")
        world_pv = G4PVPlacement(
            None,
            G4ThreeVector(),
            world_lv,
            "World",
            None,
            False,
            0,
            self.check_overlaps,
        )

        housing_solid = G4Box(
            "Housing", HOUSING_HALF_X, HOUSING_HALF_Y, HOUSING_HALF_Z
        )
        housing_lv = G4LogicalVolume(
            housing_solid, self.materials["ABS"], "Housing")
        G4PVPlacement(
            None,
            G4ThreeVector(),
            housing_lv,
            "Housing",
            world_lv,
            False,
            0,
            self.check_overlaps,
        )

        crystal_solid = G4Box(
            "Crystal", CRYSTAL_HALF_X, CRYSTAL_HALF_Y, CRYSTAL_HALF_Z
        )
        crystal_lv = G4LogicalVolume(
            crystal_solid, self.materials["CsI_Tl"], "Crystal")
        self.crystal_lv = crystal_lv
        G4PVPlacement(
            None,
            G4ThreeVector(),
            crystal_lv,
            "Crystal",
            housing_lv,
            False,
            0,
            self.check_overlaps,
        )

        # A None source builds the bare detector (world + housing +
        # crystal), used by the matrix simulation modes whose primaries
        # start on sampling surfaces, not inside source volumes.
        if self.source is not None:
            self._construct_source(world_lv)

        return world_pv

    def _place_volume(
        self,
        lv: G4LogicalVolume,
        name: str,
        position: G4ThreeVector,
        mother_lv: G4LogicalVolume,
        rotate_to_y: bool,
    ) -> None:
        """Place ``lv``, optionally rotating its z axis onto y."""
        if rotate_to_y:
            transform = G4Transform3D(_rotate_to_y(), position)
            G4PVPlacement(
                transform,
                lv,
                name,
                mother_lv,
                False,
                0,
                self.check_overlaps,
            )
        else:
            G4PVPlacement(
                None,
                position,
                lv,
                name,
                mother_lv,
                False,
                0,
                self.check_overlaps,
            )

    def _construct_source(self, world_lv: G4LogicalVolume) -> None:
        """Place the shield (if any), the container and the source volume."""
        spec = self.source
        shield = spec.shield

        if shield is not None:
            self._construct_shield(shield, world_lv)
            anchor_z = shield.top_z
        else:
            anchor_z = DETECTOR_FRONT_Z + DETECTOR_GAP_MM

        if spec.container is not None:
            container_position = _container_position(
                spec.container, anchor_z
            )
            position = _source_position(spec, container_position)
            self._construct_container(
                spec.container, container_position, world_lv
            )
        else:
            position = G4ThreeVector(
                0.0,
                0.0,
                _bare_source_center_z(spec.geometry, anchor_z) * mm,
            )
        self.source_center = position

        geometry = spec.geometry
        if isinstance(geometry, Sandwich):
            self.source_center = G4ThreeVector(
                position.x,
                position.y,
                position.z + geometry.active_center_offset * mm,
            )
            self._construct_sandwich(geometry, position, world_lv)
            return

        material = self.materials[spec.material]
        solid = self._build_source_solid(geometry)
        source_lv = G4LogicalVolume(solid, material, "Source")
        rotate_to_y = isinstance(geometry, Cylinder) and geometry.axis == "y"
        self._place_volume(
            source_lv, "Source", position, world_lv, rotate_to_y
        )

    def _build_source_solid(self, geometry):
        match geometry:
            case Box():
                return G4Box(
                    "SourceBox",
                    0.5 * geometry.size_x * mm,
                    0.5 * geometry.size_y * mm,
                    0.5 * geometry.size_z * mm,
                )
            case Disk():
                return G4Tubs(
                    "SourceDisk",
                    0.0,
                    geometry.radius * mm,
                    0.5 * geometry.thickness * mm,
                    0.0,
                    twopi,
                )
            case Sphere():
                return G4Sphere(
                    "SourceSphere",
                    0.0,
                    geometry.radius * mm,
                    0.0,
                    twopi,
                    0.0,
                    math.pi,
                )
            case Ellipsoid():
                return G4Ellipsoid(
                    "SourceEllipsoid",
                    geometry.semi_x * mm,
                    geometry.semi_y * mm,
                    geometry.semi_z * mm,
                )
            case Cylinder():
                return G4Tubs(
                    "SourceCylinder",
                    0.0,
                    geometry.radius * mm,
                    geometry.half_length * mm,
                    0.0,
                    twopi,
                )
            case _:
                raise ValueError(f"unsupported source geometry: {geometry!r}")

    def _construct_sandwich(
        self,
        sandwich: Sandwich,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ) -> None:
        n_layers = len(sandwich.layers)
        z = position.z - 0.5 * sandwich.total_thickness * mm
        for i, layer in enumerate(sandwich.layers):
            half_thickness = 0.5 * layer.thickness * mm
            layer_center = G4ThreeVector(
                position.x, position.y, z + half_thickness
            )
            if layer.active:
                name = "Source"
            elif i == 0:
                name = "SourceCladFront"
            elif i == n_layers - 1:
                name = "SourceCladBack"
            else:
                name = f"SourceClad{i}"
            solid = G4Tubs(
                name, 0.0, sandwich.radius * mm, half_thickness, 0.0, twopi
            )
            layer_lv = G4LogicalVolume(
                solid, self.materials[layer.material], name)
            G4PVPlacement(
                None,
                layer_center,
                layer_lv,
                name,
                world_lv,
                False,
                0,
                self.check_overlaps,
            )
            z += layer.thickness * mm

    def _construct_shield(
        self, shield: BetaShield, world_lv: G4LogicalVolume
    ) -> None:
        """Place the box parts of a composite shield (e.g. beta shield).

        The shield's lowest face must rest on the detector front surface:
        this pins the shield (and the source assembly mounted on its top
        plane) to the detector geometry, so the two sets of constants
        cannot drift apart.
        """
        lowest_face = min(
            part.center[2] - part.half_size[2] for part in shield.parts
        )
        if abs(lowest_face - DETECTOR_FRONT_Z / mm) > 1e-9:
            raise RuntimeError(
                f"shield {shield.material!r} lowest face at "
                f"z = {lowest_face:.6g} mm does not rest on the detector "
                f"front surface (z = {DETECTOR_FRONT_Z / mm} mm)"
            )
        material = self.materials[shield.material]
        for part in shield.parts:
            solid = G4Box(
                part.name,
                part.half_size[0] * mm,
                part.half_size[1] * mm,
                part.half_size[2] * mm,
            )
            part_lv = G4LogicalVolume(solid, material, part.name)
            G4PVPlacement(
                None,
                G4ThreeVector(
                    part.center[0] * mm,
                    part.center[1] * mm,
                    part.center[2] * mm,
                ),
                part_lv,
                part.name,
                world_lv,
                False,
                0,
                self.check_overlaps,
            )

    def _construct_container(
        self,
        container: Tube | Cup,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ) -> None:
        """Place the source container (tube or cup) at its center."""
        if isinstance(container, Cup):
            self._construct_cup(container, position, world_lv)
            return

        tube_solid = G4Tubs(
            "SourceTube",
            container.inner_radius * mm,
            container.outer_radius * mm,
            container.half_length * mm,
            0.0,
            twopi,
        )
        tube_lv = G4LogicalVolume(
            tube_solid, self.materials[container.material], "SourceTube"
        )
        self._place_volume(
            tube_lv,
            "SourceTube",
            position,
            world_lv,
            rotate_to_y=container.axis == "y",
        )

    def _construct_cup(
        self,
        cup: Cup,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ) -> None:
        """Build a bottomed vertical cylinder: wall tube plus bottom disk.

        The wall spans from the top of the bottom plate to the cup rim;
        both parts share the cup material and meet at the inner bottom
        plane without overlapping.
        """
        material = self.materials[cup.material]
        wall_half_length = 0.5 * (cup.height - cup.bottom_thickness) * mm
        wall_solid = G4Tubs(
            "SourceCupWall",
            cup.inner_radius * mm,
            cup.outer_radius * mm,
            wall_half_length,
            0.0,
            twopi,
        )
        wall_lv = G4LogicalVolume(wall_solid, material, "SourceCupWall")
        G4PVPlacement(
            None,
            G4ThreeVector(
                position.x,
                position.y,
                position.z + 0.5 * cup.bottom_thickness * mm,
            ),
            wall_lv,
            "SourceCupWall",
            world_lv,
            False,
            0,
            self.check_overlaps,
        )

        bottom_solid = G4Tubs(
            "SourceCupBottom",
            0.0,
            cup.outer_radius * mm,
            0.5 * cup.bottom_thickness * mm,
            0.0,
            twopi,
        )
        bottom_lv = G4LogicalVolume(
            bottom_solid, material, "SourceCupBottom")
        G4PVPlacement(
            None,
            G4ThreeVector(
                position.x,
                position.y,
                position.z - 0.5 * (cup.height - cup.bottom_thickness) * mm,
            ),
            bottom_lv,
            "SourceCupBottom",
            world_lv,
            False,
            0,
            self.check_overlaps,
        )
