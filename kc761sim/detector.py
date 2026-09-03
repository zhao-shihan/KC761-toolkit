"""Detector and source geometry for the KC761 simulation."""

from __future__ import annotations

import math

from geant4_pybind import (
    G4Box,
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
    Box,
    Cylinder,
    Disk,
    Sandwich,
    SourceSpec,
    Sphere,
    Tube,
)

WORLD_HALF_SIZE = 15.0 * cm

CRYSTAL_HALF_X = 5.0 * mm
CRYSTAL_HALF_Y = 5.0 * mm
CRYSTAL_HALF_Z = 12.7 * mm

HOUSING_WALL_THICKNESS = 1.0 * mm
HOUSING_HALF_X = CRYSTAL_HALF_X + HOUSING_WALL_THICKNESS
HOUSING_HALF_Y = CRYSTAL_HALF_Y + HOUSING_WALL_THICKNESS
HOUSING_HALF_Z = CRYSTAL_HALF_Z + HOUSING_WALL_THICKNESS

DETECTOR_FRONT_Z = HOUSING_HALF_Z

#: Air gap between the housing front face and the nearest source plane.
DETECTOR_GAP_MM = 1.0


def _rotate_to_y() -> G4RotationMatrix:
    """Rotation mapping a z-symmetric solid onto a y symmetry axis."""
    rot = G4RotationMatrix()
    rot.rotateX(-90.0 * deg)
    return rot


def _tube_center_z(tube: Tube, near_z: float) -> float:
    """Center z of a source tube so its nearest face sits at ``near_z``."""
    if tube.axis == "y":
        return near_z + tube.outer_radius
    if tube.axis == "z":
        return near_z + tube.half_length
    raise ValueError(f"unsupported tube axis: {tube.axis!r}")


def _bare_source_center_z(
    geometry: Box | Cylinder | Disk | Sandwich | Sphere, near_z: float
) -> float:
    """Center z of a bare (container-less) source given its near face."""
    match geometry:
        case Box():
            return near_z + geometry.size_z / 2.0
        case Disk():
            return near_z + geometry.thickness / 2.0
        case Sandwich():
            return near_z + geometry.total_thickness / 2.0
        case Sphere():
            return near_z + geometry.radius
        case Cylinder():
            if geometry.axis != "z":
                raise ValueError(
                    f"a {geometry.axis}-axis source cylinder outside a "
                    f"container is not supported (axis must be 'z' to face "
                    f"the detector)"
                )
            return near_z + geometry.half_length
        case _:
            raise ValueError(f"unknown geometry {geometry!r}")


class DetectorConstruction(G4VUserDetectorConstruction):
    def __init__(
        self,
        source: SourceSpec,
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
        """Place the container (if any) and the source material volume."""
        spec = self.source
        near_z = DETECTOR_FRONT_Z + DETECTOR_GAP_MM

        if spec.container is not None:
            container_position = G4ThreeVector(
                0.0, 0.0, _tube_center_z(spec.container, near_z) * mm
            )
            offset = spec.container_offset
            if offset is None:
                offset = (0.0, 0.0, 0.0)
            position = G4ThreeVector(
                container_position.x + offset[0] * mm,
                container_position.y + offset[1] * mm,
                container_position.z + offset[2] * mm,
            )
            self._construct_container(
                spec.container, container_position, world_lv
            )
        else:
            position = G4ThreeVector(
                0.0, 0.0, _bare_source_center_z(spec.geometry, near_z) * mm
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

    def _construct_container(
        self,
        tube: Tube,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ) -> G4LogicalVolume:
        material_name = self.source.container_material
        if material_name is None:  # guarded by SourceSpec.__post_init__
            raise ValueError(
                f"no container material defined for {self.source.key!r}"
            )
        material = self.materials[material_name]

        tube_solid = G4Tubs(
            "SourceTube",
            tube.inner_radius * mm,
            tube.outer_radius * mm,
            tube.half_length * mm,
            0.0,
            twopi,
        )
        tube_lv = G4LogicalVolume(tube_solid, material, "SourceTube")
        self._place_volume(
            tube_lv,
            "SourceTube",
            position,
            world_lv,
            rotate_to_y=tube.axis == "y",
        )
        return tube_lv
