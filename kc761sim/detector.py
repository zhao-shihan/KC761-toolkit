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
    source_center_z,
)

WORLD_HALF_SIZE = 15.0 * cm

CRYSTAL_HALF_X = 5.0 * mm
CRYSTAL_HALF_Y = 5.0 * mm
CRYSTAL_HALF_Z = 12.7 * mm

HOUSING_HALF_X = CRYSTAL_HALF_X + 1.0 * mm
HOUSING_HALF_Y = CRYSTAL_HALF_Y + 1.0 * mm
HOUSING_HALF_Z = CRYSTAL_HALF_Z + 1.0 * mm

DETECTOR_FRONT_Z = HOUSING_HALF_Z


class DetectorConstruction(G4VUserDetectorConstruction):
    def __init__(
        self,
        source: SourceSpec,
        materials: dict[str, G4Material],
        check_overlaps: bool = True,
    ):
        super().__init__()
        self.source = source
        self.materials = materials
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

    def _construct_source(self, world_lv: G4LogicalVolume):
        spec = self.source
        center_z = source_center_z(spec, DETECTOR_FRONT_Z)
        position = G4ThreeVector(0.0, 0.0, center_z * mm)
        self.source_center = position

        if spec.container is not None:
            container_position = position
            offset = spec.container_offset
            if offset is not None:
                container_position = G4ThreeVector(
                    position.x - offset[0] * mm,
                    position.y - offset[1] * mm,
                    position.z - offset[2] * mm,
                )
            self._construct_container(
                spec.container,
                position=container_position,
                world_lv=world_lv,
            )

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

        if isinstance(geometry, Cylinder) and geometry.axis == "y":
            rot = G4RotationMatrix()
            rot.rotateX(-90.0 * deg)
            transform = G4Transform3D(rot, position)
            G4PVPlacement(
                transform,
                source_lv,
                "Source",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )
        else:
            G4PVPlacement(
                None,
                position,
                source_lv,
                "Source",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )

    def _build_source_solid(self, geometry):
        name = f"Source{geometry.kind.capitalize()}"
        if isinstance(geometry, Box):
            return G4Box(
                name,
                0.5 * geometry.size_x * mm,
                0.5 * geometry.size_y * mm,
                0.5 * geometry.size_z * mm,
            )
        if isinstance(geometry, Disk):
            return G4Tubs(
                name,
                0.0,
                geometry.radius * mm,
                0.5 * geometry.thickness * mm,
                0.0,
                twopi,
            )
        if isinstance(geometry, Sphere):
            return G4Sphere(
                name,
                0.0,
                geometry.radius * mm,
                0.0,
                twopi,
                0.0,
                math.pi,
            )
        if isinstance(geometry, Cylinder):
            return G4Tubs(
                name,
                0.0,
                geometry.radius * mm,
                geometry.half_length * mm,
                0.0,
                twopi,
            )
        raise ValueError(f"unsupported source geometry: {geometry!r}")

    def _construct_sandwich(
        self,
        sandwich: Sandwich,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ):
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
        spec = self.source
        if spec.key == "th232":
            material = self.materials["G4_Pyrex_Glass"]
        elif spec.key == "ra226":
            material = self.materials["G4_STAINLESS-STEEL"]
        else:
            raise ValueError(f"no container material defined for {spec.key!r}")

        tube_solid = G4Tubs(
            "SourceTube",
            tube.inner_radius * mm,
            tube.outer_radius * mm,
            tube.half_length * mm,
            0.0,
            twopi,
        )
        tube_lv = G4LogicalVolume(tube_solid, material, "SourceTube")

        if tube.axis == "y":
            rot = G4RotationMatrix()
            rot.rotateX(-90.0 * deg)
            transform = G4Transform3D(rot, position)
            G4PVPlacement(
                transform,
                tube_lv,
                "SourceTube",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )
        else:
            G4PVPlacement(
                None,
                position,
                tube_lv,
                "SourceTube",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )
        return tube_lv
