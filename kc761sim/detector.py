"""Detector and source geometry for the kc761 gamma-spectrometry simulation.

Coordinate system
-----------------
* The world is a 30 cm x 30 cm x 30 cm air cube centred on the origin.
* The CsI(Tl) crystal (10 x 10 x 25.4 mm) is wrapped in a 0.5 mm ABS housing
  (11 x 11 x 26.4 mm).  The assembly is centred on the origin with the crystal
  long axis along **z**; the 10 x 10 mm end faces are the detection faces.
* The housing front face facing the sources is at z = +13.2 mm.  Every source
  is placed on the +z side with a 1 mm face-to-face gap along z.
"""

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
    Disk,
    SourceSpec,
    Sphere,
    Tube,
    source_center_z,
)

#: World half-size (cm).
WORLD_HALF_SIZE = 15.0 * cm

#: CsI(Tl) crystal half dimensions (mm).
CRYSTAL_HALF_X = 5.0 * mm
CRYSTAL_HALF_Y = 5.0 * mm
CRYSTAL_HALF_Z = 12.7 * mm

#: ABS housing half dimensions (mm): crystal + 0.5 mm wall on every side.
HOUSING_HALF_X = CRYSTAL_HALF_X + 0.5 * mm
HOUSING_HALF_Y = CRYSTAL_HALF_Y + 0.5 * mm
HOUSING_HALF_Z = CRYSTAL_HALF_Z + 0.5 * mm

#: z of the housing front face (towards the sources), mm.
DETECTOR_FRONT_Z = HOUSING_HALF_Z


class DetectorConstruction(G4VUserDetectorConstruction):
    """Builds the world, the detector and the source.

    Visual attributes are intentionally NOT set here: rendering (colors,
    transparency, visibility, projection) is fully controlled by the
    visualization macros (script/vis.mac), which the application executes in
    interactive mode.  ``check_overlaps`` enables the Geant4 geometry overlap
    check (used as a debug aid; it is turned off for quiet batch runs).
    """

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
        #: logical volume of the CsI(Tl) crystal, used for energy scoring
        self.crystal_lv: G4LogicalVolume | None = None
        #: world-frame centre of the source volume, used for vertex sampling
        self.source_center: G4ThreeVector | None = None

    def Construct(self):
        air = self.materials["G4_AIR"]

        # ---- World ---------------------------------------------------------
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

        # ---- Detector ------------------------------------------------------
        # The housing is placed in the world and the crystal inside the housing.
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

        # ---- Source ---------------------------------------------------------
        self._construct_source(world_lv)

        return world_pv

    def _construct_source(self, world_lv: G4LogicalVolume):
        spec = self.source
        material = self.materials[spec.material]
        center_z = source_center_z(spec, DETECTOR_FRONT_Z)
        position = G4ThreeVector(0.0, 0.0, center_z * mm)
        self.source_center = position

        # The container tube (when present) and the active source volume are
        # both placed in the world as siblings: the source sits in the tube's
        # hollow, which is not part of the tube solid, so nesting it as a
        # daughter of the tube would put it outside its mother's solid.
        if spec.container is not None:
            self._construct_container(
                spec.container, position=position, world_lv=world_lv
            )

        geometry = spec.geometry
        if isinstance(geometry, Box):
            solid = G4Box(
                "SourceBox",
                0.5 * geometry.size_x * mm,
                0.5 * geometry.size_y * mm,
                0.5 * geometry.size_z * mm,
            )
        elif isinstance(geometry, Disk):
            solid = G4Tubs(
                "SourceDisk",
                0.0,
                geometry.radius * mm,
                0.5 * geometry.thickness * mm,
                0.0,
                twopi,
            )
        elif isinstance(geometry, Sphere):
            solid = G4Sphere(
                "SourceSphere",
                0.0,
                geometry.radius * mm,
                0.0,
                twopi,
                0.0,
                math.pi,
            )
        else:
            raise ValueError(f"unsupported source geometry: {geometry!r}")

        source_lv = G4LogicalVolume(solid, material, "Source")
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

    def _construct_container(
        self,
        tube: Tube,
        position: G4ThreeVector,
        world_lv: G4LogicalVolume,
    ) -> G4LogicalVolume:
        """Build and place the container tube (sibling of the source volume)."""
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
            # rotate the (z-axis) tube so its axis points along y
            rot = G4RotationMatrix()
            rot.rotateX(90.0 * deg)
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
