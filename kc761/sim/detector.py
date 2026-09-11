"""Detector and source geometry construction (Geant4, imported lazily).

The construction follows the geometry constants (D-34); every
length is derived from :mod:`kc761.sim.geometry` and the source dataclasses, so
no dimension appears twice. The source builders are pure (no Geant4) and live
here next to the construction they feed.
"""

from __future__ import annotations

import math

from kc761.errors import ValidationError
from kc761.sim.geometry import DEFAULT_GEOMETRY, DetectorGeometry
from kc761.sim.sources import (
    BetaShield,
    Box,
    Cup,
    Cylinder,
    Disk,
    Ellipsoid,
    PlaneGammaSource,
    Sandwich,
    SourceSpec,
    Sphere,
    SphereGammaSource,
)


def build_plane_gamma_source(
    geometry: DetectorGeometry = DEFAULT_GEOMETRY,
) -> PlaneGammaSource:
    """Plane-mode source: crystal-face-sized square on the housing front.

    The plane spans the crystal end face exactly (``2*CRYSTAL_HALF_X`` by
    ``2*CRYSTAL_HALF_Y``) and sits on the housing front surface.
    """
    return PlaneGammaSource(
        size_x_mm=2.0 * geometry.crystal_half_x_mm,
        size_y_mm=2.0 * geometry.crystal_half_y_mm,
        z_mm=geometry.detector_front_z_mm,
    )


def build_sphere_gamma_source(
    geometry: DetectorGeometry = DEFAULT_GEOMETRY,
) -> SphereGammaSource:
    """Sphere-mode source: the housing's minimal circumscribed sphere."""
    radius = math.sqrt(
        geometry.housing_half_x_mm**2
        + geometry.housing_half_y_mm**2
        + geometry.housing_half_z_mm**2
    )
    return SphereGammaSource(radius_mm=radius)


def _geometry_half_z_mm(geometry) -> float:  # noqa: ANN001
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
                raise ValidationError(
                    f"a {geometry.axis}-axis source cylinder is only supported "
                    "inside a tube container (axis must be 'z' to face the detector)"
                )
            return geometry.half_length
        case _:
            raise ValidationError(f"unknown source geometry {geometry!r}")


def build_detector(source, materials, *, geometry: DetectorGeometry = DEFAULT_GEOMETRY,
                   check_overlaps: bool = True):  # noqa: ANN001, ANN201
    """Return a Geant4 detector construction; Geant4 is imported here.

    ``source`` may be ``None`` for the bare detector used by the matrix modes
    (their primaries start on sampling surfaces, not inside source volumes).
    """
    from geant4_pybind import (
        G4Box,
        G4Ellipsoid,
        G4LogicalVolume,
        G4PVPlacement,
        G4RotationMatrix,
        G4Sphere,
        G4ThreeVector,
        G4Transform3D,
        G4Tubs,
        G4VUserDetectorConstruction,
        deg,
        mm,
        twopi,
    )

    world_half = geometry.world_half_mm * mm
    _box = G4Box
    _ellipsoid = G4Ellipsoid
    _logical = G4LogicalVolume
    _placement = G4PVPlacement
    _rotation = G4RotationMatrix
    _sphere = G4Sphere
    _three = G4ThreeVector
    _transform = G4Transform3D
    _tubs = G4Tubs
    _base = G4VUserDetectorConstruction
    _mm = mm
    _twopi = twopi

    def _rotate_to_y():  # noqa: ANN202
        rotation = _rotation()
        rotation.rotateX(-90.0 * deg)
        return rotation

    def _container_position(container, anchor_z: float):  # noqa: ANN001, ANN202
        """Container volume center for an assembly anchor plane at ``anchor_z``.

        The assembly is centered on the detector axis. A cup sits with its
        bottom face on the shield top plane; a tube faces the detector with its
        near face at ``anchor_z``.
        """
        if isinstance(container, Cup):
            return _three(0.0, 0.0, (anchor_z + 0.5 * container.height) * _mm)
        if container.axis == "z":
            return _three(0.0, 0.0, (anchor_z + container.half_length) * _mm)
        if container.axis == "y":
            return _three(0.0, 0.0, (anchor_z + container.outer_radius) * _mm)
        raise ValidationError(f"unsupported tube axis: {container.axis!r}")

    def _source_position(spec: SourceSpec, container_position):  # noqa: ANN001, ANN202
        container = spec.container
        if isinstance(container, Cup):
            source_z = (
                container_position.z
                - 0.5 * container.height * _mm
                + (container.bottom_thickness + _geometry_half_z_mm(spec.geometry)) * _mm
            )
            return _three(container_position.x, container_position.y, source_z)
        offset = spec.container_offset
        if offset is None:
            return container_position
        return _three(
            container_position.x + offset[0] * _mm,
            container_position.y + offset[1] * _mm,
            container_position.z + offset[2] * _mm,
        )

    class _Detector(_base):
        def __init__(self) -> None:
            super().__init__()
            self.source = source
            self.materials = materials
            self.check_overlaps = check_overlaps
            self.crystal_lv = None
            self.source_center = None

        def Construct(self):  # noqa: N802 - Geant4 API
            air = self.materials["G4_AIR"]
            world_solid = _box("World", world_half, world_half, world_half)
            world_lv = _logical(world_solid, air, "World")
            world_pv = _placement(
                None, _three(), world_lv, "World", None, False, 0, self.check_overlaps
            )

            housing_solid = _box(
                "Housing",
                geometry.housing_half_x_mm * _mm,
                geometry.housing_half_y_mm * _mm,
                geometry.housing_half_z_mm * _mm,
            )
            housing_lv = _logical(housing_solid, self.materials["ABS"], "Housing")
            _placement(
                None, _three(), housing_lv, "Housing", world_lv, False, 0, self.check_overlaps
            )

            crystal_solid = _box(
                "Crystal",
                geometry.crystal_half_x_mm * _mm,
                geometry.crystal_half_y_mm * _mm,
                geometry.crystal_half_z_mm * _mm,
            )
            crystal_lv = _logical(crystal_solid, self.materials["CsI_Tl"], "Crystal")
            self.crystal_lv = crystal_lv
            _placement(
                None, _three(), crystal_lv, "Crystal", housing_lv, False, 0, self.check_overlaps
            )

            if self.source is not None:
                self._construct_source(world_lv)
            return world_pv

        def _place_volume(self, lv, name, position, mother_lv, rotate_to_y):  # noqa: ANN001
            if rotate_to_y:
                _placement(
                    _transform(_rotate_to_y(), position),
                    lv,
                    name,
                    mother_lv,
                    False,
                    0,
                    self.check_overlaps,
                )
            else:
                _placement(
                    None, position, lv, name, mother_lv, False, 0, self.check_overlaps
                )

        def _construct_source(self, world_lv):  # noqa: ANN001
            spec = self.source
            shield = spec.shield
            if shield is not None:
                self._construct_shield(shield, world_lv)
                anchor_z = shield.top_z
            else:
                anchor_z = geometry.detector_front_z_mm + geometry.detector_gap_mm

            if spec.container is not None:
                container_position = _container_position(spec.container, anchor_z)
                position = _source_position(spec, container_position)
                self._construct_container(spec.container, container_position, world_lv)
            else:
                position = _three(0.0, 0.0, _bare_center_z(spec.geometry, anchor_z) * _mm)
            self.source_center = position

            if isinstance(spec.geometry, Sandwich):
                self.source_center = _three(
                    position.x,
                    position.y,
                    position.z + spec.geometry.active_center_offset * _mm,
                )
                self._construct_sandwich(spec.geometry, position, world_lv)
                return

            material = self.materials[spec.material]
            solid = self._build_source_solid(spec.geometry)
            source_lv = _logical(solid, material, "Source")
            rotate_to_y = isinstance(spec.geometry, Cylinder) and spec.geometry.axis == "y"
            self._place_volume(source_lv, "Source", position, world_lv, rotate_to_y)

        @staticmethod
        def _build_source_solid(source_geometry):  # noqa: ANN001, ANN205
            match source_geometry:
                case Box():
                    return _box(
                        "SourceBox",
                        0.5 * source_geometry.size_x * _mm,
                        0.5 * source_geometry.size_y * _mm,
                        0.5 * source_geometry.size_z * _mm,
                    )
                case Disk():
                    return _tubs(
                        "SourceDisk",
                        0.0,
                        source_geometry.radius * _mm,
                        0.5 * source_geometry.thickness * _mm,
                        0.0,
                        _twopi,
                    )
                case Sphere():
                    return _sphere(
                        "SourceSphere",
                        0.0,
                        source_geometry.radius * _mm,
                        0.0,
                        _twopi,
                        0.0,
                        math.pi,
                    )
                case Ellipsoid():
                    return _ellipsoid(
                        "SourceEllipsoid",
                        source_geometry.semi_x * _mm,
                        source_geometry.semi_y * _mm,
                        source_geometry.semi_z * _mm,
                    )
                case Cylinder():
                    return _tubs(
                        "SourceCylinder",
                        0.0,
                        source_geometry.radius * _mm,
                        source_geometry.half_length * _mm,
                        0.0,
                        _twopi,
                    )
                case _:
                    raise ValidationError(f"unsupported source geometry: {source_geometry!r}")

        def _construct_sandwich(self, sandwich, position, world_lv):  # noqa: ANN001
            n_layers = len(sandwich.layers)
            z = position.z - 0.5 * sandwich.total_thickness * _mm
            for index, layer in enumerate(sandwich.layers):
                half_thickness = 0.5 * layer.thickness * _mm
                layer_center = _three(position.x, position.y, z + half_thickness)
                if layer.active:
                    name = "Source"
                elif index == 0:
                    name = "SourceCladFront"
                elif index == n_layers - 1:
                    name = "SourceCladBack"
                else:
                    name = f"SourceClad{index}"
                solid = _tubs(name, 0.0, sandwich.radius * _mm, half_thickness, 0.0, _twopi)
                layer_lv = _logical(solid, self.materials[layer.material], name)
                _placement(
                    None,
                    layer_center,
                    layer_lv,
                    name,
                    world_lv,
                    False,
                    0,
                    self.check_overlaps,
                )
                z += layer.thickness * _mm

        def _construct_shield(self, shield: BetaShield, world_lv):  # noqa: ANN001
            lowest = shield.lowest_z
            if abs(lowest - geometry.detector_front_z_mm) > 1e-9:
                raise ValidationError(
                    f"shield {shield.material!r} lowest face at z = {lowest:.6g} mm "
                    f"does not rest on the detector front surface "
                    f"(z = {geometry.detector_front_z_mm} mm)"
                )
            material = self.materials[shield.material]
            for part in shield.parts:
                solid = _box(
                    part.name,
                    part.half_size[0] * _mm,
                    part.half_size[1] * _mm,
                    part.half_size[2] * _mm,
                )
                part_lv = _logical(solid, material, part.name)
                _placement(
                    None,
                    _three(
                        part.center[0] * _mm,
                        part.center[1] * _mm,
                        part.center[2] * _mm,
                    ),
                    part_lv,
                    part.name,
                    world_lv,
                    False,
                    0,
                    self.check_overlaps,
                )

        def _construct_container(self, container, position, world_lv):  # noqa: ANN001
            if isinstance(container, Cup):
                self._construct_cup(container, position, world_lv)
                return
            tube_solid = _tubs(
                "SourceTube",
                container.inner_radius * _mm,
                container.outer_radius * _mm,
                container.half_length * _mm,
                0.0,
                _twopi,
            )
            tube_lv = _logical(tube_solid, self.materials[container.material], "SourceTube")
            self._place_volume(
                tube_lv, "SourceTube", position, world_lv, container.axis == "y"
            )

        def _construct_cup(self, cup: Cup, position, world_lv):  # noqa: ANN001
            material = self.materials[cup.material]
            wall_half_length = 0.5 * (cup.height - cup.bottom_thickness) * _mm
            wall_solid = _tubs(
                "SourceCupWall",
                cup.inner_radius * _mm,
                cup.outer_radius * _mm,
                wall_half_length,
                0.0,
                _twopi,
            )
            wall_lv = _logical(wall_solid, material, "SourceCupWall")
            _placement(
                None,
                _three(
                    position.x,
                    position.y,
                    position.z + 0.5 * cup.bottom_thickness * _mm,
                ),
                wall_lv,
                "SourceCupWall",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )

            bottom_solid = _tubs(
                "SourceCupBottom",
                0.0,
                cup.outer_radius * _mm,
                0.5 * cup.bottom_thickness * _mm,
                0.0,
                _twopi,
            )
            bottom_lv = _logical(bottom_solid, material, "SourceCupBottom")
            _placement(
                None,
                _three(
                    position.x,
                    position.y,
                    position.z - 0.5 * (cup.height - cup.bottom_thickness) * _mm,
                ),
                bottom_lv,
                "SourceCupBottom",
                world_lv,
                False,
                0,
                self.check_overlaps,
            )

    return _Detector()


def _bare_center_z(geometry, near_z: float) -> float:  # noqa: ANN001
    """Center z of a bare (container-less) source given its near face."""
    return near_z + _geometry_half_z_mm(geometry)


__all__ = [
    "build_detector",
    "build_plane_gamma_source",
    "build_sphere_gamma_source",
]
