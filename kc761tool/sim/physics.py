"""Physics list and radioactive-decay/GPS configuration (D-32).

``build_physics_list`` returns the unchanged physics list:
``G4EmPenelopePhysics`` plus the decay, hadronic and ion extensions with the
0.1 mm default cut. Geant4 is imported inside the builders only.
"""

from __future__ import annotations

from kc761tool.errors import ValidationError
from kc761tool.sim.sources import (
    Box,
    Cylinder,
    Disk,
    Ellipsoid,
    Sandwich,
    SourceSpec,
    Sphere,
)

DEFAULT_CUT_MM = 0.1


def build_physics_list():  # noqa: ANN201 - returns a Geant4 physics list
    """Build the frozen physics list (D-32); Geant4 is imported here."""
    from geant4_pybind import (
        G4ChargeExchangePhysics,
        G4DecayPhysics,
        G4EmExtraPhysics,
        G4EmPenelopePhysics,
        G4HadronElasticPhysicsXS,
        G4HadronInelasticQBBC,
        G4IonElasticPhysics,
        G4IonPhysicsXS,
        G4NeutronTrackingCut,
        G4RadioactiveDecayPhysics,
        G4StoppingPhysics,
        G4VModularPhysicsList,
        mm,
    )

    class _PhysicsList(G4VModularPhysicsList):
        def __init__(self) -> None:
            super().__init__()
            self.SetDefaultCutValue(DEFAULT_CUT_MM * mm)
            for physics in (
                G4EmPenelopePhysics(),
                G4EmExtraPhysics(),
                G4DecayPhysics(),
                G4RadioactiveDecayPhysics(),
                G4HadronElasticPhysicsXS(),
                G4StoppingPhysics(),
                G4IonPhysicsXS(),
                G4IonElasticPhysics(),
                G4HadronInelasticQBBC(),
                G4ChargeExchangePhysics(),
                G4NeutronTrackingCut(),
            ):
                self.RegisterPhysics(physics)

    return _PhysicsList()


def configure_radioactive_decay(source: SourceSpec) -> None:
    """Apply the radioactive-decay limits for the source's decay chain."""
    from geant4_pybind import G4UImanager

    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(
        f"/process/had/rdm/thresholdForVeryLongDecayTime {source.threshold_years:.6g} year"
    )
    if source.nucleus_limits is not None:
        amin, amax, zmin, zmax = source.nucleus_limits
        ui.ApplyCommand(f"/process/had/rdm/nucleusLimits {amin} {amax} {zmin} {zmax}")


def gps_volume_commands(geometry) -> list[str]:  # noqa: ANN001
    """GPS commands sampling primary vertices inside the source geometry."""
    match geometry:
        case Box():
            return [
                "/gps/pos/shape Para",
                f"/gps/pos/halfx {0.5 * geometry.size_x} mm",
                f"/gps/pos/halfy {0.5 * geometry.size_y} mm",
                f"/gps/pos/halfz {0.5 * geometry.size_z} mm",
            ]
        case Cylinder():
            if geometry.axis not in ("y", "z"):
                raise ValidationError(f"unsupported cylinder axis: {geometry.axis!r}")
            commands = [
                "/gps/pos/shape Cylinder",
                f"/gps/pos/radius {geometry.radius} mm",
                f"/gps/pos/halfz {geometry.half_length} mm",
            ]
            if geometry.axis == "y":
                commands += ["/gps/pos/rot1 0 0 1", "/gps/pos/rot2 1 0 0"]
            return commands
        case Disk():
            return [
                "/gps/pos/shape Cylinder",
                f"/gps/pos/radius {geometry.radius} mm",
                f"/gps/pos/halfz {0.5 * geometry.thickness} mm",
            ]
        case Sandwich():
            return [
                "/gps/pos/shape Cylinder",
                f"/gps/pos/radius {geometry.radius} mm",
                f"/gps/pos/halfz {0.5 * geometry.active_thickness} mm",
            ]
        case Sphere():
            return ["/gps/pos/shape Sphere", f"/gps/pos/radius {geometry.radius} mm"]
        case Ellipsoid():
            return [
                "/gps/pos/shape Ellipsoid",
                f"/gps/pos/halfx {geometry.semi_x} mm",
                f"/gps/pos/halfy {geometry.semi_y} mm",
                f"/gps/pos/halfz {geometry.semi_z} mm",
            ]
        case _:
            raise ValidationError(f"unsupported source geometry: {geometry!r}")


def configure_gps(source: SourceSpec, detector) -> None:  # noqa: ANN001
    """Configure the general particle source to decay the source nuclide."""
    from geant4_pybind import G4UImanager, mm

    ui = G4UImanager.GetUIpointer()
    z, a = source.nuclide
    ui.ApplyCommand("/gps/particle ion")
    ui.ApplyCommand(f"/gps/ion {z} {a} 0 0")
    ui.ApplyCommand("/gps/energy 0 eV")
    ui.ApplyCommand("/gps/ang/type iso")
    ui.ApplyCommand("/gps/pos/type Volume")
    for command in gps_volume_commands(source.geometry):
        ui.ApplyCommand(command)
    center = detector.source_center
    # "centre" is the Geant4 UI command name and is kept verbatim (D-182).
    ui.ApplyCommand(
        f"/gps/pos/centre {center.x / mm} {center.y / mm} {center.z / mm} mm"
    )


__all__ = [
    "DEFAULT_CUT_MM",
    "build_physics_list",
    "configure_gps",
    "configure_radioactive_decay",
    "gps_volume_commands",
]
