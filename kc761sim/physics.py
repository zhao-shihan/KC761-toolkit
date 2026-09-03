"""Physics list and radioactive-decay/GPS configuration for KC761 simulation."""

from __future__ import annotations

from geant4_pybind import (
    G4ChargeExchangePhysics,
    G4DecayPhysics,
    G4EmExtraPhysics,
    G4EmLivermorePhysics,
    # G4EmPenelopePhysics,
    G4HadronElasticPhysicsXS,
    G4HadronInelasticQBBC,
    G4IonElasticPhysics,
    G4IonPhysicsXS,
    G4NeutronTrackingCut,
    G4RadioactiveDecayPhysics,
    G4StoppingPhysics,
    G4UImanager,
    G4VModularPhysicsList,
    mm,
)
from .config import Box, Cylinder, Disk, Sandwich, SourceSpec, Sphere


class PhysicsList(G4VModularPhysicsList):
    def __init__(self):
        super().__init__()
        self.SetDefaultCutValue(0.1 * mm)
        self.RegisterPhysics(G4EmLivermorePhysics())
        # self.RegisterPhysics(G4EmPenelopePhysics())
        self.RegisterPhysics(G4EmExtraPhysics())
        self.RegisterPhysics(G4DecayPhysics())
        self.RegisterPhysics(G4RadioactiveDecayPhysics())
        self.RegisterPhysics(G4HadronElasticPhysicsXS())
        self.RegisterPhysics(G4StoppingPhysics())
        self.RegisterPhysics(G4IonPhysicsXS())
        self.RegisterPhysics(G4IonElasticPhysics())
        self.RegisterPhysics(G4HadronInelasticQBBC())
        self.RegisterPhysics(G4ChargeExchangePhysics())
        self.RegisterPhysics(G4NeutronTrackingCut())


def configure_radioactive_decay(source: SourceSpec) -> None:
    """Apply radioactive-decay limits for the source's decay chain."""
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(
        f"/process/had/rdm/thresholdForVeryLongDecayTime {source.threshold_years:.6g} year"
    )
    if source.nucleus_limits is not None:
        amin, amax, zmin, zmax = source.nucleus_limits
        ui.ApplyCommand(
            f"/process/had/rdm/nucleusLimits {amin} {amax} {zmin} {zmax}"
        )


def _gps_volume_commands(geometry) -> list[str]:
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
                raise ValueError(
                    f"unsupported cylinder axis: {geometry.axis!r}"
                )
            commands = [
                "/gps/pos/shape Cylinder",
                f"/gps/pos/radius {geometry.radius} mm",
                f"/gps/pos/halfz {geometry.half_length} mm",
            ]
            if geometry.axis == "y":
                commands += [
                    "/gps/pos/rot1 0 0 1",
                    "/gps/pos/rot2 1 0 0",
                ]
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
            return [
                "/gps/pos/shape Sphere",
                f"/gps/pos/radius {geometry.radius} mm",
            ]
        case _:
            raise ValueError(f"unsupported source geometry: {geometry!r}")


def configure_gps(source: SourceSpec, detector) -> None:
    """Configure the general particle source to decay the source nuclide."""
    ui = G4UImanager.GetUIpointer()
    z, a = source.nuclide
    ui.ApplyCommand("/gps/particle ion")
    ui.ApplyCommand(f"/gps/ion {z} {a} 0 0")
    ui.ApplyCommand("/gps/energy 0 eV")
    ui.ApplyCommand("/gps/ang/type iso")

    ui.ApplyCommand("/gps/pos/type Volume")
    for command in _gps_volume_commands(source.geometry):
        ui.ApplyCommand(command)

    center = detector.source_center
    ui.ApplyCommand(
        f"/gps/pos/centre {center.x / mm} {center.y / mm} {center.z / mm} mm"
    )
