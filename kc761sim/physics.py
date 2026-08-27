"""Physics list and radioactive-decay/GPS configuration for KC761 simulation."""

from __future__ import annotations

from geant4_pybind import (
    G4ChargeExchangePhysics,
    G4DecayPhysics,
    G4EmExtraPhysics,
    G4EmStandardPhysics_option4,
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


class PhysicsList(G4VModularPhysicsList):
    def __init__(self):
        super().__init__()
        self.SetDefaultCutValue(0.7 * mm)
        self.RegisterPhysics(G4EmStandardPhysics_option4())
        self.RegisterPhysics(G4EmExtraPhysics())
        self.RegisterPhysics(G4DecayPhysics())
        self.RegisterPhysics(G4HadronElasticPhysicsXS())
        self.RegisterPhysics(G4StoppingPhysics())
        self.RegisterPhysics(G4IonPhysicsXS())
        self.RegisterPhysics(G4IonElasticPhysics())
        self.RegisterPhysics(G4HadronInelasticQBBC())
        self.RegisterPhysics(G4ChargeExchangePhysics())
        self.RegisterPhysics(G4NeutronTrackingCut())
        self.RegisterPhysics(G4RadioactiveDecayPhysics())


def configure_radioactive_decay(source) -> None:
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(
        f"/process/had/rdm/thresholdForVeryLongDecayTime {source.threshold_years:.6g} year"
    )
    if source.nucleus_limits is not None:
        amin, amax, zmin, zmax = source.nucleus_limits
        ui.ApplyCommand(
            f"/process/had/rdm/nucleusLimits {amin} {amax} {zmin} {zmax}"
        )


def configure_gps(source, detector) -> None:
    ui = G4UImanager.GetUIpointer()
    z, a = source.nuclide
    ui.ApplyCommand("/gps/particle ion")
    ui.ApplyCommand(f"/gps/ion {z} {a} 0 0")
    ui.ApplyCommand("/gps/energy 0 eV")
    ui.ApplyCommand("/gps/ang/type iso")

    geometry = source.geometry
    center = detector.source_center
    ui.ApplyCommand("/gps/pos/type Volume")
    if geometry.kind == "box":
        ui.ApplyCommand("/gps/pos/shape Para")
        ui.ApplyCommand(f"/gps/pos/halfx {0.5 * geometry.size_x} mm")
        ui.ApplyCommand(f"/gps/pos/halfy {0.5 * geometry.size_y} mm")
        ui.ApplyCommand(f"/gps/pos/halfz {0.5 * geometry.size_z} mm")
    elif geometry.kind == "cylinder":
        ui.ApplyCommand("/gps/pos/shape Cylinder")
        ui.ApplyCommand(f"/gps/pos/radius {geometry.radius} mm")
        ui.ApplyCommand(f"/gps/pos/halfz {geometry.half_length} mm")
        if geometry.axis == "y":
            ui.ApplyCommand("/gps/pos/rot1 0 0 1")
            ui.ApplyCommand("/gps/pos/rot2 1 0 0")
    elif geometry.kind == "sandwich":
        ui.ApplyCommand("/gps/pos/shape Cylinder")
        ui.ApplyCommand(f"/gps/pos/radius {geometry.radius} mm")
        ui.ApplyCommand(
            f"/gps/pos/halfz {0.5 * geometry.active_thickness} mm"
        )
    elif geometry.kind == "disk":
        ui.ApplyCommand("/gps/pos/shape Cylinder")
        ui.ApplyCommand(f"/gps/pos/radius {geometry.radius} mm")
        ui.ApplyCommand(f"/gps/pos/halfz {0.5 * geometry.thickness} mm")
    elif geometry.kind == "sphere":
        ui.ApplyCommand("/gps/pos/shape Sphere")
        ui.ApplyCommand(f"/gps/pos/radius {geometry.radius} mm")
    else:
        raise ValueError(f"unsupported source geometry: {geometry!r}")
    ui.ApplyCommand(
        f"/gps/pos/centre {center.x / mm} {center.y / mm} {center.z / mm} mm"
    )
