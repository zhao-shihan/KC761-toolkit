"""Physics list, radioactive-decay and particle-source configuration."""

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
    mm)


class PhysicsList(G4VModularPhysicsList):
    """The ``QBBC_EMZ`` reference physics list + radioactive decay.

    The Geant4 physics-list factory (which would build the true ``QBBC_EMZ``)
    cannot be used here: in geant4_pybind 0.1.3 a factory-returned list is
    handed to the run manager with a non-owning reference that
    ``SetUserInitialization`` cannot take ownership of, and the bound
    ``G4VModularPhysicsList`` has no ``ReplacePhysics``/``RemovePhysics``, so
    the default ``G4EmStandardPhysics`` of ``QBBC`` cannot be swapped for
    ``G4EmStandardPhysics_option4``.

    Following the geant4_pybind B3 example, the list is therefore built as a
    Python subclass of ``G4VModularPhysicsList`` that registers the same
    components as ``QBBC`` (see ``source/physics_lists/lists/src/QBBC.cc``)
    with the EMZ replacement: ``G4EmStandardPhysics_option4`` instead of the
    standard EM physics.

    ``QBBC`` does NOT include radioactive decay (it only registers the
    particle-decay physics), so ``G4RadioactiveDecayPhysics`` is registered
    explicitly here to enable the decay of the source nuclei.
    """

    def __init__(self):
        super().__init__()
        self.SetDefaultCutValue(0.7 * mm)  # same default cut as QBBC
        # EM physics (EMZ = option4)
        self.RegisterPhysics(G4EmStandardPhysics_option4())
        # Synchrotron radiation & gamma-nuclear physics
        self.RegisterPhysics(G4EmExtraPhysics())
        # Decays
        self.RegisterPhysics(G4DecayPhysics())
        # Hadron physics (QBBC components)
        self.RegisterPhysics(G4HadronElasticPhysicsXS())
        self.RegisterPhysics(G4StoppingPhysics())
        self.RegisterPhysics(G4IonPhysicsXS())
        self.RegisterPhysics(G4IonElasticPhysics())
        self.RegisterPhysics(G4HadronInelasticQBBC())
        self.RegisterPhysics(G4ChargeExchangePhysics())
        self.RegisterPhysics(G4NeutronTrackingCut())
        # Radioactive decay (QBBC does not provide it by default)
        self.RegisterPhysics(G4RadioactiveDecayPhysics())


def configure_radioactive_decay(source) -> None:
    """Enable the decay of the (long-lived) source nuclides.

    Geant4 >= 11.2 ignores at-rest decays whose sampled decay time exceeds a
    global threshold whose default is 1 year; all five sources would therefore
    be stable by default.  The threshold is raised per source (e.g. to 1e60
    years so that the whole Th-232 / Ra-226 chain can decay within one event).

    For Am-241 the decay is additionally restricted to the source nuclide
    itself via ``/process/had/rdm/nucleusLimits``, so that the long-lived
    daughter Np-237 (2.14 Myr) does not produce a spurious decay chain.

    The commands require the RDM messenger, which is created during run
    initialisation; this function must therefore be called after
    ``G4RunManager::Initialize``.
    """
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
    """Configure the general particle source (GPS) for the selected source.

    One ion of the source nuclide (at rest) is shot per event, at a position
    sampled uniformly inside the source volume.  Uniform sampling is achieved
    with the GPS volume distribution combined with ``/gps/pos/confine``, which
    rejects any point not located inside the physical volume ``Source``.

    The ``/gps`` messenger is created when the primary-generator action (and
    with it the G4GeneralParticleSource) is built during run initialisation,
    and the confined volume must exist, so this function must be called after
    ``G4RunManager::Initialize``.
    """
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
        # GPS has no "Box" shape; a parallelepiped with zero skew angles is an
        # axis-aligned box.
        ui.ApplyCommand("/gps/pos/shape Para")
        ui.ApplyCommand(f"/gps/pos/halfx {0.5 * geometry.size_x} mm")
        ui.ApplyCommand(f"/gps/pos/halfy {0.5 * geometry.size_y} mm")
        ui.ApplyCommand(f"/gps/pos/halfz {0.5 * geometry.size_z} mm")
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
    ui.ApplyCommand("/gps/pos/confine Source")
