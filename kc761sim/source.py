"""Primary source generation for the KC761 simulation.

Simple geometries are sampled with the general particle source (GPS).
The coated-sphere geometry (a thin spherical-cap shell hugging a support
ball) gets a dedicated generator: the GPS Volume/Sphere sampler only
knows the outer radius, so it cannot draw efficiently from the cap
volume. Instead a particle gun emits the source ion at rest, uniformly
distributed inside the coating and isotropic in direction.
"""

from __future__ import annotations

import math

from geant4_pybind import (
    G4GeneralParticleSource,
    G4ParticleGun,
    G4ParticleTable,
    G4RandomDirection,
    G4ThreeVector,
    G4UniformRand,
    G4VUserPrimaryGeneratorAction,
    mm,
    twopi,
)
from .config import CoatedSphere, SourceSpec


def _coated_sphere_sampling_parameters(
    geometry: CoatedSphere,
) -> tuple[float, float, float, float]:
    """Precompute the per-event sampling constants of a coating volume.

    Returns ``(r**3_inner, r**3_outer, cos_theta_min, cos_theta_max)``
    so the event loop does not re-derive them.
    """
    return (
        (geometry.radius * mm) ** 3,
        (geometry.outer_radius * mm) ** 3,
        math.cos(math.radians(geometry.theta_min)),
        math.cos(math.radians(geometry.theta_max)),
    )


def _sample_coated_sphere_position(
    shell: tuple[float, float, float, float], center: G4ThreeVector
) -> G4ThreeVector:
    """Sample a point uniformly inside the coating cap volume.

    The coating spans ``[radius, outer_radius]`` in radius and
    ``[theta_min, theta_max]`` in polar angle (measured from the +z
    axis), so ``cos(theta)`` is uniform and the radius follows a
    ``r**3`` distribution.
    """
    r3_inner, r3_outer, cos_min, cos_max = shell
    radius = (r3_inner + G4UniformRand() * (r3_outer - r3_inner)) ** (
        1.0 / 3.0
    )
    cos_theta = cos_max + G4UniformRand() * (cos_min - cos_max)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = twopi * G4UniformRand()
    return center + G4ThreeVector(
        radius * sin_theta * math.cos(phi),
        radius * sin_theta * math.sin(phi),
        radius * cos_theta,
    )


class PrimarySource(G4VUserPrimaryGeneratorAction):
    """Primary vertex generator for one radioactive source.

    Non-coated geometries use the GPS configured by
    :func:`physics.configure_gps`; the coated sphere uses a dedicated
    particle gun with per-event position and direction sampling.
    """

    def __init__(self, source: SourceSpec, detector):
        super().__init__()
        self.source = source
        self.detector = detector
        self.gps = G4GeneralParticleSource()
        self.gun: G4ParticleGun | None = None
        self._shell: tuple[float, float, float, float] | None = None
        self._gun_ion_ready = False
        if isinstance(source.geometry, CoatedSphere):
            self.gun = G4ParticleGun(1)
            self.gun.SetParticleEnergy(0.0)
            self.gun.SetParticleTime(0.0)
            self._shell = _coated_sphere_sampling_parameters(source.geometry)

    def _prepare_gun(self) -> None:
        """Attach the source ion to the gun.

        Deferred to the first event: the action is built when it is
        registered with the run manager, before the physics is
        constructed, and the ion table rejects lookups until then.
        """
        if self._gun_ion_ready:
            return
        z, a = self.source.nuclide
        ion_table = G4ParticleTable.GetParticleTable().GetIonTable()
        ion = ion_table.GetIon(z, a, 0.0)
        if ion is None:
            raise RuntimeError(
                f"failed to build ion {z}-{a} for source {self.source.key!r}"
            )
        self.gun.SetParticleDefinition(ion)
        self._gun_ion_ready = True

    def GeneratePrimaries(self, event) -> None:
        if self.gun is not None:
            self._prepare_gun()
            self.gun.SetParticlePosition(
                _sample_coated_sphere_position(
                    self._shell, self.detector.source_center
                )
            )
            self.gun.SetParticleMomentumDirection(G4RandomDirection())
            self.gun.GeneratePrimaryVertex(event)
        else:
            self.gps.GeneratePrimaryVertex(event)
