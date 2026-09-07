"""Custom primary generators for the response-matrix simulation modes.

The matrix modes launch one gamma per event from a sampling surface (a
plane or a sphere), which the GPS ion-decay paradigm of the radioactive
modes cannot express, so they use a dedicated
:class:`G4VUserPrimaryGeneratorAction` built on :class:`G4ParticleGun`.
All randomness comes from the global CLHEP engine (``G4UniformRand``), so
the seed conventions of :mod:`kc761sim.runner` apply unchanged.

Event ``k`` (global numbering ``event_offset + event.GetEventID()``)
samples into primary column ``axis.active[k % n_active]`` with a uniform
draw inside the (possibly clamped) column range -- the assignment is
deterministic in the event number, only the within-column draw is random.
The primary gamma energy is left on a shared :class:`GammaEventState` for
the matrix event action to score the (E_gamma, deposition) pair into the
G histogram.
"""

from __future__ import annotations

import math

from geant4_pybind import (
    G4ParticleGun,
    G4ParticleTable,
    G4ThreeVector,
    G4UniformRand,
    G4VUserPrimaryGeneratorAction,
    keV,
    mm,
)
from .sources import MatrixSource, PlaneGammaSource, SphereGammaSource

_TWOPI = 2.0 * math.pi


class GammaEventState:
    """Per-event scratch shared by the generator and the matrix event action.

    Carries the primary gamma energy from ``GeneratePrimaries`` to the
    end-of-event scoring; the value is in Geant4 internal energy units
    (MeV).
    """

    def __init__(self) -> None:
        self.e_gamma = 0.0  # MeV


class _SurfaceGammaGenerator(G4VUserPrimaryGeneratorAction):
    """Common primary bookkeeping; subclasses sample the surface."""

    def __init__(self, source: MatrixSource, state: GammaEventState,
                 event_offset: int = 0):
        super().__init__()
        self.source = source
        self.axis = source.axis
        self.state = state
        self.event_offset = event_offset
        self.gun = G4ParticleGun()
        table = G4ParticleTable.GetParticleTable()
        self.gun.SetParticleDefinition(table.FindParticle("gamma"))

    def _primary_energy(self, event_id: int) -> float:
        """E_gamma (MeV) of the event: fixed column, uniform within it."""
        column = self.axis.active[
            (self.event_offset + event_id) % self.axis.n_active]
        return self.axis.primary_energy(column, G4UniformRand()) * keV

    def _sample_surface(self) -> tuple[G4ThreeVector, G4ThreeVector]:
        """Position (mm) and unit momentum direction; subclass hook."""
        raise NotImplementedError

    def GeneratePrimaries(self, event) -> None:
        e_gamma = self._primary_energy(event.GetEventID())
        self.state.e_gamma = e_gamma
        position, direction = self._sample_surface()
        self.gun.SetParticleEnergy(e_gamma)
        self.gun.SetParticlePosition(position)
        self.gun.SetParticleMomentumDirection(direction)
        self.gun.GeneratePrimaryVertex(event)


class PlaneGammaGenerator(_SurfaceGammaGenerator):
    """Plane source: uniform square position, Lambertian direction toward -z."""

    def _sample_surface(self):
        half_x = 0.5 * self.source.size_x * mm
        half_y = 0.5 * self.source.size_y * mm
        position = G4ThreeVector(
            (G4UniformRand() * 2.0 - 1.0) * half_x,
            (G4UniformRand() * 2.0 - 1.0) * half_y,
            self.source.z_mm * mm,
        )
        # Lambertian (cosine-weighted) about the -z normal:
        # cos(theta) = sqrt(u), phi uniform.
        cos_t = math.sqrt(G4UniformRand())
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = _TWOPI * G4UniformRand()
        direction = G4ThreeVector(
            -sin_t * math.cos(phi), -sin_t * math.sin(phi), -cos_t)
        return position, direction


class SphereGammaGenerator(_SurfaceGammaGenerator):
    """Sphere source: uniform sphere point, Lambertian about the inward normal."""

    def _sample_surface(self):
        radius = self.source.radius * mm
        # Uniform point on the sphere; n is the inward unit normal.
        cos_t0 = 2.0 * G4UniformRand() - 1.0
        sin_t0 = math.sqrt(max(0.0, 1.0 - cos_t0 * cos_t0))
        phi0 = _TWOPI * G4UniformRand()
        nx = -sin_t0 * math.cos(phi0)
        ny = -sin_t0 * math.sin(phi0)
        nz = -cos_t0
        normal = G4ThreeVector(nx, ny, nz)
        position = G4ThreeVector(-radius * nx, -radius * ny, -radius * nz)

        # Local tangent frame (e1, e2) orthogonal to the inward normal.
        reference = G4ThreeVector(0.0, 0.0, 1.0)
        e1 = normal.cross(reference)
        if e1.mag2() < 1e-12:  # normal ~ parallel to z: any tangent works
            e1 = G4ThreeVector(1.0, 0.0, 0.0)
        else:
            e1 = e1.unit()
        e2 = normal.cross(e1)

        # Lambertian about the inward normal in the local frame, rotated
        # back to the world frame: d = e1*sin_t*cos_phi + e2*sin_t*sin_phi
        # + n*cos_t.
        cos_t = math.sqrt(G4UniformRand())
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = _TWOPI * G4UniformRand()
        c1 = sin_t * math.cos(phi)
        c2 = sin_t * math.sin(phi)
        direction = G4ThreeVector(
            e1.x * c1 + e2.x * c2 + normal.x * cos_t,
            e1.y * c1 + e2.y * c2 + normal.y * cos_t,
            e1.z * c1 + e2.z * c2 + normal.z * cos_t,
        )
        return position, direction


def make_gamma_generator(source: MatrixSource, state: GammaEventState,
                         event_offset: int = 0):
    """Build the sampling generator for a matrix-mode source spec."""
    if isinstance(source, PlaneGammaSource):
        return PlaneGammaGenerator(source, state, event_offset)
    if isinstance(source, SphereGammaSource):
        return SphereGammaGenerator(source, state, event_offset)
    raise TypeError(f"not a matrix-mode source: {source!r}")
