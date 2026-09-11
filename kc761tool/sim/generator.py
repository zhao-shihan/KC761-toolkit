"""Primary generation for the matrix modes and the deterministic RNG seeds.

F-SIM-4 (plane/sphere Lambertian sampling) and F-SIM-7 (deterministic seed
derivation) live here. The pure sampling and seed functions carry no Geant4
import, so they are unit-testable without Geant4; the Geant4 action is built
lazily by :func:`make_gamma_generator`.
"""

from __future__ import annotations

import math
from typing import Final

from kc761tool.errors import ValidationError
from kc761tool.sim.sources import PlaneGammaSource, PrimaryAxis, SphereGammaSource

_TWOPI: Final = 2.0 * math.pi
_MASK64: Final = (1 << 64) - 1
_TWO_POW_63: Final = 1 << 63


class GammaEventState:
    """Per-event scratch shared by the matrix generator and event action.

    Carries the primary gamma energy (Geant4 internal energy units, MeV) from
    ``GeneratePrimaries`` to the end-of-event scoring.
    """

    def __init__(self) -> None:
        self.e_gamma = 0.0  # MeV


def splitmix64(value: int) -> int:
    """SplitMix64 finalizer (public-domain mixer); returns a 64-bit integer."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _stream_seed(base_seed: int, tag: int, index: int) -> int:
    """Map ``(base_seed, tag, index)`` to a positive 63-bit G4 seed (F-SIM-7)."""
    if base_seed < 0 or base_seed >= _TWO_POW_63:
        raise ValidationError(f"base seed must lie in [0, 2**63), got {base_seed!r}")
    if index < 0:
        raise ValidationError(f"stream index must be non-negative, got {index!r}")
    mixed = splitmix64((base_seed + tag) & _MASK64)
    mixed = splitmix64((mixed + index) & _MASK64)
    return mixed % _TWO_POW_63


def column_seed(base_seed: int, column: int) -> int:
    """Independent RNG seed of one primary column (F-SIM-7, D-123)."""
    return _stream_seed(base_seed, 0x01, column)


def block_seed(base_seed: int, block: int) -> int:
    """Independent RNG seed of one source-mode event block (F-SIM-7, D-123)."""
    return _stream_seed(base_seed, 0x02, block)


def lambertian_cos_sin(cos_u: float) -> tuple[float, float]:
    """Return ``(cos_t, sin_t)`` for a cosine-weighted direction (F-SIM-4).

    ``cos(theta) = sqrt(u)`` with ``u`` uniform in [0, 1) gives a Lambertian
    distribution about the local normal; ``sin_t`` is clamped for round-off.
    """
    if not 0.0 <= cos_u < 1.0:
        raise ValidationError(f"lambertian draw must lie in [0, 1), got {cos_u!r}")
    cos_t = math.sqrt(cos_u)
    return cos_t, math.sqrt(max(0.0, 1.0 - cos_t * cos_t))


def sample_plane_surface(
    source: PlaneGammaSource,
    u_x: float,
    u_y: float,
    u_cos: float,
    u_phi: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Uniform square position and Lambertian direction toward ``-z`` (F-SIM-4)."""
    half_x = 0.5 * source.size_x_mm
    half_y = 0.5 * source.size_y_mm
    position = (
        (2.0 * u_x - 1.0) * half_x,
        (2.0 * u_y - 1.0) * half_y,
        source.z_mm,
    )
    cos_t, sin_t = lambertian_cos_sin(u_cos)
    phi = _TWOPI * u_phi
    direction = (-sin_t * math.cos(phi), -sin_t * math.sin(phi), -cos_t)
    return position, direction


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 0.0:
        raise ValidationError("cannot normalize a zero-length vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def sample_sphere_surface(
    source: SphereGammaSource,
    u_point_cos: float,
    u_point_phi: float,
    u_cos: float,
    u_phi: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Uniform sphere point and inward-Lambertian direction (F-SIM-4).

    ``inward`` means the direction is cosine-weighted about the local inward
    normal ``n`` (pointing from the surface toward the origin).
    """
    cos_t0 = 2.0 * u_point_cos - 1.0
    sin_t0 = math.sqrt(max(0.0, 1.0 - cos_t0 * cos_t0))
    phi0 = _TWOPI * u_point_phi
    normal = (-sin_t0 * math.cos(phi0), -sin_t0 * math.sin(phi0), -cos_t0)
    position = tuple(-source.radius_mm * component for component in normal)

    reference = (0.0, 0.0, 1.0)
    e1 = _cross(normal, reference)
    e1 = (1.0, 0.0, 0.0) if _norm(e1) < 1e-12 else _unit(e1)
    e2 = _cross(normal, e1)

    cos_t, sin_t = lambertian_cos_sin(u_cos)
    phi = _TWOPI * u_phi
    c1 = sin_t * math.cos(phi)
    c2 = sin_t * math.sin(phi)
    direction = (
        e1[0] * c1 + e2[0] * c2 + normal[0] * cos_t,
        e1[1] * c1 + e2[1] * c2 + normal[1] * cos_t,
        e1[2] * c1 + e2[2] * c2 + normal[2] * cos_t,
    )
    return position, direction


def make_gamma_generator(
    column_slice,  # noqa: ANN001 - sources.ColumnSlice
    axis: PrimaryAxis,
    source: PlaneGammaSource | SphereGammaSource,
    base_seed: int,
    state,  # noqa: ANN001 - GammaEventState set by the caller
):  # noqa: ANN201
    """Build the Geant4 primary generator for one worker's column slice.

    The generator reseeds the global Geant4 engine with
    ``column_seed(base_seed, column)`` whenever it advances to a new column, so
    each column's events form a stream fixed by ``(base_seed, column)``
    (F-SIM-7). The per-event energy is drawn uniformly inside the column.
    """
    from geant4_pybind import (
        G4ParticleGun,
        G4ParticleTable,
        G4Random,
        G4ThreeVector,
        G4UniformRand,
        G4VUserPrimaryGeneratorAction,
        keV,
        mm,
    )

    class _SurfaceGammaGenerator(G4VUserPrimaryGeneratorAction):
        def __init__(self) -> None:
            super().__init__()
            self._slice = column_slice
            self._axis = axis
            self._source = source
            self._base_seed = base_seed
            self._state = state
            self._local_event = 0
            self._current_column = None
            self._set_seed = G4Random.setTheSeed
            # Resolve the per-event callables once (D-172): these are looked up
            # on every primary otherwise, and none of them changes the F-SIM-4
            # sampling or the F-SIM-7 seed chain.
            self._uniform = G4UniformRand
            self._locate = column_slice.locate
            self._energy_bounds = axis.energy_bounds_kev
            self._is_plane = isinstance(source, PlaneGammaSource)
            self._column_seed = column_seed
            self._gun = G4ParticleGun()
            table = G4ParticleTable.GetParticleTable()
            self._gun.SetParticleDefinition(table.FindParticle("gamma"))

        def GeneratePrimaries(self, event) -> None:  # noqa: ANN001, N802
            local = self._local_event
            self._local_event += 1
            column, _within = self._locate(local)
            if column != self._current_column:
                self._set_seed(self._column_seed(self._base_seed, column))
                self._current_column = column

            lo, hi = self._energy_bounds(column)
            uniform = self._uniform
            e_gamma_kev = lo + uniform() * (hi - lo)
            self._state.e_gamma = e_gamma_kev * keV

            if self._is_plane:
                position, direction = sample_plane_surface(
                    self._source,
                    uniform(),
                    uniform(),
                    uniform(),
                    uniform(),
                )
            else:
                position, direction = sample_sphere_surface(
                    self._source,
                    uniform(),
                    uniform(),
                    uniform(),
                    uniform(),
                )
            self._gun.SetParticleEnergy(e_gamma_kev * keV)
            self._gun.SetParticlePosition(
                G4ThreeVector(position[0] * mm, position[1] * mm, position[2] * mm)
            )
            self._gun.SetParticleMomentumDirection(
                G4ThreeVector(direction[0], direction[1], direction[2])
            )
            self._gun.GeneratePrimaryVertex(event)

    return _SurfaceGammaGenerator()


__all__ = [
    "block_seed",
    "column_seed",
    "lambertian_cos_sin",
    "make_gamma_generator",
    "sample_plane_surface",
    "sample_sphere_surface",
    "splitmix64",
]
