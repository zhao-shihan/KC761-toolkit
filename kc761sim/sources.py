"""Pure-data source specifications for the response-matrix simulation modes.

The matrix modes launch one primary gamma per event from a sampling
surface (a square plane in front of the scintillator, or a sphere
wrapping the housing); they share no structure with the radioactive-decay
:class:`kc761sim.config.SourceSpec` paradigm, so they live in their own
dataclasses.  All geometric quantities are derived from the single source
of truth in :mod:`kc761sim.detector` by the builders there; this module
stays free of Geant4 imports.

The primary axis reuses the input calibration file's deposition-energy
edges (identical binning, by design).  A column can only receive events
while its upper edge is positive -- a gamma cannot carry a negative
kinetic energy -- so fully negative columns are skipped (zero counts) and
the single straddling column samples the clamped range
``[max(lower_edge, 0), upper_edge]``.  The skipped columns produce all-zero
columns in R and an efficiency of 0 (distinguishable from a truly detected
zero efficiency by their zero per-column event count).
"""

from __future__ import annotations

from dataclasses import dataclass

# Mode identifiers stored in the G output files.
MODE_PLANE = 1
MODE_SPHERE = 2

# Per-mode metadata written to the G output files:
# ``(mode, mode_name, geometry parameter name)``.
_MODE_METADATA = {
    MODE_PLANE: ("plane_front_gamma", "plane_z_mm"),
    MODE_SPHERE: ("sphere_circumscribed_gamma", "sphere_radius_mm"),
}


@dataclass(frozen=True)
class PrimaryAxis:
    """Primary-energy binning shared by both matrix modes.

    ``edges`` are the input calibration file's deposition-energy edges
    (strictly increasing, keV); ``active`` holds the column indices with a
    positive upper edge -- the columns that actually receive events.
    """

    edges: tuple[float, ...]
    active: tuple[int, ...]

    @property
    def n_active(self) -> int:
        return len(self.active)

    def primary_energy(self, column: int, u: float) -> float:
        """Sample E_gamma (keV) for one event of an *active* column.

        ``column`` indexes the primary axis and must be active (its upper
        edge > 0); ``u`` is a uniform draw in [0, 1).  The straddling zero
        column samples the clamped range ``[max(lo, 0), hi]``; other
        columns ``[lo, hi]``.
        """
        lo = max(self.edges[column], 0.0)
        hi = self.edges[column + 1]
        if not (0.0 <= u < 1.0 and lo < hi):
            raise ValueError(
                f"primary_energy: column {column} is not active "
                f"(edges [{self.edges[column]:g}, {hi:g}] keV) or u={u:g} "
                f"is outside [0, 1)")
        return lo + u * (hi - lo)


def make_primary_axis(edges) -> PrimaryAxis:
    """Build the primary axis from calibration-file deposition edges (keV)."""
    edges_tuple = tuple(float(e) for e in edges)
    if len(edges_tuple) < 2:
        raise ValueError("primary axis needs at least two edges")
    if any(b <= a for a, b in zip(edges_tuple, edges_tuple[1:])):
        raise ValueError("primary axis edges are not strictly increasing")
    active = tuple(
        j for j in range(len(edges_tuple) - 1) if edges_tuple[j + 1] > 0.0
    )
    if not active:
        raise ValueError(
            "primary axis has no active column (no positive upper edge); "
            "the calibration file's energy range is unusable")
    return PrimaryAxis(edges=edges_tuple, active=active)


@dataclass(frozen=True)
class PlaneGammaSource:
    """Square plane gamma source in front of the scintillator (mode 1).

    Lengths in mm; ``z_mm`` is the housing front surface (the source plane
    coincides with the housing front face); ``angular`` names the
    direction distribution (``"lambertian"``: cosine-weighted toward -z).
    """

    size_x: float  # mm, = 2 * CRYSTAL_HALF_X
    size_y: float  # mm, = 2 * CRYSTAL_HALF_Y
    z_mm: float  # mm, = DETECTOR_FRONT_Z
    axis: PrimaryAxis
    angular: str = "lambertian"


@dataclass(frozen=True)
class SphereGammaSource:
    """Circumscribed sphere gamma source wrapping the housing (mode 2).

    The sphere is centered on the detector center and tangent to the eight
    housing corners (radius = sqrt(HX^2 + HY^2 + HZ^2) in mm); ``angular``
    is ``"inward-lambertian"`` (cosine-weighted about the local inward
    normal).
    """

    radius: float
    axis: PrimaryAxis
    angular: str = "inward-lambertian"


MatrixSource = PlaneGammaSource | SphereGammaSource


def mode_metadata(source: MatrixSource) -> tuple[int, str, str, float]:
    """``(mode, mode_name, geometry parameter name, geometry parameter)``.

    The single dispatch point for everything mode-specific that the
    simulation-file records; the geometry parameter is ``z_mm`` for the
    plane and ``radius`` for the sphere.
    """
    if isinstance(source, PlaneGammaSource):
        mode_name, geometry_name = _MODE_METADATA[MODE_PLANE]
        return MODE_PLANE, mode_name, geometry_name, source.z_mm
    if isinstance(source, SphereGammaSource):
        mode_name, geometry_name = _MODE_METADATA[MODE_SPHERE]
        return MODE_SPHERE, mode_name, geometry_name, source.radius
    raise TypeError(f"not a matrix-mode source: {source!r}")
