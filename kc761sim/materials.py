"""Material definitions for the KC761 simulation."""

from __future__ import annotations

from geant4_pybind import G4Element, G4Material, G4NistManager, cm3, g

from .config import Sandwich, SourceSpec

ABS_DENSITY_G_CM3 = 1.05

CSI_TL_DENSITY_G_CM3 = 4.51


def _element(nist: G4NistManager, name: str) -> G4Element:
    el = nist.FindOrBuildElement(name)
    if el is None:
        raise RuntimeError(f"failed to build NIST element {name!r}")
    return el


def build_csi_tl(nist: G4NistManager) -> G4Material:
    """CsI(Tl) scintillator crystal (Tl at 1000 ppm molar)."""
    molar_mass = {"Cs": 132.90545, "I": 126.90447, "Tl": 204.3833}
    mol = {"Cs": 999.5, "I": 999.5, "Tl": 1.0}
    total_mass = sum(molar_mass[k] * mol[k] for k in mol)
    mass_fraction = {k: molar_mass[k] * mol[k] / total_mass for k in mol}

    mat = G4Material(
        "CsI_Tl",
        CSI_TL_DENSITY_G_CM3 * g / cm3,
        3,
    )
    mat.AddElementByMassFraction(_element(nist, "Cs"), mass_fraction["Cs"])
    mat.AddElementByMassFraction(_element(nist, "I"), mass_fraction["I"])
    mat.AddElementByMassFraction(_element(nist, "Tl"), mass_fraction["Tl"])
    return mat


def build_abs(nist: G4NistManager) -> G4Material:
    """ABS plastic used for the detector housing."""
    mat = G4Material("ABS", ABS_DENSITY_G_CM3 * g / cm3, 3)
    mat.AddElementByMassFraction(_element(nist, "C"), 0.865)
    mat.AddElementByMassFraction(_element(nist, "H"), 0.082)
    mat.AddElementByMassFraction(_element(nist, "N"), 0.053)
    return mat


def build_k2co3(nist: G4NistManager, density: float) -> G4Material:
    mat = G4Material("K2CO3", density * g / cm3, 3)
    mat.AddElementByNumberOfAtoms(_element(nist, "K"), 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "C"), 1)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 3)
    return mat


def build_lu2o3(nist: G4NistManager, density: float) -> G4Material:
    mat = G4Material("Lu2O3", density * g / cm3, 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "Lu"), 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 3)
    return mat


def build_thorium_nitrate_pentahydrate(
    nist: G4NistManager, density: float
) -> G4Material:
    mat = G4Material("Th(NO3)4-5H2O", density * g / cm3, 4)
    mat.AddElementByNumberOfAtoms(_element(nist, "Th"), 1)
    mat.AddElementByNumberOfAtoms(_element(nist, "N"), 4)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 17)
    mat.AddElementByNumberOfAtoms(_element(nist, "H"), 10)
    return mat


# Builders for custom source materials; each receives (nist, density).
_CUSTOM_MATERIAL_BUILDERS = {
    "K2CO3": build_k2co3,
    "Lu2O3": build_lu2o3,
    "Th(NO3)4-5H2O": build_thorium_nitrate_pentahydrate,
}


def _require_material(
    mats: dict[str, G4Material],
    nist: G4NistManager,
    densities: dict[str, float],
    name: str,
    density: float | None,
) -> None:
    """Make sure material ``name`` exists in ``mats``.

    ``density`` (g/cm^3) must be supplied when a custom (non-NIST) material
    is requested; requesting the same custom material with a conflicting
    density is an error.
    """
    if name.startswith("G4_"):
        if name not in mats:
            mats[name] = nist.FindOrBuildMaterial(name)
        return

    if name not in _CUSTOM_MATERIAL_BUILDERS:
        raise RuntimeError(f"unknown custom source material {name!r}")
    if density is None:
        raise RuntimeError(
            f"custom material {name!r} requires a density; only the primary "
            f"source material can derive one from 'mass_g' or 'density'"
        )

    # Check the density for consistency up front, otherwise a conflicting
    # request is silently swallowed by the ``name in mats`` early return.
    previous = densities.get(name)
    if previous is not None and abs(previous - density) > 1e-9:
        raise RuntimeError(
            f"material {name!r} requested with conflicting densities "
            f"{previous:.6g} and {density:.6g} g/cm^3"
        )
    if name in mats:
        return
    densities[name] = density
    mats[name] = _CUSTOM_MATERIAL_BUILDERS[name](nist, density)


def build_all_materials(*specs: SourceSpec) -> dict[str, G4Material]:
    """Build every material needed by the detector and the given sources."""
    nist = G4NistManager.Instance()
    mats: dict[str, G4Material] = {
        "CsI_Tl": build_csi_tl(nist),
        "ABS": build_abs(nist),
    }
    densities: dict[str, float] = {}

    # Detector-facing world volume.
    mats["G4_AIR"] = nist.FindOrBuildMaterial("G4_AIR")

    for spec in specs:
        _require_material(mats, nist, densities, spec.material,
                          spec.effective_density)
        geometry = spec.geometry
        if isinstance(geometry, Sandwich):
            for layer in geometry.layers:
                _require_material(mats, nist, densities, layer.material, None)
        if spec.container_material is not None:
            _require_material(mats, nist, densities, spec.container_material,
                              None)

    return mats
