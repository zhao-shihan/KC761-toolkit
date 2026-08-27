"""Material definitions for the KC761 simulation."""

from __future__ import annotations

from geant4_pybind import (
    G4Material,
    G4NistManager,
    cm3,
    g,
)

ABS_DENSITY_G_CM3 = 1.05

CSI_TL_DENSITY_G_CM3 = 4.51


def _element(nist: G4NistManager, name: str):
    el = nist.FindOrBuildElement(name)
    if el is None:
        raise RuntimeError(f"failed to build NIST element {name!r}")
    return el


def build_csi_tl(nist: G4NistManager) -> G4Material:
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


def build_thorium_nitrate_pentahydrate(nist: G4NistManager, density: float) -> G4Material:
    mat = G4Material("Th(NO3)4-5H2O", density * g / cm3, 4)
    mat.AddElementByNumberOfAtoms(_element(nist, "Th"), 1)
    mat.AddElementByNumberOfAtoms(_element(nist, "N"), 4)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 17)
    mat.AddElementByNumberOfAtoms(_element(nist, "H"), 10)
    return mat


def _ensure_material(
    mats: dict[str, G4Material],
    nist: G4NistManager,
    key: str,
    spec,
) -> None:
    if key in mats:
        return
    if key.startswith("G4_"):
        mats[key] = nist.FindOrBuildMaterial(key)
        return
    density = spec.effective_density
    if density is None:
        raise RuntimeError(f"no density available for material {key!r}")
    if key == "K2CO3":
        mats[key] = build_k2co3(nist, density)
    elif key == "Lu2O3":
        mats[key] = build_lu2o3(nist, density)
    elif key == "Th(NO3)4-5H2O":
        mats[key] = build_thorium_nitrate_pentahydrate(nist, density)
    else:
        raise RuntimeError(f"unknown source material {key!r}")


def build_all_materials() -> dict[str, G4Material]:
    nist = G4NistManager.Instance()
    mats: dict[str, G4Material] = {}

    mats["CsI_Tl"] = build_csi_tl(nist)
    mats["ABS"] = build_abs(nist)

    from . import config

    for key, spec in config.SOURCES.items():
        _ensure_material(mats, nist, spec.material, spec)
        if isinstance(spec.geometry, config.Sandwich):
            for layer in spec.geometry.layers:
                _ensure_material(mats, nist, layer.material, spec)

    mats["G4_AIR"] = nist.FindOrBuildMaterial("G4_AIR")

    mats["G4_Pyrex_Glass"] = nist.FindOrBuildMaterial("G4_Pyrex_Glass")
    mats["G4_STAINLESS-STEEL"] = nist.FindOrBuildMaterial("G4_STAINLESS-STEEL")
    return mats
