"""Material definitions for the kc761 gamma-spectrometry simulation.

Custom materials are built from NIST elements; materials whose key starts with
``G4_`` are taken verbatim from the NIST material database.
"""

from __future__ import annotations

from geant4_pybind import (
    G4Material,
    G4NistManager,
    cm3,
    g,
)

#: Density of the ABS plastic housing (common ABS density), g/cm3.
ABS_DENSITY_G_CM3 = 1.05

#: CsI(Tl) scintillator density, g/cm3 (from the task specification).
CSI_TL_DENSITY_G_CM3 = 4.51


def _element(nist: G4NistManager, name: str):
    el = nist.FindOrBuildElement(name)
    if el is None:
        raise RuntimeError(f"failed to build NIST element {name!r}")
    return el


def build_csi_tl(nist: G4NistManager) -> G4Material:
    """CsI(Tl) with 0.1 mol% Tl doping.

    0.1 mol% Tl is taken relative to the host CsI (1 mol Tl per 1000 mol of
    CsI).  ``AddElementByNumberOfAtoms`` in this geant4_pybind build accepts
    only integer atom counts, so the trace dopant is added by mass fraction,
    which is the correct way to represent sub-mole-percent dopants.  The Tl
    mass fraction is ~8e-4 and has a negligible effect on photon transport.
    """
    # molar masses [g/mol]
    molar_mass = {"Cs": 132.90545, "I": 126.90447, "Tl": 204.3833}
    # amount of substance [mol] for 0.1 mol% Tl relative to CsI
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
    """ABS plastic (acrylonitrile-butadiene-styrene), approximate elemental
    composition by mass: C 0.865, H 0.082, N 0.053 (typical ABS); density
    1.05 g/cm3."""
    mat = G4Material("ABS", ABS_DENSITY_G_CM3 * g / cm3, 3)
    mat.AddElementByMassFraction(_element(nist, "C"), 0.865)
    mat.AddElementByMassFraction(_element(nist, "H"), 0.082)
    mat.AddElementByMassFraction(_element(nist, "N"), 0.053)
    return mat


def build_k2co3(nist: G4NistManager, density: float) -> G4Material:
    """Anhydrous potassium carbonate K2CO3 at the given density (g/cm3),
    derived in the code from the stated total mass and box volume."""
    mat = G4Material("K2CO3", density * g / cm3, 3)
    mat.AddElementByNumberOfAtoms(_element(nist, "K"), 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "C"), 1)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 3)
    return mat


def build_lu2o3(nist: G4NistManager, density: float) -> G4Material:
    """Lutetium oxide Lu2O3 at the given density (g/cm3)."""
    mat = G4Material("Lu2O3", density * g / cm3, 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "Lu"), 2)
    mat.AddElementByNumberOfAtoms(_element(nist, "O"), 3)
    return mat


def build_thorium_nitrate_pentahydrate(nist: G4NistManager, density: float) -> G4Material:
    """Thorium nitrate pentahydrate Th(NO3)4 * 5 H2O at the given density
    (g/cm3), derived in the code from the stated total mass and source
    volume."""
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
    """Build (and cache) the material ``key`` used by the source ``spec``.

    NIST materials (``G4_`` prefix) are looked up directly; custom materials
    are built at the source density (mass/volume when ``mass_g`` is set).
    """
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
    """Build every material referenced by the source configurations and the
    detector.  Returns a dict keyed by the ``SourceSpec.material`` keys (and
    the per-layer keys of sandwich sources) plus the detector materials."""
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

    # Container materials (Pyrex glass tube for Th-232, stainless-steel tube
    # for Ra-226); always build them so the detector code can rely on their key.
    mats["G4_Pyrex_Glass"] = nist.FindOrBuildMaterial("G4_Pyrex_Glass")
    mats["G4_STAINLESS-STEEL"] = nist.FindOrBuildMaterial("G4_STAINLESS-STEEL")
    return mats
