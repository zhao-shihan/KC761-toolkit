"""Material compositions for the KC761 simulation (D-32/D-34).

The compositions and densities are the unchanged values (D-34);
the module separates the pure data tables (importable without Geant4) from the
lazy Geant4 builders so tests can check the data without loading Geant4.

Provenance:
* ``CsI_Tl``: the Tl mole fraction is ``1 / 1999`` of the Cs+I pairs (the
  "1000 ppm molar" label is a rounded description; the numeric composition is
  fixed by D-34).
* ``ABS``: representative acrylonitrile-butadiene-styrene mass fractions
  (assumed).
* ``R4600``: the exact formulation is unpublished; a representative C/H/O mix
  is used (assumed).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from kc761.errors import ValidationError
from kc761.sim.sources import Sandwich, SourceSpec

# --- pure composition data -------------------------------------------------
CSI_TL_DENSITY_G_CM3: Final = 4.51
CSI_TL_ATOMIC_MASS_G_MOL: Final[dict[str, float]] = {
    "Cs": 132.90545,
    "I": 126.90447,
    "Tl": 204.3833,
}
CSI_TL_MOLAR_AMOUNTS: Final[dict[str, float]] = {"Cs": 999.5, "I": 999.5, "Tl": 1.0}

ABS_DENSITY_G_CM3: Final = 1.05
ABS_MASS_FRACTIONS: Final[tuple[tuple[str, float], ...]] = (
    ("C", 0.865),
    ("H", 0.082),
    ("N", 0.053),
)

R4600_DENSITY_G_CM3: Final = 1.166
R4600_MASS_FRACTIONS: Final[tuple[tuple[str, float], ...]] = (
    ("C", 0.70166),
    ("H", 0.07038),
    ("O", 0.22796),
)

#: Custom source materials given as ``(element, atom count)`` pairs; the
#: density comes from the source's ``mass_g``/``density`` (F-SIM data).
CUSTOM_MATERIAL_ATOMS: Final[dict[str, tuple[tuple[str, int], ...]]] = {
    "K2CO3": (("K", 2), ("C", 1), ("O", 3)),
    "Lu2O3": (("Lu", 2), ("O", 3)),
    "Th(NO3)4-5H2O": (("Th", 1), ("N", 4), ("O", 17), ("H", 10)),
}

#: Custom materials built unconditionally from mass fractions by
#: :func:`build_all_materials`; source geometries (containers, shields,
#: sandwiches) may reference them without supplying a density.
PREBUILT_MATERIALS: Final[frozenset[str]] = frozenset({"CsI_Tl", "ABS", "R4600"})

MATERIAL_PROVENANCE: Final[dict[str, str]] = {
    "CsI_Tl": "assumed (Tl mole ratio 1/1999; see module docstring)",
    "ABS": "assumed (representative ABS mass fractions)",
    "R4600": "assumed (unpublished formulation; representative C/H/O mix)",
    "K2CO3": "measured stoichiometry, density from source mass",
    "Lu2O3": "measured stoichiometry, density from source mass",
    "Th(NO3)4-5H2O": "measured stoichiometry, density from source mass",
}


def csi_tl_mass_fractions() -> dict[str, float]:
    """Return the CsI(Tl) element mass fractions from the frozen molar amounts."""
    total = sum(
        CSI_TL_ATOMIC_MASS_G_MOL[element] * amount
        for element, amount in CSI_TL_MOLAR_AMOUNTS.items()
    )
    return {
        element: CSI_TL_ATOMIC_MASS_G_MOL[element] * amount / total
        for element, amount in CSI_TL_MOLAR_AMOUNTS.items()
    }


def _check_material_lookup(name: str) -> None:
    if name.startswith("G4_"):
        return
    if name in PREBUILT_MATERIALS:
        return
    if name not in CUSTOM_MATERIAL_ATOMS:
        raise ValidationError(f"unknown custom source material {name!r}")


def _require_custom_density(name: str, density: float | None) -> float:
    if density is None:
        raise ValidationError(
            f"custom material {name!r} requires a density; only the primary "
            "source material can derive one from 'mass_g' or 'density'"
        )
    if not np.isfinite(density) or density <= 0.0:
        raise ValidationError(f"material {name!r} density must be positive, got {density!r}")
    return float(density)


def build_all_materials(*specs: SourceSpec):  # noqa: ANN201
    """Build every material needed by the detector and the given sources.

    Geant4 is imported here, inside the function, so importing this module in a
    no-Geant4 environment stays possible (architecture rule 5).
    """
    from geant4_pybind import G4Material, G4NistManager, cm3, g

    nist = G4NistManager.Instance()

    def element(name: str):  # noqa: ANN202
        found = nist.FindOrBuildElement(name)
        if found is None:
            raise ValidationError(f"failed to build NIST element {name!r}")
        return found

    csi = G4Material("CsI_Tl", CSI_TL_DENSITY_G_CM3 * g / cm3, 3)
    for element_name, fraction in csi_tl_mass_fractions().items():
        csi.AddElementByMassFraction(element(element_name), fraction)

    abs_plastic = G4Material("ABS", ABS_DENSITY_G_CM3 * g / cm3, 3)
    for element_name, fraction in ABS_MASS_FRACTIONS:
        abs_plastic.AddElementByMassFraction(element(element_name), fraction)

    resin = G4Material("R4600", R4600_DENSITY_G_CM3 * g / cm3, 3)
    for element_name, fraction in R4600_MASS_FRACTIONS:
        resin.AddElementByMassFraction(element(element_name), fraction)

    materials = {
        "CsI_Tl": csi,
        "ABS": abs_plastic,
        "R4600": resin,
        "G4_AIR": nist.FindOrBuildMaterial("G4_AIR"),
    }
    densities: dict[str, float] = {
        "CsI_Tl": CSI_TL_DENSITY_G_CM3,
        "ABS": ABS_DENSITY_G_CM3,
        "R4600": R4600_DENSITY_G_CM3,
    }

    def require(name: str, density: float | None) -> None:
        if name.startswith("G4_"):
            if name not in materials:
                materials[name] = nist.FindOrBuildMaterial(name)
            return
        if name in PREBUILT_MATERIALS:
            # Built unconditionally above; a source geometry (shield/container)
            # may reference it without a density (Ra226/Th232 use R4600).
            if density is not None:
                value = _require_custom_density(name, density)
                previous = densities.get(name)
                if previous is not None and abs(previous - value) > 1e-9:
                    raise ValidationError(
                        f"material {name!r} requested with conflicting densities "
                        f"{previous:.6g} and {value:.6g} g/cm^3"
                    )
            return
        _check_material_lookup(name)
        value = _require_custom_density(name, density)
        previous = densities.get(name)
        if previous is not None and abs(previous - value) > 1e-9:
            raise ValidationError(
                f"material {name!r} requested with conflicting densities "
                f"{previous:.6g} and {value:.6g} g/cm^3"
            )
        if name in materials:
            return
        densities[name] = value
        atoms = CUSTOM_MATERIAL_ATOMS[name]
        material = G4Material(name, value * g / cm3, len(atoms))
        for element_name, count in atoms:
            material.AddElementByNumberOfAtoms(element(element_name), count)
        materials[name] = material

    for spec in specs:
        if not isinstance(spec, SourceSpec):  # pragma: no cover - defensive
            raise ValidationError(f"expected a SourceSpec, got {type(spec).__name__}")
        require(spec.material, spec.effective_density)
        geometry = spec.geometry
        if isinstance(geometry, Sandwich):
            for layer in geometry.layers:
                require(layer.material, None)
        if spec.container is not None:
            require(spec.container.material, None)
        if spec.shield is not None:
            # The shield material is registered too (the builder relied
            # on R4600 being built unconditionally, which would break for any
            # other shield material).
            require(spec.shield.material, None)

    return materials
