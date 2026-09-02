"""Energy-dependent normalization scale model.

The scale corrects the overall difference between the simulated and the real
experimental spectra and is allowed to vary with energy.  It is a linear
interpolant between two reference energies ``E_lo`` and ``E_hi`` (the
dataset's fit-window lower/upper bin centers), with 2 parameters per dataset:

   s(E) = s0 + (s1 - s0) (E - E_lo) / (E_hi - E_lo).

``s0`` and ``s1`` are the scale values at ``E_lo`` and ``E_hi`` respectively, so
both are directly interpretable on-curve values with the same units.  The
reference energies are not fixed: they are recomputed at every evaluation
from the current calibration (the fit-window bin centers move with the energy
calibration), so ``s0``/``s1`` always anchor the current fit window.  With
``s0 = s1`` the scale reduces to that constant (the initial ``x0``), matching the
previous constant-scale behavior.  During fitting both values share per-dataset
bounds relative to the initial overall normalization (``[SCALE_REL_LO,
SCALE_REL_HI] * initial_scale``).
"""

from __future__ import annotations

import re

import numba

N_SCALE = 2  # (s0, s1) scale at the lower/upper reference energies
PARAM_NAMES_SCALE = ["s0", "s1"]

# The fitted scale reference values may vary within [LO, HI] * initial_scale
# around each dataset's initial overall-normalization estimate (both start there).
SCALE_REL_LO = 0.05
SCALE_REL_HI = 1.95


def scale_bounds(initial_scale: float) -> list[tuple[float, float]]:
    """Per-dataset scale bounds: each reference value within ``[LO, HI] * initial_scale``."""
    initial_scale = float(initial_scale)
    return [(SCALE_REL_LO * initial_scale, SCALE_REL_HI * initial_scale)] * N_SCALE


_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_names(label: str, index: int) -> list[str]:
    """Per-dataset scale-parameter names ``(s0, s1)``."""
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_") or str(index)
    return [f"{p}_{clean}" for p in PARAM_NAMES_SCALE]


@numba.njit(inline="always", cache=True)
def scale_model(scale_params, energy, energy_low, energy_high):
    """Linear scale between the two reference points.

    ``scale_params = [s0, s1]`` are the scale values at the reference
    energies ``energy_low`` and ``energy_high``; ``s(E) = s0 + (s1 - s0) t``
    with ``t = (E - energy_low) / (energy_high - energy_low)``.  ``scale_params``
    is a float64 array of length 2 and ``energy`` a float64 scalar or array.
    """
    t = (energy - energy_low) / (energy_high - energy_low)
    return scale_params[0] + (scale_params[1] - scale_params[0]) * t
