"""Energy-dependent normalization scale model.

The scale corrects the overall difference between the simulated and the real
experimental spectra and is allowed to vary with energy:

   s(E) = s0 * 2 / (1 + exp(s1*(E - s2)))  =  s0 * 2 * expit(s1*(s2 - E)).

With the defaults ``s1 = s2 = 0`` this reduces to the constant ``s0``.  ``s0``
and ``s1`` have the fixed bounds below; ``s2``'s bound is per dataset (see
``fitparamspace.FitParamSpace.s2_bounds``) because it is tied to each dataset's
fit window.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import expit

N_SCALE = 3  # (s0, s1, s2)
PARAM_NAMES_SCALE = ["s0", "s1", "s2"]
BOUNDS_SCALE = [(1e-3, 1e3), (-0.05, 0.05)]

_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_names(label: str, index: int) -> list[str]:
    """Per-dataset scale-parameter names ``(s0, s1, s2)``."""
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_") or str(index)
    return [f"{p}_{clean}" for p in PARAM_NAMES_SCALE]


def scale_model(s: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    """s(E) = s0 * 2 / (1 + exp(s1*(E - s2))) for ``s = [s0, s1, s2]``."""
    s0, s1, s2 = np.asarray(s, dtype=float)[:3]
    e = np.asarray(e, dtype=float)
    return s0 * 2.0 * expit(s1 * (s2 - e))
