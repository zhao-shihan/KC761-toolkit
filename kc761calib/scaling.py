"""Energy-dependent normalization scale model.

The scale corrects the overall difference between the simulated and the real
experimental spectra and is allowed to vary with energy.  It is a cubic
polynomial parameterized by the function values at four fixed energy knots
inside each dataset's fit window ``[elow, ehigh]``:

   knots  = (E_min, .75*E_min + .25*E_max, .25*E_min + .75*E_max, E_max)
   s(E)   = cubic interpolant of the four values (s0, s1, s2, s3) at the knots.

With all four values equal the curve reduces to that constant (the initial
``x0``), matching the old sigmoid's ``s1 = s2 = 0`` behavior.  During fitting the
four knots share per-dataset bounds relative to that initial overall
normalization (``[SCALE_REL_LO, SCALE_REL_HI] * s_initial``).
"""

from __future__ import annotations

import re

import numpy as np

N_SCALE = 4  # (s0, s1, s2, s3) values at the four knots
PARAM_NAMES_SCALE = ["s0", "s1", "s2", "s3"]

# The fitted scale knots may vary within [LO, HI] * s_initial around each
# dataset's initial overall-normalization estimate (s0..s3 all start there).
SCALE_REL_LO = 0.05
SCALE_REL_HI = 1.95


def scale_bounds(initial: float) -> list[tuple[float, float]]:
    """Per-dataset scale bounds: each knot within ``[LO, HI] * initial``."""
    initial = float(initial)
    return [(SCALE_REL_LO * initial, SCALE_REL_HI * initial)] * N_SCALE


_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_names(label: str, index: int) -> list[str]:
    """Per-dataset scale-parameter names ``(s0, s1, s2, s3)``."""
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_") or str(index)
    return [f"{p}_{clean}" for p in PARAM_NAMES_SCALE]


def scale_knots(elow: float, ehigh: float) -> np.ndarray:
    """Four cubic knots on the fit window ``(E_min, E_max)``."""
    elow, ehigh = float(elow), float(ehigh)
    return np.array([elow,
                     0.75 * elow + 0.25 * ehigh,
                     0.25 * elow + 0.75 * ehigh,
                     ehigh])


# After normalizing E to t = (E - E_min)/(E_max - E_min) the knots sit at the
# fixed positions (0, .25, .75, 1), so the monomial coefficients of the
# interpolating cubic are a fixed linear map of the four values (well
# conditioned: all nodes lie in [0, 1]).
_SCALE_KNOTS_T = np.array([0.0, 0.25, 0.75, 1.0])
_SCALE_COEF_MAP = np.linalg.inv(
    np.stack([_SCALE_KNOTS_T**k for k in range(N_SCALE)], axis=1))


def scale_model(s: np.ndarray | list[float], e: np.ndarray | float,
                elow: float, ehigh: float) -> np.ndarray:
    """Cubic s(E) interpolating the four values ``s=[s0, s1, s2, s3]``.

    ``s0``..``s3`` are the scale values at ``scale_knots(elow, ehigh)``; the
    unique cubic through them is evaluated in monomial form by Horner.
    """
    vals = np.asarray(s, dtype=float)[:N_SCALE]
    e = np.asarray(e, dtype=float)
    t = (e - elow) / (ehigh - elow)
    c0, c1, c2, c3 = _SCALE_COEF_MAP @ vals
    return c0 + t * (c1 + t * (c2 + t * c3))
