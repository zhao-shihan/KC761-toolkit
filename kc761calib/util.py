"""Small shared helpers for kc761calib."""

from __future__ import annotations

import math

import numpy as np

_MISSING = object()


def bernstein_basis(t: np.ndarray | float, degree: int) -> np.ndarray:
    """Bernstein basis polynomials of ``degree`` on the normalized ``t`` in [0, 1].

    Returns the last-axis vector ``[B_0, ..., B_degree]`` with
        B_i(t) = C(degree, i) (1-t)^(degree-i) t^i.
    The basis is a non-negative partition of unity, so a flat coefficient
    vector yields a constant and any weighted sum lies inside the convex hull
    of the coefficients; weighted by control values it is the corresponding
    Bezier curve.  ``t`` is expected to already be normalized; callers map
    their own coordinate to [0, 1].
    """
    if not isinstance(degree, int) or degree < 0:
        raise ValueError(f"degree must be a non-negative integer, got {degree!r}")
    t = np.asarray(t, dtype=float)
    omt = 1.0 - t
    out = np.empty(t.shape + (degree + 1,))
    for i in range(degree + 1):
        out[..., i] = math.comb(degree, i) * omt ** (degree - i) * t ** i
    return out


def broadcast(value, n: int, name: str, default=_MISSING):
    """Expand None/scalar/1-or-n sequence into a length-n list."""
    if value is None:
        if default is _MISSING:
            raise ValueError(f"{name}: value is None (expected a scalar or "
                             f"sequence of length 1 or {n})")
        return [default] * n
    if np.isscalar(value) or isinstance(value, (str, bytes)):
        return [value] * n
    seq = list(value)
    if len(seq) == n:
        return seq
    if len(seq) == 1:
        return seq * n
    raise ValueError(f"{name}: expected 1 or {n} values, got {len(seq)}")
