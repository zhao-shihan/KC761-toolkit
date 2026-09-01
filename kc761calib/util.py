"""Small shared helpers for kc761calib."""

from __future__ import annotations

import numpy as np

_MISSING = object()


def bezier2_basis(t: np.ndarray | float) -> np.ndarray:
    """Quadratic Bernstein basis on the normalized parameter t in [0, 1].

    Returns the last-axis vector [B0, B1, B2] with
        B0 = (1-t)^2,  B1 = 2(1-t)t,  B2 = t^2.
    The weights are a non-negative partition of unity, so a flat control
    triplet yields a constant and any evaluation lies inside the convex hull
    of the three control values. ``t`` is expected to already be normalized;
    callers map their own coordinate to [0, 1] (see ``scale_model``).
    """
    t = np.asarray(t, dtype=float)
    omt = 1.0 - t
    return np.stack([omt * omt, 2.0 * omt * t, t * t], axis=-1)


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
