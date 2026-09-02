"""Small shared helpers for kc761calib."""

from __future__ import annotations

import numba
import numpy as np

_MISSING = object()


@numba.njit(inline="always", cache=True)
def _bernstein_basis(t, degree):
    """JIT core of :func:`bernstein_basis` (no argument validation).

    ``t`` is a float64 scalar or 1-D float64 array; the binomial coefficients
    are computed with the exact recurrence ``C(d, i) = C(d, i-1) * (d - i +
    1) / i`` kept in float64, because ``math.comb`` is not available under
    numba and the int64 form overflows for ``degree`` beyond ~65.
    """
    t = np.asarray(t, dtype=np.float64)
    omt = 1.0 - t
    out = np.empty(t.shape + (degree + 1,), dtype=np.float64)
    comb = 1.0
    for i in range(degree + 1):
        out[..., i] = comb * omt ** (degree - i) * t ** i
        if i < degree:
            comb = comb * (degree - i) / (i + 1)
    return out


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
        raise ValueError(
            f"degree must be a non-negative integer, got {degree!r}")
    return _bernstein_basis(t, degree)


@numba.njit(inline="always", cache=True)
def quadratic_bezier_param(x, x_lo, x_hi, x_mid):
    """Curve parameter ``t(x)`` of a quadratic Bezier with control abscissae.

    The control points have abscissae ``(x_lo, x_mid, x_hi)`` with
    ``x_lo < x_mid < x_hi``.  The curve abscissa is

        x(t) = x_lo + 2 (x_mid - x_lo) t + (x_lo - 2 x_mid + x_hi) t^2,

    strictly increasing on ``t`` in [0, 1] (its derivative is the positive
    convex combination ``2 ((1 - t)(x_mid - x_lo) + t (x_hi - x_mid))``), so
    every interval point has a unique parameter: the in-interval root of the
    quadratic, written in the rationalized form

        t = (x - x_lo) / (x_mid - x_lo + sqrt((x_mid - x_lo)^2
            + (x_lo - 2 x_mid + x_hi) (x - x_lo))),

    which is exact at both endpoints (t = 0 and t = 1) and free of
    cancellation for any ``x_mid`` inside the interval.  ``x`` is a float64
    scalar or array; the result has the same shape.
    """
    a = x_mid - x_lo
    d = x - x_lo
    return d / (a + np.sqrt(a * a + (x_lo - 2.0 * x_mid + x_hi) * d))


@numba.njit(inline="always", cache=True)
def quadratic_bezier(x, x_lo, x_hi, x_mid, y0, y1, y2):
    """Quadratic Bezier curve ``y(x)`` with control points ``(x_lo, y0)``,
    ``(x_mid, y1)`` and ``(x_hi, y2)``, with ``x_lo < x_mid < x_hi``.

    The value is the Bezier ordinate at the curve parameter mapping to ``x``
    (:func:`quadratic_bezier_param`):

        y = y0 + 2 (y1 - y0) t + (y0 - 2 y1 + y2) t^2.

    ``x`` is a float64 scalar or array; the result has the same shape.
    """
    t = quadratic_bezier_param(x, x_lo, x_hi, x_mid)
    return y0 + (2.0 * (y1 - y0) + (y0 - 2.0 * y1 + y2) * t) * t


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
