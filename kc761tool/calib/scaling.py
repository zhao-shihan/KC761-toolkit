"""Per-dataset quadratic-Bezier scale model (F-CAL-2).

The scale corrects the overall difference between the simulated and the real
spectra across a dataset's fixed fit channel window ``[x_lo, x_hi]``. It is the
quadratic **Bezier curve** with control points ``(x_lo, s1)``, ``(s0, s2)`` and
``(x_hi, s3)`` (the frozen form, docs/plan.md D-103), written as a function of
the channel by inverting the control-abscissa curve:

    x(t) = x_lo + 2 (s0 - x_lo) t + (x_lo - 2 s0 + x_hi) t^2,
    s(t) = (1-t)^2 s1 + 2 (1-t) t s2 + t^2 s3,
    t(x) = d / (a + sqrt(a^2 + c d)), a = s0 - x_lo, c = x_lo - 2 s0 + x_hi, d = x - x_lo.

``s0`` is the middle control **abscissa** (a free fit parameter) and
``s1, s2, s3`` are the scale values at the control abscissae
``(x_lo, s0, x_hi)``. The curve is a parabola in the ``(x, s)`` plane; ``s(x)``
is a polynomial only in the degenerate case ``s0 = (x_lo + x_hi)/2``, where the
abscissa parametrization becomes linear and the family reduces to the
quadratic polynomials (the degree-2 Bernstein basis). Fixing ``s0`` would
therefore change the model class and is **not** done.

The value and its exact parameter derivatives are analytic (D-60/F-CAL-2); no
finite differences enter production.
"""

from __future__ import annotations

import re
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761tool.core._checks import as_float_array
from kc761tool.errors import ValidationError

N_SCALE: Final = 4
"""Bezier parameters per dataset ``(s0, s1, s2, s3)`` (F-CAL-2)."""

PARAM_NAMES_SCALE: Final[tuple[str, ...]] = ("s0", "s1", "s2", "s3")

S0_MARGIN: Final = 1e-3
"""Inset keeping ``s0`` strictly inside the window (F-CAL-2)."""

SCALE_REL_LO: Final = 0.01
SCALE_REL_HI: Final = 3.0
"""Scale values are bounded to ``[LO, HI] * initial_scale`` (D-103)."""

_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_names(label: str, index: int) -> tuple[str, ...]:
    """Per-dataset scale-parameter names ``(s0, s1, s2, s3)`` for reports."""
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_") or str(index)
    return tuple(f"{name}_{clean}" for name in PARAM_NAMES_SCALE)


def scale_bounds(
    initial_scale: float, channel_low: int, channel_high: int
) -> tuple[tuple[float, float], ...]:
    """Per-dataset bounds of ``(s0, s1, s2, s3)`` (D-103).

    ``s0`` lies in ``(channel_low, channel_high)`` inset by :data:`S0_MARGIN`;
    ``s1, s2, s3`` each lie in ``[SCALE_REL_LO, SCALE_REL_HI] * initial_scale``.
    """
    scale = float(initial_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValidationError(
            f"initial_scale must be positive and finite, got {initial_scale!r}"
        )
    s0_bounds = (
        float(channel_low) + S0_MARGIN,
        float(channel_high) - S0_MARGIN,
    )
    if s0_bounds[1] <= s0_bounds[0]:
        raise ValidationError(
            f"fit channel window [{channel_low}, {channel_high}] is too narrow "
            "to hold the scale control channel s0 (F-CAL-2)"
        )
    value_bounds = (SCALE_REL_LO * scale, SCALE_REL_HI * scale)
    return (s0_bounds, value_bounds, value_bounds, value_bounds)


def _check_scale_params(scale_params: NDArray[np.float64]) -> NDArray[np.float64]:
    params = as_float_array("scale_params", scale_params, ndim=1)
    if params.size != N_SCALE:
        raise ValidationError(
            f"scale_params must have {N_SCALE} entries, got {params.size}"
        )
    return params


def _check_window(channel_low: int, channel_high: int, s0: float) -> None:
    if not 0 <= channel_low <= channel_high:
        raise ValidationError(
            f"channel window [{channel_low}, {channel_high}] must satisfy 0 <= low <= high"
        )
    if not (float(channel_low) < s0 < float(channel_high)):
        raise ValidationError(
            f"scale control channel s0={s0!r} must lie strictly inside "
            f"({channel_low}, {channel_high}) (F-CAL-2)"
        )


def _bezier_parameter(
    channels: NDArray[np.float64],
    channel_low: float,
    channel_high: float,
    s0: float,
) -> NDArray[np.float64]:
    """Curve parameter ``t(channel)`` of the control-abscissa Bezier (F-CAL-2).

    ``x(t) = x_lo + 2 (s0 - x_lo) t + (x_lo - 2 s0 + x_hi) t^2`` is strictly
    increasing for ``s0`` strictly inside the window, with the in-interval root
    written in the rationalized, cancellation-free form
    ``t = d / (a + sqrt(a^2 + c d))``.
    """
    a = float(s0) - channel_low
    d = channels - channel_low
    discriminant = a * a + (channel_low - 2.0 * float(s0) + channel_high) * d
    if np.any(discriminant < 0.0):
        raise ValidationError(
            "Bezier discriminant is negative; s0 is outside the window (F-CAL-2)"
        )
    return d / (a + np.sqrt(discriminant))


def scale_curve(
    scale_params: NDArray[np.float64],
    channels: NDArray[np.float64],
    channel_low: int,
    channel_high: int,
) -> NDArray[np.float64]:
    """Scale ``s(channel)`` of the quadratic Bezier ``(s0, s1, s2, s3)``."""
    params = _check_scale_params(scale_params)
    _check_window(channel_low, channel_high, float(params[0]))
    channel_values = as_float_array("channels", channels)
    t = _bezier_parameter(
        channel_values, float(channel_low), float(
            channel_high), float(params[0])
    )
    s1, s2, s3 = (float(value) for value in params[1:])
    return s1 + (2.0 * (s2 - s1) + (s1 - 2.0 * s2 + s3) * t) * t


def scale_curve_grad(
    scale_params: NDArray[np.float64],
    channels: NDArray[np.float64],
    channel_low: int,
    channel_high: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scale and its exact derivatives ``d s / d(s0..s3)`` (F-CAL-2).

    Returns ``(value, gradient)`` with ``value`` of the channel shape and
    ``gradient`` of shape ``(4, *channel.shape)``.
    """
    params = _check_scale_params(scale_params)
    _check_window(channel_low, channel_high, float(params[0]))
    channel_values = as_float_array("channels", channels)
    s0, s1, s2, s3 = (float(value) for value in params)

    t = _bezier_parameter(channel_values, float(
        channel_low), float(channel_high), s0)
    omt = 1.0 - t
    value = s1 + (2.0 * (s2 - s1) + (s1 - 2.0 * s2 + s3) * t) * t

    gradient = np.empty((N_SCALE, *value.shape), dtype=np.float64)
    gradient[1] = omt * omt
    gradient[2] = 2.0 * omt * t
    gradient[3] = t * t

    dx_dt = 2.0 * (s0 - float(channel_low)) + 2.0 * (
        float(channel_low) - 2.0 * s0 + float(channel_high)
    ) * t
    if np.any(dx_dt <= 0.0):
        raise ValidationError(
            "Bezier abscissa is not strictly increasing on the window (F-CAL-2)"
        )
    ds_dt = 2.0 * (s2 - s1) + 2.0 * (s1 - 2.0 * s2 + s3) * t
    gradient[0] = ds_dt * (-2.0 * t * omt / dx_dt)
    return value, gradient


__all__ = [
    "N_SCALE",
    "PARAM_NAMES_SCALE",
    "S0_MARGIN",
    "SCALE_REL_HI",
    "SCALE_REL_LO",
    "scale_bounds",
    "scale_curve",
    "scale_curve_grad",
    "scale_names",
]
