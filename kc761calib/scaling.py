"""Per-dataset normalization scale model over the fit channel window.

The scale corrects the overall difference between the simulated and the real
experimental spectra and is allowed to vary across the dataset's fixed fit
channel window ``[ch_lo, ch_hi]`` (0-based, inclusive channels).  It is the
quadratic Bezier curve with control points ``(ch_lo, s1)``, ``(s0, s2)`` and
``(ch_hi, s3)`` -- 4 parameters per dataset -- written as a function of the
channel ``ch`` by inverting the control-abscissa curve:

   t(ch) = (ch - ch_lo) / (s0 - ch_lo
           + sqrt((s0 - ch_lo)^2 + (ch_lo - 2 s0 + ch_hi) (ch - ch_lo))),
   s(ch) = s1 + 2 (s2 - s1) t(ch) + (s1 - 2 s2 + s3) t(ch)^2.

``s0`` is the channel of the middle control point (bounded strictly inside the
fit window, ``ch_lo < s0 < ch_hi``) and ``s1, s2, s3`` are the scale values at
the control abscissae ``(ch_lo, s0, ch_hi)``, so all three are directly
interpretable on-curve values in scale units.  With ``s1 = s2 = s3`` the scale
reduces to that constant (the initial ``x0``, whose ``s0`` sits at the window
midpoint).  During fitting ``s1, s2, s3`` share per-dataset bounds relative to
the initial overall normalization (``[SCALE_REL_LO, SCALE_REL_HI] *
initial_scale``).
"""

from __future__ import annotations

import re

import numba

from .util import quadratic_bezier

N_SCALE = 4  # (s0, s1, s2, s3) control channel + scale at (ch_lo, s0, ch_hi)
PARAM_NAMES_SCALE = ["s0", "s1", "s2", "s3"]

# s0 must stay strictly inside the fit window (ch_lo < s0 < ch_hi) so that the
# curve parameter t(ch) is well-defined at every bin center.  The bounds are
# inset by this tiny channel margin; it also keeps the discriminant
# (s0 - ch_lo)^2 + (ch_lo - 2 s0 + ch_hi)(ch - ch_lo) strictly positive at the
# window endpoints in float64: its true minimum there is the margin squared,
# while the rounding error of the ~window^2 terms is ~1e-16 * window^2, so
# this margin is safe for any window up to ~2048 channels.
S0_MARGIN = 1e-3

# The fitted scale values may vary within [LO, HI] * initial_scale around each
# dataset's initial overall-normalization estimate (all start there).
SCALE_REL_LO = 0.01
SCALE_REL_HI = 3.0


def scale_bounds(initial_scale: float, channel_low: int,
                 channel_high: int) -> list[tuple[float, float]]:
    """Per-dataset scale bounds for the fit window ``[channel_low, channel_high]``.

    ``s0`` within ``(channel_low, channel_high)`` inset by ``S0_MARGIN``
    (implementing the strict constraint ``ch_lo < s0 < ch_hi``); ``s1, s2, s3``
    each within ``[SCALE_REL_LO, SCALE_REL_HI] * initial_scale``.
    """
    initial_scale = float(initial_scale)
    s0_bounds = (float(channel_low) + S0_MARGIN,
                 float(channel_high) - S0_MARGIN)
    if s0_bounds[1] <= s0_bounds[0]:
        raise ValueError(
            f"fit channel window [{channel_low}, {channel_high}] is too narrow "
            "to hold the scale control channel s0")
    value_bounds = (SCALE_REL_LO * initial_scale, SCALE_REL_HI * initial_scale)
    return [s0_bounds, value_bounds, value_bounds, value_bounds]


_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_names(label: str, index: int) -> list[str]:
    """Per-dataset scale-parameter names ``(s0, s1, s2, s3)``."""
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_") or str(index)
    return [f"{p}_{clean}" for p in PARAM_NAMES_SCALE]


@numba.njit(inline="always", cache=True)
def scale_model(scale_params, channel, channel_low, channel_high):
    """Quadratic Bezier scale over the fixed fit channel window.

    ``scale_params = [s0, s1, s2, s3]``: ``s0`` is the middle control channel
    and ``s1, s2, s3`` are the scale values at the control abscissae
    ``(channel_low, s0, channel_high)`` -- i.e. the quadratic Bezier curve
    with control points ``(channel_low, s1)``, ``(s0, s2)``,
    ``(channel_high, s3)`` evaluated at ``channel``.  ``scale_params`` is a
    float64 array of length 4 and ``channel`` a float64 scalar or array of
    channel values; ``channel_low``/``channel_high`` bound the curve's domain.
    """
    s0 = scale_params[0]
    s1 = scale_params[1]
    s2 = scale_params[2]
    s3 = scale_params[3]
    return quadratic_bezier(channel, channel_low, channel_high, s0, s1, s2, s3)
