"""Shared plotting utilities (W3/W4).

One implementation of the style, colour palette, saving and axis helpers,
used by both the calibration and the unfolding reports. The visual style
follows the pre-rewrite figures (docs/plan.md D-70). The implementation lives
in :mod:`kc761.plotting.style` and is re-exported here so callers import from
one place.
"""

from __future__ import annotations

from kc761.plotting.style import (
    CALIB_BAND_SCALE,
    COLOR_CALIB,
    COLOR_DATA,
    COLOR_FIT,
    COLOR_PARAM_BOX,
    COLOR_PARAM_EDGE,
    COLOR_RESIDUAL_LEVEL,
    COLOR_RESIDUAL_POINTS,
    COLOR_RESIDUAL_ZERO,
    COLOR_RESOL,
    COLOR_SCALE,
    REFERENCE_LINES_KEV,
    RESIDUAL_MAX,
    RESOL_BAND_SCALE,
    apply_style,
    save_figure,
    style_axes,
)

__all__ = [
    "CALIB_BAND_SCALE",
    "COLOR_CALIB",
    "COLOR_DATA",
    "COLOR_FIT",
    "COLOR_PARAM_BOX",
    "COLOR_PARAM_EDGE",
    "COLOR_RESIDUAL_LEVEL",
    "COLOR_RESIDUAL_POINTS",
    "COLOR_RESIDUAL_ZERO",
    "COLOR_RESOL",
    "COLOR_SCALE",
    "REFERENCE_LINES_KEV",
    "RESIDUAL_MAX",
    "RESOL_BAND_SCALE",
    "apply_style",
    "save_figure",
    "style_axes",
]
