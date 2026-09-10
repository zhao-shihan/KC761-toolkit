"""Shared figure style, palette and saving helpers (D-70).

One implementation of the visual conventions used by every report so the
calibration and unfolding figures cannot drift apart. The palette follows the
pre-rewrite figures (D-70). Saving is atomic like the product protocol
(F-IO-1/D-17): an existing plot is refused unless ``force`` is set, the figure
goes to a ``.part`` file and is renamed onto the target afterwards.

This module is a leaf: it must not import ``kc761.core``, ``schema``,
``calib``, ``unfold``, ``sim`` or ``cli``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import matplotlib

# Must run before pyplot is imported; 3.11+ ignores a use() after it.
matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backend_bases import FigureCanvasBase  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from kc761.errors import UsageError  # noqa: E402

#: Artist colours, grouped by role. Names match the pre-rewrite figures.
COLOR_DATA: Final = "blue"
COLOR_FIT: Final = "red"
COLOR_SIM_RAW: Final = "dimgray"
COLOR_SCALE: Final = "seagreen"
COLOR_RESIDUAL_POINTS: Final = "darkgoldenrod"
COLOR_RESIDUAL_ZERO: Final = "black"
COLOR_RESIDUAL_LEVEL: Final = "red"
COLOR_REF_LINE: Final = "dimgray"
COLOR_CALIB: Final = "darkgreen"
COLOR_RESOL: Final = "darkolivegreen"
COLOR_PARAM_BOX: Final = "white"
COLOR_PARAM_EDGE: Final = "gray"

#: Fixed residual-panel half-range so every dataset panel is comparable.
RESIDUAL_MAX: Final = 0.6

#: Reference gamma lines (keV) marked on calibration/resolution curves.
REFERENCE_LINES_KEV: Final = (59.54, 661.66, 2614.51)

#: Covariance-band magnification (the 1-sigma bands are scaled for visibility;
#: the legends state the factor).
CALIB_BAND_SCALE: Final = 30.0
RESOL_BAND_SCALE: Final = 10.0


def apply_style() -> None:
    """Apply the shared rcParams (idempotent)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "axes.grid": False,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def style_axes(ax, *, grid: bool = True) -> None:
    """Apply the shared per-axes conventions (inward ticks, optional grid)."""
    ax.tick_params(direction="in", which="both", top=True, right=True)
    if grid:
        ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def save_figure(fig: Figure, path: str | Path, *, force: bool = False) -> Path:
    """Save a figure atomically, inferring the format from the extension.

    A missing or unrecognized extension falls back to PDF. An existing target
    is refused unless ``force`` is set (D-17). The figure is closed afterwards
    and the final output path is returned.
    """
    target = Path(path)
    supported = set(FigureCanvasBase.get_supported_filetypes())
    fmt = target.suffix.lower().lstrip(".")
    if fmt not in supported:
        target = target.with_suffix(".pdf")
        fmt = "pdf"
    if target.exists() and not force:
        raise UsageError(f"refusing to overwrite existing plot {target}; pass --force")
    if target.is_dir():
        raise UsageError(f"plot target {target} is a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    try:
        fig.savefig(part, format=fmt, bbox_inches="tight", pad_inches=0.5)
        os.replace(part, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    return target


__all__ = [
    "CALIB_BAND_SCALE",
    "COLOR_CALIB",
    "COLOR_DATA",
    "COLOR_FIT",
    "COLOR_PARAM_BOX",
    "COLOR_PARAM_EDGE",
    "COLOR_REF_LINE",
    "COLOR_RESIDUAL_LEVEL",
    "COLOR_RESIDUAL_POINTS",
    "COLOR_RESIDUAL_ZERO",
    "COLOR_RESOL",
    "COLOR_SCALE",
    "COLOR_SIM_RAW",
    "REFERENCE_LINES_KEV",
    "RESIDUAL_MAX",
    "RESOL_BAND_SCALE",
    "apply_style",
    "save_figure",
    "style_axes",
]
