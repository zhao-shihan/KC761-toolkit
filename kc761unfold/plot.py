"""Plot an UnfoldResult.

Unfold mode: three stacked panels with height ratio 2:2:1 -- a linear
y-axis spectrum panel, a logarithmic y-axis spectrum panel and a
relative-residual panel.  Calibration-only mode: the two spectrum
panels alone.  The energy x-axis of the spectrum and residual panels
is logarithmic; bins whose calibration energy is zero or negative (the
lowest channels, c0 < 0) cannot be shown on a log axis and are
truncated at the first bin ending above zero.  Spectrum layers are
drawn as histograms (stairs) with two nested uncertainty bands: the
outer (lighter) band carries the total uncertainty, the inner
(stronger) one the systematic part.  The palette
follows kc761calib: blue data, red unfolded spectrum, dimgray refolded
prediction, darkgoldenrod residual points with red +/-0.3 level guides.
"""

from __future__ import annotations
from .types import UnfoldResult
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path

import matplotlib
matplotlib.use("Agg")


# Palette (kc761calib conventions): colors of the plotted artists.
_COLOR_DATA = "blue"  # calibrated spectrum (histogram + uncertainty bars)
_COLOR_FIT = "red"  # unfolded spectrum (histogram + uncertainty bars)
_COLOR_REFOLD = "dimgray"  # refolded prediction (stairs)
_COLOR_RESIDUAL_POINTS = "darkgoldenrod"  # residual points
_COLOR_RESIDUAL_ZERO = "black"  # residual zero line
_COLOR_RESIDUAL_LEVEL = "red"  # residual +/- level guides

_BAND_ALPHA_TOTAL = 0.15  # outer band: the total uncertainty
_BAND_ALPHA_SYST = 0.30  # inner band: the systematic part
_RESIDUAL_MAX = 0.6


def _save_fig(fig, out_plot: str) -> Path:
    for ax in fig.axes:
        ax.tick_params(direction="in", which="both")
    out = Path(out_plot)
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in FigureCanvasBase.get_supported_filetypes():
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def _positive_start(edges: np.ndarray) -> int:
    """First bin index whose lower edge is positive (log-x truncation).

    The calibration maps the lowest channels to energies <= 0 (c0 < 0
    by construction): bins at or below zero cannot be drawn on a
    logarithmic axis, so the spectrum and residual panels are truncated
    at the first bin fully above zero.
    """
    return int(np.searchsorted(np.asarray(edges, dtype=float), 0.0,
                               side="right"))


def _draw_layer(ax, centers: np.ndarray, edges: np.ndarray,
                counts: np.ndarray, sigma_total: np.ndarray,
                sigma_syst: np.ndarray, color: str, label: str,
                ls: str = "-") -> tuple:
    """One spectrum layer: histogram plus the two nested uncertainty bands."""
    total_band = ax.fill_between(centers,
                                 np.maximum(counts - sigma_total, 0.0),
                                 counts + sigma_total, step="mid",
                                 color=color, alpha=_BAND_ALPHA_TOTAL,
                                 linewidth=0)
    syst_band = ax.fill_between(centers,
                                np.maximum(counts - sigma_syst, 0.0),
                                counts + sigma_syst, step="mid",
                                color=color, alpha=_BAND_ALPHA_SYST,
                                linewidth=0)
    stairs = ax.stairs(counts, edges, lw=0.8, color=color,
                       ls=ls, alpha=0.8)
    return stairs, total_band, syst_band, label


def _spectrum_panel(ax, result: UnfoldResult, title: str, *, log: bool):
    # Truncate at the first bin fully above zero: the calibration maps
    # the lowest channels to energies <= 0, which a log x-axis cannot
    # show.
    lo = _positive_start(result.energy_edges)
    centers = result.centers[lo:]
    edges = result.energy_edges[lo:]
    # bottom layer: the calibrated spectrum (raw counts on the energy
    # axis); dashed when it is a reference next to the unfolded result,
    # solid when it is the result itself (calibration-only mode).
    calib_ls = "--" if not result.calib_only else "-"
    calib = _draw_layer(ax, centers, edges,
                        result.data_counts[lo:],
                        result.data_sigma_total[lo:],
                        result.data_sigma_syst[lo:],
                        _COLOR_DATA, "Calibrated spectrum", ls=calib_ls)
    refolded = None
    unfolded = None
    if not result.calib_only:
        # middle layer: the refolded prediction (dotted histogram, no bands)
        refolded = ax.stairs(result.refolded[lo:], edges, ls=":",
                             lw=0.8, color=_COLOR_REFOLD)
        # top layer: the unfolded spectrum
        unfolded = _draw_layer(ax, centers, edges,
                               result.counts[lo:],
                               result.sigma_total[lo:],
                               result.sigma_syst[lo:],
                               _COLOR_FIT, "Unfolded spectrum")
    if log:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)

    # Legend order: unfolded first, then the refolded prediction, then
    # the calibrated spectrum; each spectrum layer shares one entry for
    # its histogram + total-uncertainty band and a separate one for the
    # systematic band.
    handles_out = []
    labels_out = []
    if unfolded is not None:
        handles_out += [(unfolded[0], unfolded[1]), unfolded[2], refolded]
        labels_out += [unfolded[3], "Syst. unc. band",
                       "Refolded spectrum"]
    handles_out += [(calib[0], calib[1]), calib[2]]
    labels_out += [calib[3], "Syst. unc. band"]
    ax.legend(handles_out, labels_out, fontsize=8, loc="upper right",
              ncol=2)


def _residual_panel(ax, result: UnfoldResult, title: str):
    # Same positive-energy truncation as the spectrum panels so the
    # residual panel covers the same logarithmic x range.
    lo = _positive_start(result.energy_edges)
    centers = result.centers[lo:]
    data_counts = result.data_counts[lo:]
    refolded = result.refolded[lo:]
    sigma_total = result.data_sigma_total[lo:]
    ok = refolded > 0.0
    rel = (data_counts[ok] - refolded[ok]) / refolded[ok]
    ax.errorbar(centers[ok], rel,
                yerr=sigma_total[ok] / refolded[ok],
                fmt="o", ms=1.5, lw=0.8, color=_COLOR_RESIDUAL_POINTS,
                alpha=0.8)
    ax.axhline(0.0, color=_COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=_COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    ax.set_xscale("log")
    ax.set_xlim(result.energy_edges[lo], result.energy_edges[-1])
    ax.set_ylim(-_RESIDUAL_MAX, _RESIDUAL_MAX)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def plot_result(result: UnfoldResult, out_path: str | Path) -> Path:
    """Render the result; returns the final output path."""
    n_panels = 2 if result.calib_only else 3
    fig = plt.figure(figsize=(9.5, 10.5))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=[2.0, 2.0, 1.0]
                          if n_panels == 3 else [1.0, 1.0],
                          hspace=0.35)
    _spectrum_panel(fig.add_subplot(gs[0]), result,
                    "Spectrum (log x)", log=False)
    _spectrum_panel(fig.add_subplot(gs[1]), result,
                    "Spectrum (log x, log y)", log=True)
    if n_panels == 3:
        _residual_panel(fig.add_subplot(gs[2]), result, "Relative residuals")
    return _save_fig(fig, out_path)
