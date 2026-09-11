"""Unfold figure (D-153).

This module reproduces the unfolding figure: three stacked
panels with height ratio 2:2:1 (linear-y spectrum, log-y spectrum, relative
residuals), or the two spectrum panels alone in ``calib_only`` mode. The energy
x-axis is logarithmic with rotated ticks; bins at or below zero energy are
truncated at the first bin ending above zero. Spectrum layers are histograms
(stairs) with two nested uncertainty bands (outer total, inner systematic).
Plotting is per-package on purpose (D-153).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Must run before pyplot is imported; 3.11+ ignores a use() after it.
matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase

from kc761tool.errors import UsageError
from kc761tool.schema.products import UNFOLD_MODE_CALIB_ONLY
from kc761tool.unfold.types import UnfoldResult

# Palette (calibration-figure conventions): colors of the plotted artists.
_COLOR_DATA = "blue"  # calibrated spectrum (histogram + uncertainty bars)
_COLOR_FIT = "red"  # unfolded spectrum (histogram + uncertainty bars)
_COLOR_REFOLD = "dimgray"  # refolded prediction (stairs)
_COLOR_RESIDUAL_POINTS = "darkgoldenrod"  # residual points
_COLOR_RESIDUAL_ZERO = "black"  # residual zero line
_COLOR_RESIDUAL_LEVEL = "red"  # residual +/- level guides

_BAND_ALPHA_TOTAL = 0.15  # outer band: the total uncertainty
_BAND_ALPHA_SYST = 0.30  # inner band: the systematic part
_RESIDUAL_MAX = 0.6
_LOG_X_LABELROTATION = 45.0


def _save_fig(fig, out_plot: str | Path, force: bool) -> Path:
    for ax in fig.axes:
        ax.tick_params(direction="in", which="both")
    out = Path(out_plot)
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in FigureCanvasBase.get_supported_filetypes():
        out = out.with_suffix(".pdf")
    if out.exists() and not force:
        plt.close(fig)
        raise UsageError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def _log_x_axis(ax) -> None:
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="both", labelrotation=_LOG_X_LABELROTATION, labelsize=9)


def _positive_start(edges: np.ndarray) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), 0.0, side="right"))


def _draw_layer(ax, centers, edges, counts, sigma_total, sigma_syst, color, label, ls="-"):
    total_band = ax.fill_between(
        centers,
        np.maximum(counts - sigma_total, 0.0),
        counts + sigma_total,
        step="mid",
        color=color,
        alpha=_BAND_ALPHA_TOTAL,
        linewidth=0,
    )
    syst_band = ax.fill_between(
        centers,
        np.maximum(counts - sigma_syst, 0.0),
        counts + sigma_syst,
        step="mid",
        color=color,
        alpha=_BAND_ALPHA_SYST,
        linewidth=0,
    )
    stairs = ax.stairs(counts, edges, lw=0.8, color=color, ls=ls, alpha=0.8)
    return stairs, total_band, syst_band, label


def _series(result: UnfoldResult):
    """Return the series on the common primary energy axis."""
    assert result.unfolded is not None
    edges = np.asarray(result.unfolded.axis.edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.asarray(result.unfolded.values, dtype=float)
    if result.mode == UNFOLD_MODE_CALIB_ONLY:
        variances = result.unfolded.variances
        sigma = (
            np.zeros_like(counts)
            if variances is None
            else np.sqrt(np.maximum(np.asarray(variances, dtype=float), 0.0))
        )
        return edges, centers, counts, sigma, sigma, counts, sigma, sigma, None
    assert result.sigma_total is not None
    assert result.sigma_statistical is not None
    assert result.sigma_systematic is not None
    assert result.data_window is not None
    assert result.refolded is not None
    total = np.asarray(result.sigma_total.values, dtype=float)
    syst = np.asarray(result.sigma_systematic.values, dtype=float)
    data = np.asarray(result.data_window.values, dtype=float)
    data_var = result.data_window.variances
    data_total = (
        np.zeros_like(data)
        if data_var is None
        else np.sqrt(np.maximum(np.asarray(data_var, dtype=float), 0.0))
    )
    syst_frac = 0.0 if result.settings is None else float(result.settings.syst_frac)
    data_syst = syst_frac * np.abs(data)
    refolded = np.asarray(result.refolded.values, dtype=float)
    # The reported primary bins and the reported channel rows coincide for the
    # frozen 1:1 axis convention (D-121); on a coarse synthetic fixture they can
    # differ, so the layers are aligned to the common length for the figure.
    n = min(counts.size, data.size, refolded.size)
    edges = edges[: n + 1]
    centers = centers[:n]
    return (
        edges, centers, counts[:n], total[:n], syst[:n],
        data[:n], data_total[:n], data_syst[:n], refolded[:n],
    )


def _spectrum_panel(ax, result: UnfoldResult, title: str, *, log: bool) -> None:
    edges, centers, counts, total, syst, data, data_total, data_syst, refolded = _series(result)
    lo = _positive_start(edges)
    centers = centers[lo:]
    edges = edges[lo:]
    calib_ls = "--" if result.mode != UNFOLD_MODE_CALIB_ONLY else "-"
    calib = _draw_layer(
        ax, centers, edges, data[lo:], data_total[lo:], data_syst[lo:],
        _COLOR_DATA, "Calibrated spectrum", ls=calib_ls,
    )
    refolded_handle = None
    unfolded = None
    if result.mode != UNFOLD_MODE_CALIB_ONLY:
        refolded_handle = ax.stairs(refolded[lo:], edges, ls=":", lw=0.8, color=_COLOR_REFOLD)
        unfolded = _draw_layer(
            ax, centers, edges, counts[lo:], total[lo:], syst[lo:],
            _COLOR_FIT, "Unfolded spectrum",
        )
    if log:
        ax.set_yscale("log")
    _log_x_axis(ax)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)

    handles_out = []
    labels_out = []
    if unfolded is not None:
        handles_out += [(unfolded[0], unfolded[1]), unfolded[2], refolded_handle]
        labels_out += [unfolded[3], "Syst. unc. band", "Refolded spectrum"]
    handles_out += [(calib[0], calib[1]), calib[2]]
    labels_out += [calib[3], "Syst. unc. band"]
    ax.legend(handles_out, labels_out, fontsize=8, loc="upper right", ncol=2)


def _residual_panel(ax, result: UnfoldResult, title: str) -> None:
    edges, centers, _counts, _total, _syst, data, data_total, _data_syst, refolded = _series(result)
    assert refolded is not None
    lo = _positive_start(edges)
    centers = centers[lo:]
    data = data[lo:]
    refolded = refolded[lo:]
    data_total = data_total[lo:]
    ok = refolded > 0.0
    rel = (data[ok] - refolded[ok]) / refolded[ok]
    ax.errorbar(
        centers[ok], rel, yerr=data_total[ok] / refolded[ok],
        fmt="o", ms=1.5, lw=0.8, color=_COLOR_RESIDUAL_POINTS, alpha=0.8,
    )
    ax.axhline(0.0, color=_COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=_COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    _log_x_axis(ax)
    ax.set_xlim(edges[lo], edges[-1])
    ax.set_ylim(-_RESIDUAL_MAX, _RESIDUAL_MAX)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def plot_unfold(result: UnfoldResult, *, path: str | Path, force: bool = False) -> Path:
    """Render the result into ``path`` (D-153)."""
    if result.unfolded is None:
        raise UsageError("nothing to plot: the unfold result carries no spectrum")
    calib_only = result.mode == UNFOLD_MODE_CALIB_ONLY
    n_panels = 2 if calib_only else 3
    fig = plt.figure(figsize=(9.5, 10.5))
    gs = fig.add_gridspec(
        n_panels,
        1,
        height_ratios=[2.0, 2.0, 1.0] if n_panels == 3 else [1.0, 1.0],
        hspace=0.5,
    )
    _spectrum_panel(fig.add_subplot(gs[0]), result, "Spectrum", log=False)
    _spectrum_panel(fig.add_subplot(gs[1]), result, "Spectrum (log y-axis)", log=True)
    if n_panels == 3:
        _residual_panel(fig.add_subplot(gs[2]), result, "Relative residuals")
    return _save_fig(fig, path, force)


__all__ = ["plot_unfold"]
