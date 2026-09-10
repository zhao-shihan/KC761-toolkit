"""Unfold figure on the shared plotting style (W4, D-70).

Two panels for the full unfold: the unfolded primary spectrum with the strict
statistical/systematic bands, and the measured channel window against the
refolded prediction with a ratio panel. ``calib_only`` draws the relabeled
spectrum alone. The figure only draws the F-SOLVE/F-UNC/F-UNF results; no
numerics live here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

import kc761.plotting as kp
from kc761.errors import UsageError
from kc761.schema.products import UNFOLD_MODE_CALIB_ONLY
from kc761.unfold.types import UnfoldResult


def _spectrum_panel(ax, result: UnfoldResult) -> None:
    assert result.unfolded is not None
    axis = result.unfolded.axis
    energy = axis.centers()
    values = np.asarray(result.unfolded.values, dtype=np.float64)
    total = np.asarray(result.sigma_total.values, dtype=np.float64)
    stat = np.asarray(result.sigma_statistical.values, dtype=np.float64)
    syst = np.asarray(result.sigma_systematic.values, dtype=np.float64)
    ax.fill_between(energy, values - total, values + total, color=kp.COLOR_FIT, alpha=0.18, lw=0)
    ax.errorbar(
        energy,
        values,
        yerr=total,
        fmt="o",
        ms=1.8,
        lw=0.8,
        color=kp.COLOR_DATA,
        label="unfolded",
        zorder=3,
    )
    ax.plot(energy, values, "-", color=kp.COLOR_FIT, lw=1.2)
    ax.plot(energy, stat, ":", color=kp.COLOR_CALIB, lw=0.9, label="stat")
    ax.plot(energy, syst, "--", color=kp.COLOR_SCALE, lw=0.9, label="syst")
    ax.set_xlabel("Primary energy (keV)")
    ax.set_ylabel("Unfolded yield")
    ax.legend(fontsize=7)
    kp.style_axes(ax)


def _refolded_panels(fig, spec, result: UnfoldResult) -> None:
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[3.0, 1.2], hspace=0.12)
    top = fig.add_subplot(inner[0])
    bottom = fig.add_subplot(inner[1], sharex=top)
    assert result.data_window is not None
    assert result.refolded is not None
    axis = result.data_window.axis
    channel = axis.centers()
    data = np.asarray(result.data_window.values, dtype=np.float64)
    variances = np.asarray(result.data_window.variances, dtype=np.float64)
    data_sigma = np.sqrt(np.maximum(variances, 0.0))
    model = np.asarray(result.refolded.values, dtype=np.float64)
    top.errorbar(
        channel, data, yerr=data_sigma, fmt="o", ms=1.8, lw=0.8, color=kp.COLOR_DATA, label="data"
    )
    top.plot(channel, model, "-", color=kp.COLOR_FIT, lw=1.4, label="refolded")
    top.set_ylabel("Channel counts")
    top.legend(fontsize=7)
    kp.style_axes(top)
    mask = model > 0.0
    ratio = np.zeros_like(data)
    ratio[mask] = (data[mask] - model[mask]) / model[mask]
    ratio_sigma = np.zeros_like(data)
    ratio_sigma[mask] = data_sigma[mask] / model[mask]
    bottom.errorbar(
        channel[mask],
        ratio[mask],
        yerr=ratio_sigma[mask],
        fmt="o",
        ms=1.8,
        lw=0.8,
        color=kp.COLOR_RESIDUAL_POINTS,
    )
    bottom.axhline(0.0, color=kp.COLOR_RESIDUAL_ZERO, lw=0.8)
    bottom.set_ylim(-kp.RESIDUAL_MAX, kp.RESIDUAL_MAX)
    bottom.set_xlabel("Channel")
    bottom.set_ylabel("(data - model)/model")
    kp.style_axes(bottom, grid=False)


def plot_unfold(result: UnfoldResult, *, path: str | Path, force: bool = False) -> Path:
    """Render the unfold result into ``path`` and return the final path."""
    if result.unfolded is None:
        raise UsageError("nothing to plot: the unfold result carries no spectrum")
    kp.apply_style()
    if result.mode == UNFOLD_MODE_CALIB_ONLY:
        fig = plt.figure(figsize=(11.0, 4.2))
        ax = fig.add_subplot(1, 1, 1)
        axis = result.unfolded.axis
        energy = axis.centers()
        values = np.asarray(result.unfolded.values, dtype=np.float64)
        variances = result.unfolded.variances
        sigma = (
            np.zeros_like(values)
            if variances is None
            else np.sqrt(np.maximum(np.asarray(variances, dtype=np.float64), 0.0))
        )
        ax.errorbar(
            energy, values, yerr=sigma, fmt="o", ms=1.8, lw=0.8, color=kp.COLOR_DATA
        )
        ax.plot(energy, values, "-", color=kp.COLOR_FIT, lw=1.0)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        kp.style_axes(ax)
        return kp.save_figure(fig, path, force=force)

    fig = plt.figure(figsize=(11.0, 7.6))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.5, 3.0, 3.6], hspace=0.35)
    title_ax = fig.add_subplot(grid[0])
    title_ax.axis("off")
    assert result.settings is not None
    assert result.window is not None
    reduced = result.chi2 / result.dof if result.dof > 0 else float("nan")
    title_ax.text(
        0.5,
        0.5,
        f"KC761 unfold  |  alpha = {result.settings.alpha!r}  |  "
        f"chi2/dof = {result.chi2:.1f}/{result.dof} = {reduced:.3f}  |  "
        f"window [{result.window.energy_low_kev:.0f}, {result.window.energy_high_kev:.0f}] keV",
        transform=title_ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )
    _spectrum_panel(fig.add_subplot(grid[1]), result)
    _refolded_panels(fig, grid[2], result)
    return kp.save_figure(fig, path, force=force)


__all__ = ["plot_unfold"]
