"""Calibration fit figure (W3, D-70).

Uses the shared :mod:`kc761.plotting` style. Panels: one spectrum/pull row per
dataset, then the calibration curve, the resolution curve and a parameter box.
The chi-square, model and uncertainty definitions are the F-CAL/F-MODEL
formulas; this module only draws them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from numpy.typing import NDArray

import kc761.plotting as kp
from kc761.calib.model import DatasetDetail
from kc761.calib.types import FitResult
from kc761.core.model import (
    PARAM_NAMES_REPORTED,
    RESOL_E_REF_KEV,
    InternalCalibration,
    energy_kev,
)

_EPS = 1e-12


def _positive_start(edges: NDArray[np.float64]) -> int:
    """First bin whose lower edge is positive (log-x truncation)."""
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), 0.0, side="right"))


def _spectrum_panel(ax, detail: DatasetDetail) -> None:
    lo = _positive_start(detail.energy_edges_kev)
    energy = detail.energy_centers_kev[lo:]
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    ax2.plot(energy, detail.scale_curve[lo:], "--", color=kp.COLOR_SCALE, lw=0.6, label="Scale")
    ax2.set_ylabel("Scale s(ch)", color=kp.COLOR_SCALE, fontsize=8)
    ax2.tick_params(axis="y", colors=kp.COLOR_SCALE)
    band_lo = (detail.prediction - detail.model_mc_sigma)[lo:]
    band_hi = (detail.prediction + detail.model_mc_sigma)[lo:]
    ax.fill_between(energy, band_lo, band_hi, color=kp.COLOR_FIT, alpha=0.18, linewidth=0)
    ax.errorbar(
        energy,
        detail.data_counts[lo:],
        yerr=detail.data_display_sigma[lo:],
        fmt="o",
        ms=1.8,
        lw=0.8,
        color=kp.COLOR_DATA,
        label="Data",
        zorder=3,
    )
    ax.plot(energy, detail.prediction[lo:], "-", color=kp.COLOR_FIT, lw=1.4, label="Best fit")
    if np.all(detail.data_counts[lo:] > 0.0):
        ax.set_yscale("log")
    if lo > 0:
        ax.set_xscale("log")
    kp.style_axes(ax)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"{detail.label}  [ch {detail.channel_low}-{detail.channel_high}]  "
        f"chi2 = {detail.chi2:.1f}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower left")


def _residual_panel(ax, detail: DatasetDetail) -> None:
    lo = _positive_start(detail.energy_edges_kev)
    energy = detail.energy_centers_kev[lo:]
    prediction = detail.prediction[lo:]
    data = detail.data_counts[lo:]
    sigma = detail.fit_sigma[lo:]
    mask = prediction > 0.0
    ratio = (data[mask] - prediction[mask]) / prediction[mask]
    ax.errorbar(
        energy[mask],
        ratio,
        yerr=sigma[mask] / prediction[mask],
        fmt="o",
        ms=1.8,
        lw=0.8,
        color=kp.COLOR_RESIDUAL_POINTS,
        alpha=0.8,
    )
    ax.axhline(0.0, color=kp.COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=kp.COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    if lo > 0:
        ax.set_xscale("log")
    kp.style_axes(ax, grid=False)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("(data - fit) / fit")
    ax.set_ylim(-kp.RESIDUAL_MAX, kp.RESIDUAL_MAX)
    ax.set_title(f"{detail.label} residual", fontsize=9)


def _calibration_panel(ax, result: FitResult, energy_max: float) -> None:
    calibration = InternalCalibration.from_array(result.core_internal[:4])
    channel = np.linspace(0.0, result.channel_max, 400)
    energy = energy_kev(channel, calibration, channel_max=result.channel_max)
    covariance = np.asarray(result.param_cov, dtype=np.float64)[:4, :4]
    basis = np.stack([np.ones_like(channel), channel, channel**2, channel**3], axis=1)
    variance = np.maximum(np.einsum("ij,jk,ik->i", basis, covariance, basis), 0.0)
    band = kp.CALIB_BAND_SCALE * np.sqrt(variance)
    ax.plot(channel, energy, "-", color=kp.COLOR_CALIB, lw=1.4, label="E(ch)")
    ax.fill_between(
        channel,
        energy - band,
        energy + band,
        color=kp.COLOR_CALIB,
        alpha=0.3,
        linewidth=0,
        label=f"1sigma x {kp.CALIB_BAND_SCALE:g}",
    )
    kp.style_axes(ax)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title("Energy calibration", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def _resolution_panel(ax, result: FitResult, energy_max: float) -> None:
    b0, b1, b2 = result.resol_params
    covariance = np.asarray(result.param_cov, dtype=np.float64)[4:, 4:]
    energy = np.linspace(1.0, max(energy_max, 1.0), 300)
    t = np.clip(energy, 0.0, np.inf) / RESOL_E_REF_KEV
    basis = np.stack([(1.0 - t) ** 2, 2.0 * (1.0 - t) * t, t**2], axis=1)  # (n, 3)
    variance = basis @ np.array([b0 * b0, b1 * b1, b2 * b2])
    sigma = np.sqrt(np.maximum(variance, 0.0))
    grad = basis * np.array([b0, b1, b2]) / np.maximum(sigma, _EPS)[:, None]
    sigma_variance = np.maximum(np.einsum("ij,jk,ik->i", grad, covariance, grad), 0.0)
    sigma_error = np.sqrt(sigma_variance)
    relative = 100.0 * sigma / energy
    relative_error = kp.RESOL_BAND_SCALE * 100.0 * sigma_error / energy
    ax.plot(energy, relative, "-", color=kp.COLOR_RESOL, lw=1.4, label="sigma(E)/E")
    ax.fill_between(
        energy,
        relative - relative_error,
        relative + relative_error,
        color=kp.COLOR_RESOL,
        alpha=0.3,
        linewidth=0,
        label=f"1sigma x {kp.RESOL_BAND_SCALE:g}",
    )
    kp.style_axes(ax)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("sigma(E) / E (%)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Energy resolution", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def _parameter_panel(ax, result: FitResult) -> None:
    errors = np.sqrt(np.maximum(np.diag(np.asarray(result.param_cov, dtype=np.float64)), 0.0))
    values = (*result.params_reported, *result.resol_params)
    lines = [
        f"{name} = {value: .6g} +- {error:.3g}"
        for name, value, error in zip(PARAM_NAMES_REPORTED, values, errors, strict=True)
    ]
    for scale in result.scales:
        s0, s1, s2, s3 = scale.params
        lines.append(f"{scale.label}: s = {s1:.4g}, {s2:.4g}, {s3:.4g} (s0={s0:.4g})")
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={
            "boxstyle": "round",
            "fc": kp.COLOR_PARAM_BOX,
            "ec": kp.COLOR_PARAM_EDGE,
            "alpha": 0.9,
        },
    )


def plot_fit(
    result: FitResult,
    details: tuple[DatasetDetail, ...],
    *,
    path: str | Path,
    force: bool = False,
) -> Path:
    """Render the fit into ``path`` and return the final path (D-70)."""
    kp.apply_style()
    n = len(details)
    height = 3.4 * n + 5.0
    fig = plt.figure(figsize=(13.0, height))
    grid = fig.add_gridspec(
        n + 2,
        1,
        height_ratios=[0.5, *([3.4] * n), 4.4],
        hspace=0.6,
    )
    title_ax = fig.add_subplot(grid[0])
    title_ax.axis("off")
    reduced = result.chi2 / result.dof if result.dof > 0 else float("nan")
    title_ax.text(
        0.5,
        0.5,
        f"KC761 calibration  |  chi2/ndof = {result.chi2:.1f} / {result.dof} "
        f"= {reduced:.3f}  |  {result.status}",
        transform=title_ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
    )
    for index, detail in enumerate(details):
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[index + 1], wspace=0.28)
        _spectrum_panel(fig.add_subplot(inner[0, 0]), detail)
        _residual_panel(fig.add_subplot(inner[0, 1]), detail)
    footer = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=grid[n + 1], wspace=0.3, width_ratios=[1.15, 1.15, 0.9]
    )
    calibration = InternalCalibration.from_array(result.core_internal[:4])
    energy_max = float(
        energy_kev(
            np.array([result.channel_max]),
            calibration,
            channel_max=result.channel_max,
        )[0]
    )
    _calibration_panel(fig.add_subplot(footer[0, 0]), result, energy_max)
    _resolution_panel(fig.add_subplot(footer[0, 1]), result, energy_max)
    _parameter_panel(fig.add_subplot(footer[0, 2]), result)
    return kp.save_figure(fig, path, force=force)


__all__ = ["plot_fit"]
