"""Calibration fit figure (D-153).

This module reproduces the calibration figure so the
calibration figure is visually identical: same figure geometry, palette,
logarithmic energy axes with rotated ticks, scale twin axis, raw-MC stairs with
their statistical error bars, covariance bands (scaled for visibility),
reference-energy guides, parameter box and output format inference. Plotting is
per-package on purpose (D-153): the calibration and unfolding figures do not
share a style layer.
"""

from __future__ import annotations
from kc761tool.errors import UsageError
from kc761tool.core.model import (
    PARAM_NAMES_REPORTED,
    InternalCalibration,
    energy_kev,
    internal_jacobian,
    resolution_sigma_grad,
    resolution_sigma_kev,
)
from kc761tool.calib.types import FitResult
from kc761tool.calib.scaling import scale_curve
from kc761tool.calib.model import DatasetDetail
from scipy import optimize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path

import matplotlib

# Must run before pyplot is imported; 3.11+ ignores a use() after it.
matplotlib.use("Agg", force=True)


# Palette: colors of the plotted artists, grouped per panel (values).
_COLOR_DATA = "blue"  # experimental counts (uncertainty bars)
_COLOR_FIT = "red"  # best-fit folded simulation
_COLOR_SIM_RAW = "dimgray"  # scaled raw simulation (stairs)
_COLOR_SCALE = "seagreen"  # scale curve (twin axis)
_COLOR_RESIDUAL_POINTS = "darkgoldenrod"  # residual pull points
_COLOR_RESIDUAL_ZERO = "black"  # residual zero line
_COLOR_RESIDUAL_LEVEL = "red"  # residual +/- level guides
_COLOR_REF_LINE = "dimgray"  # reference-energy guide dashes
_COLOR_CALIB = "darkgreen"  # calibration curve and band
_COLOR_RESOL = "darkolivegreen"  # resolution curve and band
_COLOR_PARAM_BOX = "white"  # parameter box background
_COLOR_PARAM_EDGE = "gray"  # parameter box edge

# Fixed residual range so the panels of all datasets are comparable.
_RESIDUAL_MAX = 0.6

# Reference gamma lines (keV) marked on the calibration and resolution curves.
_REF_LINE_ENERGIES = (59.54, 661.66, 2614.51)

# Calibration/resolution band widths (the 1-sigma bands are scaled up by these
# factors for visibility; the labels state the scaling).
_CALIB_BAND_SCALE = 30.0
_RESOL_BAND_SCALE = 10.0

# Resolution panel geometry: y axis is FWHM/E when True (converted via the
# Gaussian factor) and sigma/E otherwise; ylim top in sigma/E percent.
_RESOL_AS_FWHM = True
_RESOL_YMAX_SIGMA = 10.0

_LOG_X_LABELROTATION = 45.0


def _save_fig(fig, out_plot: str | Path, force: bool) -> Path:
    """Save the figure; refuse overwrite unless ``force``."""
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
    fig.savefig(out, bbox_inches="tight", pad_inches=1.0)
    plt.close(fig)
    return out


def _cap(s: str) -> str:
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def _figure_grid(n_datasets: int):
    n_rows = n_datasets + 2
    fig = plt.figure(figsize=(15.0, 3.5 * n_datasets + 6.5))
    gs = fig.add_gridspec(
        n_rows,
        1,
        height_ratios=[0.0] + [3.5] * n_datasets + [6.5],
        hspace=0.7,
    )
    return fig, gs


def _dataset_row(fig, gs, row: int):
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row], wspace=0.3)
    return (fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1]))


def _footer_row(fig, gs, row: int):
    inner = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[row], wspace=0.3, width_ratios=[1.15, 1.15, 0.9]
    )
    return (
        fig.add_subplot(inner[0, 0]),
        fig.add_subplot(inner[0, 1]),
        fig.add_subplot(inner[0, 2]),
    )


def _title_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes,
            ha="center", va="center", fontsize=16)


def _parameter_text(result: FitResult) -> str:
    def asym(v, lo, hi):
        v_s = f"{v:.6g}"
        lo_s = f"{lo:.3g}"
        hi_s = f"{hi:.3g}"
        if lo_s == hi_s:
            return f"{v_s}"r"$\,\pm\,$"f"{hi_s}"
        return f"{v_s}$^\\text{{+{hi_s}}}_\\text{{-{lo_s}}}$"

    def rows(names, vals, los, his):
        return "\n".join(
            f"{n} = {asym(v, lo, hi)}" for n, v, lo, hi in zip(names, vals, los, his, strict=True)
        )

    covariance = np.asarray(result.param_cov, dtype=np.float64)
    coeffs = np.asarray(result.params_reported, dtype=np.float64)
    coeff_err = np.sqrt(np.maximum(np.diag(covariance)[:4], 0.0))
    transform = internal_jacobian(channel_max=result.channel_max)[:4, :4]
    inverse = np.linalg.inv(transform)
    internal_cov = inverse @ covariance[:4, :4] @ inverse.T
    slopes = np.asarray(result.core_internal, dtype=np.float64)[1:4]
    slope_err = np.sqrt(np.maximum(np.diag(internal_cov)[1:4], 0.0))
    resol = np.asarray(result.resol_params, dtype=np.float64)
    resol_err = np.sqrt(np.maximum(np.diag(covariance)[4:], 0.0))
    return "\n".join(
        [
            "=== Calibration coefficients ===",
            rows(PARAM_NAMES_REPORTED[:4], coeffs, coeff_err, coeff_err),
            "==== Calibration slopes ====",
            rows(("k1", "k2", "k3"), slopes, slope_err, slope_err),
            "=== Resolution parameters ===",
            rows(PARAM_NAMES_REPORTED[4:], resol, resol_err, resol_err),
        ]
    )


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        txt,
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "fc": _COLOR_PARAM_BOX,
            "ec": _COLOR_PARAM_EDGE,
            "alpha": 0.9,
        },
    )


def _log_x_axis(ax) -> None:
    ax.set_xscale("log")
    ax.tick_params(
        axis="x", which="both", labelrotation=_LOG_X_LABELROTATION, labelsize=9
    )


def _positive_start(edges: np.ndarray) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), 0.0, side="right"))


def _spectrum_panel(
    ax, ds: DatasetDetail, calibration: InternalCalibration, channel_max: float, title
) -> None:
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    ch_ref = np.linspace(ds.channel_low, ds.channel_high, 400)
    e_ref = energy_kev(ch_ref, calibration, channel_max=channel_max)
    pos = e_ref > 0.0
    e_curve = np.geomspace(e_ref[pos][0], e_ref[pos][-1], 300)
    ch_curve = np.interp(e_curve, e_ref[pos], ch_ref[pos])
    scale_e = scale_curve(ds.scale_params, ch_curve,
                          ds.channel_low, ds.channel_high)
    (line_scale,) = ax2.plot(
        e_curve, scale_e, "--", color=_COLOR_SCALE, lw=0.5, zorder=1, label="Scale s(ch)"
    )
    ax2.set_ylabel("Scale s(ch)")

    lo = _positive_start(ds.energy_edges_kev)
    e_lo = float(ds.energy_edges_kev[lo])
    scale_centers = scale_curve(
        ds.scale_params, ds.channel_centers, ds.channel_low, ds.channel_high
    )
    stairs_handle = ax.stairs(
        scale_centers[lo:] * ds.raw_mc_counts[lo:],
        ds.energy_edges_kev[lo:],
        lw=0.8,
        color=_COLOR_SIM_RAW,
        zorder=2,
    )
    mc_centers = ds.energy_centers_kev[lo:]
    mc_unc_handle = ax.errorbar(
        mc_centers,
        scale_centers[lo:] * ds.raw_mc_counts[lo:],
        yerr=scale_centers[lo:] * ds.raw_mc_uncertainties[lo:],
        fmt="none",
        ecolor=_COLOR_SIM_RAW,
        elinewidth=0.8,
        capsize=0,
        zorder=2.5,
    )
    m = ds.energy_centers_kev >= e_lo
    data_handle = ax.errorbar(
        ds.energy_centers_kev[m],
        ds.data_counts[m],
        yerr=ds.data_display_sigma[m],
        fmt="o",
        ms=1.5,
        lw=0.8,
        color=_COLOR_DATA,
        zorder=3,
        label="Data (bkg-subtracted)",
    )
    band_handle = ax.fill_between(
        ds.energy_centers_kev[m],
        (ds.prediction - ds.model_mc_sigma)[m],
        (ds.prediction + ds.model_mc_sigma)[m],
        color=_COLOR_FIT,
        alpha=0.18,
        linewidth=0,
        zorder=3.5,
    )
    (line_fit,) = ax.plot(
        ds.energy_centers_kev[m],
        ds.prediction[m],
        "-",
        lw=1.5,
        color=_COLOR_FIT,
        alpha=0.8,
        zorder=4,
    )
    ax.set_yscale("log")
    ax2.set_xscale("log")
    _log_x_axis(ax)
    ax.set_xlim(e_lo, ds.energy_edges_kev[-1])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.legend(
        [data_handle, (line_fit, band_handle),
         (stairs_handle, mc_unc_handle), line_scale],
        [
            "Data (bkg-subtracted)",
            "Best fit (folded MC)",
            "Raw MC (scaled)",
            "Scale s(ch)",
        ],
        fontsize=8,
        loc="lower left",
    )
    if title is not None:
        ax.set_title(title, fontsize=9)


def _residual_panel(
    ax, bin_centers, data_counts, fit_sigma, model_prediction, energy_low, energy_high, title
) -> None:
    m = bin_centers >= energy_low
    bin_centers = bin_centers[m]
    data_counts = data_counts[m]
    fit_sigma = fit_sigma[m]
    model_prediction = model_prediction[m]
    ok = model_prediction > 0
    rel = (data_counts[ok] - model_prediction[ok]) / model_prediction[ok]
    ax.errorbar(
        bin_centers[ok],
        rel,
        yerr=fit_sigma[ok] / model_prediction[ok],
        fmt="o",
        ms=1.5,
        lw=0.8,
        color=_COLOR_RESIDUAL_POINTS,
        alpha=0.8,
    )
    ax.axhline(0, color=_COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=_COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    _log_x_axis(ax)
    ax.set_xlim(energy_low, energy_high)
    ax.set_ylim(-_RESIDUAL_MAX, _RESIDUAL_MAX)
    ax.set_title(title, fontsize=9)


def _mark_energy_line(ax, x: float, y: float, *, hline: bool) -> None:
    ax.plot([x, x], [ax.get_ylim()[0], y], ":",
            color=_COLOR_REF_LINE, lw=1.0, zorder=1.5)
    if hline:
        ax.plot(
            [ax.get_xlim()[0], x], [y, y], ":", color=_COLOR_REF_LINE, lw=1.0, zorder=1.5
        )


def _calibration_panel(
    ax, result: FitResult, calibration: InternalCalibration, title: str = "Energy calibration"
) -> None:
    channel = np.linspace(0.0, result.channel_max, 400)
    energy = energy_kev(channel, calibration, channel_max=result.channel_max)
    basis = np.stack([np.ones_like(channel), channel,
                     channel**2, channel**3], axis=1)
    covariance = np.asarray(result.param_cov, dtype=float)[:4, :4]
    err = _CALIB_BAND_SCALE * np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", basis, covariance, basis), 0.0)
    )
    (line_handle,) = ax.plot(channel, energy, "-", color=_COLOR_CALIB, lw=1.5)
    band_handle = ax.fill_between(
        channel, energy - err, energy + err, color=_COLOR_CALIB, alpha=0.3, lw=0
    )
    y_lo = float(np.min(energy - err))
    y_hi = float(np.max(energy + err))
    y_pad = 0.05 * (y_hi - y_lo)
    ax.set_xlim(-0.05 * result.channel_max, 1.05 * result.channel_max)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.set_autoscale_on(False)
    for e_ref in _REF_LINE_ENERGIES:
        if not (energy[0] < e_ref < energy[-1]):
            continue

        def _cross(channel_value: float, target: float = e_ref) -> float:
            mapped = energy_kev(
                np.array([channel_value]), calibration, channel_max=result.channel_max
            )[0]
            return float(mapped) - target

        ch_ref = float(optimize.brentq(_cross, 0.0, result.channel_max))
        _mark_energy_line(ax, ch_ref, e_ref, hline=True)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    calib_label = "$E(ch) = c_0 + c_1 ch + c_2 ch^2 + c_3 ch^3$"
    if band_handle is not None:
        calib_label += (
            f" (1$\\sigma$ band $\\mathbf{{\\times {_CALIB_BAND_SCALE:g}}}$)"
        )
    ax.legend([(line_handle, band_handle)], [
              calib_label], fontsize=8, loc="upper left")


def _resolution_panel(
    ax, result: FitResult, energy_max: float, title: str = "Energy resolution"
) -> None:
    resol_params = np.asarray(result.resol_params, dtype=float)
    energy = np.linspace(1.0, energy_max, 300)
    coeff = 2 * np.sqrt(2 * np.log(2)) if _RESOL_AS_FWHM else 1.0
    sigma = resolution_sigma_kev(energy, resol_params)
    _, gradient = resolution_sigma_grad(energy, resol_params)
    relative = 100.0 * sigma / energy
    (line_handle,) = ax.plot(energy, coeff *
                             relative, "-", color=_COLOR_RESOL, lw=1.5)
    covariance = np.asarray(result.param_cov, dtype=float)[4:, 4:]
    variance = np.maximum(
        np.einsum("ij,jk,ik->i", gradient.T, covariance, gradient.T), 0.0)
    err = _RESOL_BAND_SCALE * 100.0 * np.sqrt(variance) / energy
    band_handle = ax.fill_between(
        energy,
        coeff * (relative - err),
        coeff * (relative + err),
        color=_COLOR_RESOL,
        alpha=0.3,
        lw=0,
    )
    x_pad = 0.05 * (energy_max - energy[0])
    ax.set_xlim(energy[0] - x_pad, energy_max + x_pad)
    ax.set_ylim(0.0, coeff * _RESOL_YMAX_SIGMA)
    ax.set_autoscale_on(False)
    for e_ref in _REF_LINE_ENERGIES:
        if not (energy[0] <= e_ref <= energy_max):
            continue
        y_ref = (
            coeff
            * 100.0
            * float(resolution_sigma_kev(np.array([e_ref]), resol_params)[0])
            / e_ref
        )
        _mark_energy_line(ax, e_ref, y_ref, hline=True)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(
        f"Energy resolution ({"FWHM" if _RESOL_AS_FWHM else r"$\\sigma$"}, %)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    resol_label = (
        r"$\text{FWHM}(E)\,/\,E$" if _RESOL_AS_FWHM else r"$\sigma(E)\,/\,E$"
    )
    if band_handle is not None:
        resol_label += f" (1$\\sigma$ band $\\mathbf{{\\times {_RESOL_BAND_SCALE:g}}}$)"
    ax.legend([(line_handle, band_handle)], [
              resol_label], fontsize=8, loc="upper right")


def plot_fit(
    result: FitResult,
    details: tuple[DatasetDetail, ...],
    *,
    path: str | Path,
    force: bool = False,
) -> Path:
    """Render the fit result into ``path`` (D-153)."""
    calibration = InternalCalibration.from_array(
        np.asarray(result.core_internal)[:4])
    n = len(details)
    fig, gs = _figure_grid(n)
    note = f"({n} datasets)" if n > 1 else ""
    reduced = result.chi2 / result.dof if result.dof > 0 else float("nan")
    _title_panel(
        fig.add_subplot(gs[0]),
        f"KC761 calibration  |  $\\chi^2/\\mathrm{{ndof}} = {result.chi2:.1f}/{result.dof} "
        f"= {reduced:.2f}$" + (f"  {note}" if note else ""),
    )
    for i, ds in enumerate(details):
        ax_spec, ax_pull = _dataset_row(fig, gs, i + 1)
        label = _cap(ds.label)
        spec_title = (
            f"{label}  [ch {ds.channel_low} - {ds.channel_high}]  "
            f"$\\chi^2 = {ds.chi2:.1f}$, {ds.data_counts.size} bins"
        )
        _spectrum_panel(ax_spec, ds, calibration,
                        result.channel_max, spec_title)
        lo = _positive_start(ds.energy_edges_kev)
        _residual_panel(
            ax_pull,
            ds.energy_centers_kev,
            ds.data_counts,
            ds.fit_sigma,
            ds.prediction,
            float(ds.energy_edges_kev[lo]),
            float(ds.energy_edges_kev[-1]),
            f"{label} residual",
        )
    cal_title = "Energy calibration" + (" (global)" if n > 1 else "")
    res_title = "Energy resolution" + (" (global)" if n > 1 else "")
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    energy_max = float(
        energy_kev(
            np.array([result.channel_max]),
            calibration,
            channel_max=result.channel_max,
        )[0]
    )
    _calibration_panel(ax_cal, result, calibration, cal_title)
    _resolution_panel(ax_res, result, energy_max, res_title)
    _parameter_panel(ax_params, _parameter_text(result))
    return _save_fig(fig, path, force)


__all__ = ["plot_fit"]
