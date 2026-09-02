"""PDF figures of the fit result."""

from __future__ import annotations
from .response import (PARAM_NAMES_B, PARAM_NAMES_C, PARAM_NAMES_K,
                       RESOL_E_REF, calib_model, poly_basis, reported_calib,
                       resol_sigma_model)
from .scaling import scale_model
from .fitparamspace import CALIB_K
from .util import bernstein_basis
from pathlib import Path
from scipy import optimize
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

matplotlib.use("Agg")


# Palette: colors of the plotted artists, grouped per panel.
_COLOR_DATA = "blue"  # experimental counts (errorbars)
_COLOR_FIT = "red"  # best-fit smeared simulation
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

# Calibration/resolution band widths (the 1-sigma bands are scaled up by
# these factors for visibility; the labels state the scaling).
_CALIB_BAND_SCALE = 30.0
_RESOL_BAND_SCALE = 10.0

# Resolution panel geometry: y axis is FWHM/E when True (converted via the
# Gaussian factor) and sigma/E otherwise; ylim top in sigma/E percent.
_RESOL_AS_FWHM = True
_RESOL_YMAX_SIGMA = 10.0


def _save_fig(fig, out_pdf: str) -> None:
    for ax in fig.axes:
        ax.tick_params(direction="in", which="both")
    out = Path(out_pdf)
    if not out.suffix.lower().endswith(".pdf"):
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=1.0)
    plt.close(fig)


def _cap(s: str) -> str:
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def _figure_grid(n_datasets: int):
    n_rows = n_datasets + 2
    fig = plt.figure(figsize=(15.0, 3.5 * n_datasets + 5.0))
    gs = fig.add_gridspec(
        n_rows, 1,
        height_ratios=[0.5] + [3.0] * n_datasets + [5.0],
        hspace=0.5,
    )
    return fig, gs


def _dataset_row(fig, gs, row: int):
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row],
                                    wspace=0.3)
    return (fig.add_subplot(inner[0, 0]),
            fig.add_subplot(inner[0, 1]))


def _footer_row(fig, gs, row: int):
    inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row],
                                    wspace=0.3,
                                    width_ratios=[1.15, 1.15, 0.9])
    return (fig.add_subplot(inner[0, 0]),
            fig.add_subplot(inner[0, 1]),
            fig.add_subplot(inner[0, 2]))


def _title_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=11)


def _parameter_text(result) -> str:
    def rows(names, vals, errs): return "\n".join(
        f"{n} = {v: .6g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(names, vals, errs))
    calib = result.calib_params
    calib_err = result.calib_errors
    coeffs, coeff_errors, _ = reported_calib(
        calib, result.calib_cov, result.detail.channel_max)
    resol = result.resol_params
    resol_err = result.resol_errors
    return ("\n".join([
        rows(PARAM_NAMES_C, coeffs, coeff_errors),
        rows(PARAM_NAMES_K, calib[CALIB_K], calib_err[CALIB_K]),
        rows(PARAM_NAMES_B, resol, resol_err),
    ]))


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", fc=_COLOR_PARAM_BOX, ec=_COLOR_PARAM_EDGE,
                      alpha=0.9))


def _spectrum_panel(ax, ds, calib, channel_max, title: str | None) -> None:
    ax2 = ax.twinx()
    # Draw the scale curve behind the spectrum artists (and the legend): the
    # twin axis sits below the primary axis, whose background is transparent so
    # the curve stays visible.  Layer order (bottom->top): Scale, Raw sim,
    # Data, Best-fit, Legend.
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # The scale is a function of channel: evaluate it over the channel window
    # and map the channel axis to energy with the calibration for the twin
    # (energy) axis.
    ch_curve = np.linspace(ds.channel_low, ds.channel_high, 300)
    scale_curve = scale_model(ds.scale_params, ch_curve,
                              ds.channel_low, ds.channel_high)
    e_curve = calib_model(calib, ch_curve, channel_max)
    line_scale, = ax2.plot(e_curve, scale_curve, "--",
                           color=_COLOR_SCALE, lw=0.5, zorder=1,
                           label="Scale s(c)")
    ax2.set_ylabel("Scale s(c)")

    ch_full = np.arange(ds.channel_low, ds.channel_high + 1, dtype=float)
    sb_full = scale_model(ds.scale_params, ch_full,
                          ds.channel_low, ds.channel_high)
    stairs_handle = ax.stairs(sb_full * ds.unsmeared_sim, ds.bin_edges,
                              lw=0.8, color=_COLOR_SIM_RAW, zorder=2,
                              label="Raw sim. (scaled)")
    data_handle = ax.errorbar(ds.bin_centers, ds.data_counts, yerr=ds.total_errors,
                              fmt="o", ms=1.5, lw=0.8, color=_COLOR_DATA,
                              alpha=0.6, zorder=3,
                              label="Data (-bkg, calibrated)")
    line_fit, = ax.plot(ds.bin_centers, ds.model_prediction, "-",
                        lw=1.5, color=_COLOR_FIT, alpha=0.8, zorder=4,
                        label="Best fit (smeared sim.)")
    ax.set_yscale("log")
    ax.set_xlim(ds.bin_edges[0], ds.bin_edges[-1])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")

    # Legend order (top->bottom): Data, Best-fit, Raw sim., Scale.
    ax.legend([data_handle, line_fit, stairs_handle, line_scale],
              ["Data (-bkg, calibrated)",
               "Best fit (smeared sim.)",
               "Raw sim. (scaled)",
               "Scale s(c)"],
              fontsize=8, loc="lower left")
    if title is not None:
        ax.set_title(title, fontsize=9)


def _residual_panel(ax, bin_centers, data_counts, total_errors, model_prediction,
                    energy_low, energy_high, title: str) -> None:
    ok = model_prediction > 0
    rel = (data_counts[ok] - model_prediction[ok]) / model_prediction[ok]
    ax.errorbar(bin_centers[ok], rel, yerr=total_errors[ok] / model_prediction[ok],
                fmt="o", ms=1.5, lw=0.8, color=_COLOR_RESIDUAL_POINTS, alpha=0.8)
    ax.axhline(0, color=_COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=_COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    ax.set_xlim(energy_low, energy_high)
    # Fixed range so that the panels of all datasets are directly comparable.
    ax.set_ylim(-_RESIDUAL_MAX, _RESIDUAL_MAX)
    ax.set_title(title, fontsize=9)


def _mark_energy_line(ax, x: float, y: float, *, hline: bool) -> None:
    """Dashed axis guides marking a reference energy on a curve.

    A vertical dashed line drops from the curve point ``(x, y)`` to the
    bottom edge of the panel; ``hline`` adds the horizontal dashed line from
    the y axis to the point.  Both continue to the actual axis edges (the
    padded limits), so they run spine to curve; they sit below the curve
    (``zorder`` under the default Line2D level).  The panel limits must be
    finalized before calling this (see the two panels).
    """
    ax.plot([x, x], [ax.get_ylim()[0], y], ":",
            color=_COLOR_REF_LINE, lw=1.0, zorder=1.5)
    if hline:
        ax.plot([ax.get_xlim()[0], x], [y, y], ":",
                color=_COLOR_REF_LINE, lw=1.0, zorder=1.5)


def _cov_finite_mask(cov) -> np.ndarray:
    """Mask of the identified parameters with a usable covariance block.

    Undetermined parameters appear as all-NaN rows and columns (their
    variance, and their covariance with every other parameter, is NaN);
    the band is drawn from the remaining identified sub-block.
    """
    cov = np.asarray(cov, dtype=float)
    return ~np.all(np.isnan(cov), axis=1)


def _calibration_panel(ax, calib, channel_max: float = 2048.0,
                       calib_cov=None, title: str = "Energy calibration") -> None:
    channel = np.linspace(0.0, channel_max, 400)
    energy = calib_model(calib, channel, channel_max)
    ax.plot(channel, energy, "-", color=_COLOR_CALIB, lw=1.5,
            label="$E(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$")
    err = None
    if calib_cov is not None:
        finite = _cov_finite_mask(calib_cov)
        if finite.any():
            v = poly_basis(channel, 3)[:, finite]
            cov_f = np.asarray(calib_cov, dtype=float)[np.ix_(finite, finite)]
            err = _CALIB_BAND_SCALE * np.sqrt(np.maximum(
                np.sum((v @ cov_f) * v, axis=1), 0.0))
            ax.fill_between(channel, energy - err, energy + err,
                            color=_COLOR_CALIB, alpha=0.3, lw=0,
                            label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                            f"{_CALIB_BAND_SCALE:g}}}$)")
    # Reference lines: E(x) is strictly increasing, so each energy inverts to
    # a unique channel; the guides stop on the curve and run to the axis
    # edges (padded limits, so nothing is cut short at 0).
    y_lo = np.min(energy - err) if err is not None else np.min(energy)
    y_hi = np.max(energy + err) if err is not None else np.max(energy)
    y_pad = 0.05 * (y_hi - y_lo)
    ax.set_xlim(-0.05 * channel_max, 1.05 * channel_max)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.set_autoscale_on(False)
    for e_ref in _REF_LINE_ENERGIES:
        if not (energy[0] < e_ref < energy[-1]):
            continue
        ch_ref = float(optimize.brentq(
            lambda x: float(calib_model(calib, x, channel_max)) - e_ref,
            0.0, channel_max))
        _mark_energy_line(ax, ch_ref, e_ref, hline=True)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def _resolution_panel(ax, resol_params, energy_max: float, resol_cov=None,
                      title: str = "Energy resolution") -> None:
    resol_params = np.asarray(resol_params, dtype=float)
    energy = np.linspace(1.0, energy_max, 300)
    basis = bernstein_basis(energy / RESOL_E_REF, 2)

    coeff = 2*np.sqrt(2*np.log(2)) if _RESOL_AS_FWHM else 1.0
    sigma = resol_sigma_model(resol_params, energy)
    rel = 100.0 * sigma / energy
    ax.plot(energy, coeff * rel, "-", color=_COLOR_RESOL, lw=1.5,
            label=r"$\text{FWHM}(E)\,/\,E$" if _RESOL_AS_FWHM else r"$\sigma(E)\,/\,E$")
    if resol_cov is not None:
        finite = _cov_finite_mask(resol_cov)
        if finite.any():
            grad = basis * resol_params / sigma[:, None]
            grad_f = grad[:, finite]
            cov_f = np.asarray(resol_cov, dtype=float)[np.ix_(finite, finite)]
            var = np.maximum(np.sum((grad_f @ cov_f) * grad_f, axis=1), 0.0)
            err = _RESOL_BAND_SCALE * 100.0 * np.sqrt(var) / energy
            ax.fill_between(energy, coeff * (rel - err), coeff * (rel + err),
                            color=_COLOR_RESOL, alpha=0.3, lw=0,
                            label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                            f"{_RESOL_BAND_SCALE:g}}}$)")
    # Reference lines: guides at the reference energies to the curve points,
    # running to the axis edges (the y axis for the horizontal segments).
    x_pad = 0.05 * (energy_max - energy[0])
    ax.set_xlim(energy[0] - x_pad, energy_max + x_pad)
    ax.set_ylim(0.0, coeff * _RESOL_YMAX_SIGMA)
    ax.set_autoscale_on(False)
    for e_ref in _REF_LINE_ENERGIES:
        if not (energy[0] <= e_ref <= energy_max):
            continue
        y_ref = coeff * 100.0 * resol_sigma_model(resol_params, e_ref) / e_ref
        _mark_energy_line(ax, e_ref, y_ref, hline=True)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Energy resolution (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_fit(result, out_pdf: str) -> None:
    det = result.detail
    calib = result.calib_params
    resol_params = result.resol_params
    n = len(det.datasets)

    fig, gs = _figure_grid(n)
    note = f"({n} datasets)" if n > 1 else ""
    _title_panel(
        fig.add_subplot(gs[0]),
        f"KC761 calibration  |  "
        f"$\\chi^2/\\mathrm{{ndof}} = {result.chi2:.1f}/{result.ndof} "
        f"= {result.reduced_chi2:.2f}$"
        + (f"  {note}" if note else ""))

    for i, ds in enumerate(det.datasets):
        ax_spec, ax_pull = _dataset_row(fig, gs, i + 1)
        label = _cap(ds.label)
        spec_title = (f"{label}  [ch {ds.channel_low} - {ds.channel_high}]  "
                      f"$\\chi^2 = {ds.chi2:.1f}$, {ds.n_bins} bins")
        res_title = f"{label} residual"
        _spectrum_panel(ax_spec, ds, calib, det.channel_max, spec_title)
        _residual_panel(ax_pull, ds.bin_centers, ds.data_counts, ds.total_errors,
                        ds.model_prediction, ds.bin_edges[0], ds.bin_edges[-1],
                        res_title)

    cal_title = "Energy calibration" + (" (global)" if n > 1 else "")
    res_title = "Energy resolution" + (" (global)" if n > 1 else "")
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    _, _, calib_cov_report = reported_calib(calib, result.calib_cov,
                                            det.channel_max)
    _calibration_panel(ax_cal, calib, channel_max=det.channel_max,
                       calib_cov=calib_cov_report, title=cal_title)
    _resolution_panel(ax_res, resol_params,
                      energy_max=float(calib_model(calib, det.channel_max,
                                                   det.channel_max)),
                      resol_cov=result.resol_cov, title=res_title)
    _parameter_panel(ax_params, _parameter_text(result))

    _save_fig(fig, out_pdf)
