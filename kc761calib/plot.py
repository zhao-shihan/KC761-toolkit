"""Figures of the fit result.

The output format is inferred from the file extension (any
matplotlib-supported format); when the extension is missing or
unrecognized, the figure is written as PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy import optimize

from .fitparamspace import CALIB_K
from .response import (PARAM_NAMES_B, PARAM_NAMES_C, PARAM_NAMES_K,
                       RESOL_E_REF, calib_model, poly_basis, reported_calib,
                       resol_sigma_model)
from .scaling import scale_model
from .util import bernstein_basis

matplotlib.use("Agg")


# Palette: colors of the plotted artists, grouped per panel.
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

# Calibration/resolution band widths (the 1-sigma bands are scaled up by
# these factors for visibility; the labels state the scaling).
_CALIB_BAND_SCALE = 30.0
_RESOL_BAND_SCALE = 10.0

# Resolution panel geometry: y axis is FWHM/E when True (converted via the
# Gaussian factor) and sigma/E otherwise; ylim top in sigma/E percent.
_RESOL_AS_FWHM = True
_RESOL_YMAX_SIGMA = 10.0


def _save_fig(fig, out_plot: str) -> Path:
    """Save the figure, inferring the format from the file extension.

    The extension (case-insensitive) is matched against the formats
    matplotlib can write; a missing or unrecognized extension falls back
    to PDF.  Returns the final output path.
    """
    for ax in fig.axes:
        ax.tick_params(direction="in", which="both")
    out = Path(out_plot)
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in FigureCanvasBase.get_supported_filetypes():
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=1.0)
    plt.close(fig)
    return out


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
    def rows(names, vals, uncs): return "\n".join(
        f"{n} = {v: .6g} $\\pm$ {u_: .3g}"
        for n, v, u_ in zip(names, vals, uncs))
    calib = result.calib_params
    calib_unc = result.calib_uncertainties
    coeffs, coeff_uncertainties, _ = reported_calib(
        calib, result.calib_cov, result.detail.channel_max)
    resol = result.resol_params
    resol_unc = result.resol_uncertainties
    return ("\n".join([
        "=== Calibration coefficients ===",
        rows(PARAM_NAMES_C, coeffs, coeff_uncertainties),
        "==== Calibration slopes ====",
        rows(PARAM_NAMES_K, calib[CALIB_K], calib_unc[CALIB_K]),
        "=== Resolution parameters ===",
        rows(PARAM_NAMES_B, resol, resol_unc),
    ]))


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", fc=_COLOR_PARAM_BOX, ec=_COLOR_PARAM_EDGE,
                      alpha=0.9))


def _positive_start(edges: np.ndarray) -> int:
    """First bin index whose lower edge is positive (log-x truncation).

    The calibration maps the lowest channels to energies <= 0 (c0 < 0
    by construction): bins at or below zero cannot be drawn on a
    logarithmic axis, so the spectrum panel and its residual panel are
    truncated at the first bin fully above zero.
    """
    return int(np.searchsorted(np.asarray(edges, dtype=float), 0.0,
                               side="right"))


def _spectrum_panel(ax, ds, calib, channel_max, title: str | None) -> None:
    ax2 = ax.twinx()
    # Draw the scale curve behind the spectrum artists (and the legend): the
    # twin axis sits below the primary axis, whose background is transparent so
    # the curve stays visible.  Layer order (bottom->top): Scale, Raw sim,
    # Raw sim. uncertainty bars, Data, Best-fit band, Best-fit, Legend.
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # The scale is a function of channel: evaluate it over the channel window
    # and map the channel axis to energy with the calibration for the twin
    # (energy) axis.
    ch_curve = np.linspace(ds.channel_low, ds.channel_high, 300)
    scale_curve = scale_model(ds.scale_params, ch_curve,
                              ds.channel_low, ds.channel_high)
    e_curve = calib_model(calib, ch_curve, channel_max)
    # The twin energy axis is logarithmic like the primary axis; the parts of
    # the curve at non-positive energies (the low channels) are dropped here
    # and by the bin truncation below.
    pos = e_curve > 0.0
    line_scale, = ax2.plot(e_curve[pos], scale_curve[pos], "--",
                           color=_COLOR_SCALE, lw=0.5, zorder=1,
                           label="Scale s(ch)")
    ax2.set_ylabel("Scale s(ch)")

    # Truncate at the first bin fully above zero: the calibration maps the
    # lowest channels to energies <= 0, which a log x-axis cannot show.
    lo = _positive_start(ds.bin_edges)
    e_lo = float(ds.bin_edges[lo])
    ch_full = np.arange(ds.channel_low, ds.channel_high + 1, dtype=float)
    sb_full = scale_model(ds.scale_params, ch_full,
                          ds.channel_low, ds.channel_high)
    stairs_handle = ax.stairs(sb_full[lo:] * ds.raw_sim[lo:],
                              ds.bin_edges[lo:],
                              lw=0.8, color=_COLOR_SIM_RAW, zorder=2)
    # MC statistical uncertainty bars of the raw (pre-folding) rebinned
    # sim, scaled by the same scale curve as the stairs, at the midpoints
    # of the energy-deposition bin edges.  No own legend entry: the legend
    # reuses the "Raw sim." handle, overlaid with the uncertainty-bar
    # artist.
    sim_centers = 0.5 * (ds.bin_edges[lo:-1] + ds.bin_edges[lo + 1:])
    sim_unc_handle = ax.errorbar(
        sim_centers, sb_full[lo:] * ds.raw_sim[lo:],
        yerr=sb_full[lo:] * ds.raw_sim_uncertainties[lo:],
        fmt="none", ecolor=_COLOR_SIM_RAW, elinewidth=0.8, capsize=0,
        zorder=2.5)
    # The data-side artists cover the usable bins: keep only the bins at or
    # above the truncation floor (their centers are >= e_lo exactly for the
    # bins fully above zero).
    m = ds.bin_centers >= e_lo
    data_handle = ax.errorbar(ds.bin_centers[m], ds.data_counts[m],
                              yerr=ds.data_uncertainties[m],
                              fmt="o", ms=1.5, lw=0.8, color=_COLOR_DATA,
                              zorder=3, label="Data (bkg-subtracted)")
    # Uncertainty band around the best-fit curve: the Monte Carlo
    # statistical uncertainty of the scaled folded-sim prediction, +/-
    # model_uncertainties.  No own legend entry: the legend reuses the
    # "Best fit" handle, overlaid with the band patch.
    band_handle = ax.fill_between(ds.bin_centers[m],
                                  (ds.model_prediction
                                   - ds.model_uncertainties)[m],
                                  (ds.model_prediction
                                   + ds.model_uncertainties)[m],
                                  color=_COLOR_FIT, alpha=0.18, linewidth=0,
                                  zorder=3.5)
    line_fit, = ax.plot(ds.bin_centers[m], ds.model_prediction[m], "-",
                        lw=1.5, color=_COLOR_FIT, alpha=0.8, zorder=4)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax2.set_xscale("log")
    ax.set_xlim(e_lo, ds.bin_edges[-1])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")

    # Legend order (top->bottom): Data, Best-fit (line + MC stat. band),
    # Raw sim. (line + MC stat. uncertainty bars), Scale.
    ax.legend([data_handle, (line_fit, band_handle),
               (stairs_handle, sim_unc_handle), line_scale],
              ["Data (bkg-subtracted)",
               "Best fit (folded sim.)",
               "Raw sim. (scaled)",
               "Scale s(ch)"],
              fontsize=8, loc="lower left")
    if title is not None:
        ax.set_title(title, fontsize=9)


def _residual_panel(ax, bin_centers, data_counts, combined_uncertainties,
                    model_prediction, energy_low, energy_high,
                    title: str) -> None:
    # Same positive-energy truncation as the spectrum panel so both panels
    # cover the same logarithmic x range (energy_low is the first bin edge
    # above zero).
    m = bin_centers >= energy_low
    bin_centers = bin_centers[m]
    data_counts = data_counts[m]
    combined_uncertainties = combined_uncertainties[m]
    model_prediction = model_prediction[m]
    ok = model_prediction > 0
    rel = (data_counts[ok] - model_prediction[ok]) / model_prediction[ok]
    # The uncertainty bars on the ratio carry the full per-bin sigma of the
    # numerator (data - model): the data's statistical + systematic
    # uncertainty plus the Monte Carlo statistical uncertainty of the
    # scaled folded-sim prediction, divided by the model prediction -- i.e.
    # combined_uncertainties includes the simulated spectrum's
    # finite-statistics uncertainty.
    ax.errorbar(bin_centers[ok], rel,
                yerr=combined_uncertainties[ok] / model_prediction[ok],
                fmt="o", ms=1.5, lw=0.8, color=_COLOR_RESIDUAL_POINTS, alpha=0.8)
    ax.axhline(0, color=_COLOR_RESIDUAL_ZERO, lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color=_COLOR_RESIDUAL_LEVEL, lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    ax.set_xscale("log")
    ax.set_xlim(energy_low, energy_high)
    ax.set_ylim(-_RESIDUAL_MAX, _RESIDUAL_MAX)  # fixed, for comparability
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
    line_handle, = ax.plot(channel, energy, "-", color=_COLOR_CALIB, lw=1.5)
    err = None
    band_handle = None
    if calib_cov is not None:
        finite = _cov_finite_mask(calib_cov)
        if finite.any():
            v = poly_basis(channel, 3)[:, finite]
            cov_f = np.asarray(calib_cov, dtype=float)[np.ix_(finite, finite)]
            err = _CALIB_BAND_SCALE * np.sqrt(np.maximum(
                np.sum((v @ cov_f) * v, axis=1), 0.0))
            band_handle = ax.fill_between(channel, energy - err, energy + err,
                                          color=_COLOR_CALIB, alpha=0.3, lw=0)
    # Reference lines: E(ch) is strictly increasing, so each energy inverts to
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
    # One legend entry combining the fitted curve and its band; the band is
    # the 1-sigma covariance band scaled up for visibility.
    calib_label = "$E(ch) = c_0 + c_1 ch + c_2 ch^2 + c_3 ch^3$"
    if band_handle is not None:
        calib_label += (f" (1$\\sigma$ band "
                        f"$\\mathbf{{\\times {_CALIB_BAND_SCALE:g}}}$)")
    calib_handles = ((line_handle, band_handle) if band_handle is not None
                     else line_handle)
    ax.legend([calib_handles], [calib_label], fontsize=8, loc="upper left")


def _resolution_panel(ax, resol_params, energy_max: float, resol_cov=None,
                      title: str = "Energy resolution") -> None:
    resol_params = np.asarray(resol_params, dtype=float)
    energy = np.linspace(1.0, energy_max, 300)
    basis = bernstein_basis(energy / RESOL_E_REF, 2)

    coeff = 2*np.sqrt(2*np.log(2)) if _RESOL_AS_FWHM else 1.0
    sigma = resol_sigma_model(resol_params, energy)
    rel = 100.0 * sigma / energy
    line_handle, = ax.plot(
        energy, coeff * rel, "-", color=_COLOR_RESOL, lw=1.5)
    band_handle = None
    if resol_cov is not None:
        finite = _cov_finite_mask(resol_cov)
        if finite.any():
            grad = basis * resol_params / sigma[:, None]
            grad_f = grad[:, finite]
            cov_f = np.asarray(resol_cov, dtype=float)[np.ix_(finite, finite)]
            var = np.maximum(np.sum((grad_f @ cov_f) * grad_f, axis=1), 0.0)
            err = _RESOL_BAND_SCALE * 100.0 * np.sqrt(var) / energy
            band_handle = ax.fill_between(energy, coeff * (rel - err),
                                          coeff * (rel + err),
                                          color=_COLOR_RESOL, alpha=0.3, lw=0)
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
    ax.set_ylabel(
        f"Energy resolution ({"FWHM" if _RESOL_AS_FWHM else r"$\sigma$"}, %)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    resol_label = (r"$\text{FWHM}(E)\,/\,E$" if _RESOL_AS_FWHM
                   else r"$\sigma(E)\,/\,E$")
    if band_handle is not None:
        resol_label += (f" (1$\\sigma$ band "
                        f"$\\mathbf{{\\times {_RESOL_BAND_SCALE:g}}}$)")
    resol_handles = ((line_handle, band_handle) if band_handle is not None
                     else line_handle)
    ax.legend([resol_handles], [resol_label], fontsize=8, loc="upper right")


def plot_fit(result, out_plot: str) -> Path:
    """Render the fit result into the output plot file.

    Returns the final output path (the requested path with a PDF fallback
    when the extension is missing or unrecognized), so callers can derive
    sibling output names from it.
    """
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
        _residual_panel(ax_pull, ds.bin_centers, ds.data_counts,
                        ds.combined_uncertainties, ds.model_prediction,
                        float(ds.bin_edges[_positive_start(ds.bin_edges)]),
                        ds.bin_edges[-1], res_title)

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

    return _save_fig(fig, out_plot)
