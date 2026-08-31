"""PDF figures of the fit result."""

from __future__ import annotations
from .response import (PARAM_NAMES_B, PARAM_NAMES_C, PARAM_NAMES_K, calib_model,
                       poly_basis, resol_model, reported_calib)
from .scaling import scale_model
from .fitparamspace import CALIB_K
from pathlib import Path
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")


def _save_fig(fig, out_pdf: str) -> None:
    out = Path(out_pdf)
    if not out.suffix.lower().endswith(".pdf"):
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight")
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
                                    wspace=0.45)
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
    c, err, _ = reported_calib(result.calib_params, result.calib_cov,
                               result.detail.channel_max)
    return ("\n".join([
        rows(PARAM_NAMES_C, c, err),
        rows(PARAM_NAMES_K, result.calib_params[CALIB_K],
             result.calib_errors[CALIB_K]),
        rows(PARAM_NAMES_B, result.resol_params, result.resol_errors),
    ]))


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray", alpha=0.9))


def _spectrum_panel(ax, ds, title: str | None) -> None:
    ax2 = ax.twinx()
    # Draw the scale curve behind the spectrum artists (and the legend): the
    # twin axis sits below the primary axis, whose background is transparent so
    # the curve stays visible.  Layer order (bottom->top): Scale, Raw sim,
    # Data, Best-fit, Legend.
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    e_curve = np.linspace(ds.elow, ds.ehigh, 300)
    line_scale, = ax2.plot(e_curve,
                           scale_model(ds.scale_params, e_curve, ds.elow,
                                       ds.ehigh), "--",
                           color="tab:olive", lw=1, zorder=1,
                           label="Scale s(E)")
    ax2.set_ylabel("Scale s(E)")

    centers_full = 0.5 * (ds.bin_edges[:-1] + ds.bin_edges[1:])
    sb_full = scale_model(ds.scale_params, centers_full, ds.elow, ds.ehigh)
    stairs_handle = ax.stairs(sb_full * ds.raw_sim, ds.bin_edges,
                              color="tab:gray", lw=0.8, zorder=2,
                              label="Raw sim. (scaled)")
    data_handle = ax.errorbar(ds.bin_centers, ds.bin_counts, yerr=ds.sigma,
                              fmt="o", ms=1.5, lw=0.8, alpha=0.6,
                              color="tab:blue", zorder=3,
                              label="Data (-bkg, calibrated)")
    line_fit, = ax.plot(ds.bin_centers, ds.smeared_model, "-", color="tab:red",
                        lw=1.5, zorder=4,
                        label="Best fit (smeared sim.)")
    ax.set_yscale("log")
    ax.set_xlim(ds.elow, ds.ehigh)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")

    # Legend order (top->bottom): Data, Best-fit, Raw sim., Scale.
    ax.legend([data_handle, line_fit, stairs_handle, line_scale],
              ["Data (-bkg, calibrated)",
               "Best fit (smeared sim.)",
               "Raw sim. (scaled)",
               "Scale s(E)"],
              fontsize=8, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=9)


RESID_YMAX = 0.6


def _residual_panel(ax, bin_centers, bin_counts, sigma, smeared_model, elow,
                    ehigh, title: str) -> None:
    ok = smeared_model > 0
    rel = (bin_counts[ok] - smeared_model[ok]) / smeared_model[ok]
    ax.errorbar(bin_centers[ok], rel, yerr=sigma[ok] / smeared_model[ok],
                fmt="o", ms=1.5, lw=0.8, color="tab:brown")
    ax.axhline(0, color="k", lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color="tab:red", lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Relative residual $(d-m)/m$")
    ax.set_xlim(elow, ehigh)
    # Fixed range so that the panels of all datasets are directly comparable.
    ax.set_ylim(-RESID_YMAX, RESID_YMAX)
    ax.set_title(title, fontsize=9)


CALIB_BAND_SCALE = 100.0
RESOL_BAND_SCALE = 30.0


def _calibration_panel(ax, calib, x_max: float = 2048.0,
                       calib_cov=None, title: str = "Energy calibration") -> None:
    x = np.linspace(0.0, x_max, 400)
    e = calib_model(calib, x, x_max)
    ax.plot(x, e, "-", color="tab:purple", lw=1.5,
            label="$E(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$")
    if calib_cov is not None and np.all(np.isfinite(calib_cov)):
        v = poly_basis(x, 3)
        err = CALIB_BAND_SCALE * np.sqrt(
            np.clip(np.sum((v @ calib_cov) * v, axis=1), 0, None))
        ax.fill_between(x, e - err, e + err, color="tab:purple", alpha=0.15,
                        lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                        f"{CALIB_BAND_SCALE:g}}}$)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


RESOL_YMAX = 10.0


def _resolution_panel(ax, b, e_max: float, resol_cov=None,
                      title: str = "Energy resolution") -> None:
    e_res = np.linspace(1.0, e_max, 300)
    rel = 100.0 * resol_model(b, e_res) / e_res
    ax.plot(e_res, rel, "-", color="tab:orange", lw=1.5,
            label=r"$r(E) = \sqrt{b_0^2 + b_1^2 E + b_2^2 E^2}\,/\,E$")
    if resol_cov is not None and np.all(np.isfinite(resol_cov)):
        # Delta method on r=sigma/E with sigma^2 = b0^2 + b1^2 E + b2^2 E^2:
        # d r / d b_i = b_i [1, E, E^2]_i / (E sigma).
        b = np.asarray(b, dtype=float)
        sigma = resol_model(b, e_res)
        grad = (poly_basis(e_res, 2) * b) / (e_res * sigma)[:, None]
        err = RESOL_BAND_SCALE * 100.0 * np.sqrt(
            np.clip(np.sum((grad @ resol_cov) * grad, axis=1), 0, None))
        ax.fill_between(e_res, rel - err, rel + err, color="tab:orange",
                        alpha=0.15, lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                        f"{RESOL_BAND_SCALE:g}}}$)")
    ax.set_ylim(0.0, RESOL_YMAX)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$\sigma/E$ (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_fit(result, out_pdf: str) -> None:
    det = result.detail
    calib = result.calib_params
    b = result.resol_params
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
        if n > 1:
            spec_title = (f"{label}  [{ds.elow:g}-{ds.ehigh:g} keV]  "
                          f"$\\chi^2 = {ds.chi2:.1f}$, {ds.n_bins} bins")
            pull_title = f"{label} relative residual"
        else:
            spec_title = None
            pull_title = "Relative residual"
        _spectrum_panel(ax_spec, ds, spec_title)
        _residual_panel(ax_pull, ds.bin_centers, ds.bin_counts, ds.sigma,
                        ds.smeared_model, ds.elow, ds.ehigh, pull_title)

    cal_title = "Energy calibration" + (" (global)" if n > 1 else "")
    res_title = "Energy resolution" + (" (global)" if n > 1 else "")
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    _, _, calib_cov_report = reported_calib(calib, result.calib_cov,
                                            det.channel_max)
    _calibration_panel(ax_cal, calib, x_max=det.channel_max,
                       calib_cov=calib_cov_report, title=cal_title)
    _resolution_panel(ax_res, b, e_max=float(calib_model(calib, det.channel_max,
                                                         det.channel_max)),
                      resol_cov=result.resol_cov, title=res_title)
    _parameter_panel(ax_params, _parameter_text(result))

    _save_fig(fig, out_pdf)
