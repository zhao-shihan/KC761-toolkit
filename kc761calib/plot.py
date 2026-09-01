"""PDF figures of the fit result."""

from __future__ import annotations
from .response import (PARAM_NAMES_B, PARAM_NAMES_C, PARAM_NAMES_K,
                       RESOL_T_SCALE, calib_model, poly_basis, reported_calib,
                       resol_sigma_model, resol_tau_model)
from .scaling import scale_model
from .fitparamspace import CALIB_K
from .util import bezier2_basis
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
    calib = result.calib_params
    calib_err = result.calib_errors
    coeffs, coeff_errors, _ = reported_calib(
        calib, result.calib_cov, result.detail.channel_max)
    resol = result.resol_params
    resol_err = result.resol_errors
    return ("\n".join([
        rows(PARAM_NAMES_C, coeffs, coeff_errors),
        rows(PARAM_NAMES_K, calib[CALIB_K], calib_err[CALIB_K]),
        rows(PARAM_NAMES_B[:3], resol[:3], resol_err[:3]),
        rows(PARAM_NAMES_B[3:6], resol[3:6], resol_err[3:6]),
    ]))


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray", alpha=0.9))


def _spectrum_panel(ax, ds, title: str | None) -> None:
    ax2 = ax.twinx()
    # Draw the scale curve behind the spectrum artists (and the legend): the
    # twin axis sits below the primary axis, whose background is transparent so
    # the curve stays visible.  Layer order (bottom->top): Scale, Raw sim,
    # Data, Best-fit, Legend.
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    e_curve = np.linspace(ds.bin_edges[0], ds.bin_edges[-1], 300)
    line_scale, = ax2.plot(e_curve,
                           scale_model(ds.scale_params, e_curve,
                                       ds.scale_lo, ds.scale_hi),
                           "--",
                           color="tab:olive", lw=1, zorder=1,
                           label="Scale s(E)")
    ax2.set_ylabel("Scale s(E)")

    centers_full = 0.5 * (ds.bin_edges[:-1] + ds.bin_edges[1:])
    sb_full = scale_model(ds.scale_params, centers_full,
                          ds.scale_lo, ds.scale_hi)
    stairs_handle = ax.stairs(sb_full * ds.unsmeared_sim, ds.bin_edges,
                              color="tab:gray", lw=0.8, zorder=2,
                              label="Raw sim. (scaled)")
    data_handle = ax.errorbar(ds.bin_centers, ds.data_counts,
                              yerr=ds.data_errors,
                              fmt="o", ms=1.5, lw=0.8, alpha=0.6,
                              color="tab:blue", zorder=3,
                              label="Data (-bkg, calibrated)")
    line_fit, = ax.plot(ds.bin_centers, ds.model_prediction, "-",
                        color="tab:red", lw=1.5, zorder=4,
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
               "Scale s(E)"],
              fontsize=8, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=9)


RESID_YMAX = 0.6


def _residual_panel(ax, bin_centers, data_counts, data_errors, model_prediction,
                    energy_low, energy_high, title: str) -> None:
    ok = model_prediction > 0
    rel = (data_counts[ok] - model_prediction[ok]) / model_prediction[ok]
    ax.errorbar(bin_centers[ok], rel, yerr=data_errors[ok] / model_prediction[ok],
                fmt="o", ms=1.5, lw=0.8, color="tab:gray")
    ax.axhline(0, color="k", lw=0.8)
    for level in (-0.3, 0.3):
        ax.axhline(level, color="tab:red", lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual")
    ax.set_xlim(energy_low, energy_high)
    # Fixed range so that the panels of all datasets are directly comparable.
    ax.set_ylim(-RESID_YMAX, RESID_YMAX)
    ax.set_title(title, fontsize=9)


CALIB_BAND_SCALE = 100.0
RESOL_BAND_SCALE = 30.0


def _calibration_panel(ax, calib, channel_max: float = 2048.0,
                       calib_cov=None, title: str = "Energy calibration") -> None:
    channel = np.linspace(0.0, channel_max, 400)
    energy = calib_model(calib, channel, channel_max)
    ax.plot(channel, energy, "-", color="tab:purple", lw=1.5,
            label="$E(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$")
    if calib_cov is not None and np.all(np.isfinite(calib_cov)):
        v = poly_basis(channel, 3)
        err = CALIB_BAND_SCALE * np.sqrt(np.maximum(
            np.sum((v @ calib_cov) * v, axis=1), 0.0))
        ax.fill_between(channel, energy - err, energy + err, color="tab:purple",
                        alpha=0.15, lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                        f"{CALIB_BAND_SCALE:g}}}$)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


RESOL_YMAX = 10.0


def _resolution_panel(ax, resol_params, energy_max: float, resol_cov=None,
                      title: str = "Energy resolution") -> None:
    resol_params = np.asarray(resol_params, dtype=float)
    energy = np.linspace(1.0, energy_max, 300)
    w = bezier2_basis(energy / RESOL_T_SCALE)

    sigma = resol_sigma_model(resol_params, energy)
    tau = resol_tau_model(resol_params, energy)
    std = np.sqrt(sigma * sigma + tau * tau)

    curves = [
        ("sigma", sigma, r"$\sigma\,/\,E$", "tab:orange"),
        ("tau", tau, r"$\tau\,/\,E$", "tab:green"),
        ("std", std, r"$\sqrt{\sigma^2+\tau^2}\,/\,E$", "tab:red"),
    ]
    for name, g, label, color in curves:
        rel = 100.0 * g / energy
        ax.plot(energy, rel, "-", color=color, lw=1.5, label=label)
        if resol_cov is not None and np.all(np.isfinite(resol_cov)):
            grad = np.zeros((len(energy), 6))
            if name == "sigma":
                grad[:, 0:3] = w * resol_params[:3] / sigma[:, None]
            elif name == "tau":
                grad[:, 3:6] = w
            else:  # std = sqrt(sigma^2 + tau^2)
                grad[:, 0:3] = w * resol_params[:3] / std[:, None]
                grad[:, 3:6] = w * (tau[:, None] / std[:, None])
            var = np.maximum(np.sum((grad @ resol_cov) * grad, axis=1), 0.0)
            err = RESOL_BAND_SCALE * 100.0 * np.sqrt(var) / energy
            ax.fill_between(energy, rel - err, rel + err, color=color,
                            alpha=0.12, lw=0)
    ax.set_ylim(0.0, RESOL_YMAX)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"resolution / $E$ (%)")
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
        if n > 1:
            spec_title = (f"{label}  [ch {ds.channel_low}-{ds.channel_high}]  "
                          f"$\\chi^2 = {ds.chi2:.1f}$, {ds.n_bins} bins")
            res_title = f"{label} residual"
        else:
            spec_title = None
            res_title = "residual"
        _spectrum_panel(ax_spec, ds, spec_title)
        _residual_panel(ax_pull, ds.bin_centers, ds.data_counts, ds.data_errors,
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
