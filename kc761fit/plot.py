"""PDF figures of the fit result."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path

from .params import CHANNELS, PARAM_NAMES_B, PARAM_NAMES_C, RESOLUTIONS
from .response import (CALIB_ENERGIES, RESOL_ENERGIES, poly3, poly_basis,
                       sigma_model)


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
                                    wspace=0.24)
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
    rows = lambda names, vals, errs: "\n".join(
        f"{n} = {v: .6g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(names, vals, errs))
    return ("\n".join([
        rows(result.names, result.params, result.errors),
        rows(PARAM_NAMES_C, result.params_c, result.errors_c),
        rows(PARAM_NAMES_B, result.params_b, result.errors_b),
    ]))


def _parameter_panel(ax, txt: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray", alpha=0.9))


def _spectrum_panel(ax, ds, title: str | None) -> None:
    ax.plot(ds.mu, ds.m, "-", color="tab:red", lw=1.5,
            label="Best-fit model (smeared sim.)")
    ax.errorbar(ds.mu, ds.d, yerr=ds.err, fmt="o", ms=1.5, lw=0.8,
                alpha=0.6, color="tab:blue",
                label="Data (-bkg, calibrated)")
    ax.stairs(ds.s * ds.sim_raw, ds.grid_edges,
              color="tab:gray", lw=0.8,
              label="Raw sim. (perfect res., scaled)")
    ax.set_yscale("log")
    ax.set_xlim(ds.elow, ds.ehigh)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=8, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=9)


RESID_YMAX = 0.6


def _residual_panel(ax, mu, d, err, m, elow, ehigh, title: str) -> None:
    ok = m > 0
    rel = (d[ok] - m[ok]) / m[ok]
    ax.errorbar(mu[ok], rel, yerr=err[ok] / m[ok], fmt="o",
                ms=1.5, lw=0.8, color="tab:green")
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


def _calibration_panel(ax, c, x_anchors, x_max: float = 2048.0,
                       cov_c=None, title: str = "Energy calibration") -> None:
    x = np.linspace(0.0, x_max, 400)
    e = poly3(c, x)
    ax.plot(x, e, "-", color="tab:purple", lw=1.5,
            label="$E(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$")
    if cov_c is not None and np.all(np.isfinite(cov_c)):
        v = poly_basis(x, 3)
        err = CALIB_BAND_SCALE * np.sqrt(
            np.clip(np.sum((v @ cov_c) * v, axis=1), 0, None))
        ax.fill_between(x, e - err, e + err, color="tab:purple", alpha=0.15,
                        lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                              f"{CALIB_BAND_SCALE:g}}}$)")
    ax.plot(x_anchors, CALIB_ENERGIES, "o", ms=5, mfc="none",
            color="tab:purple",
            label="Fit line positions (60/609/1461/2614 keV)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


RESOL_YMAX = 15.0


def _resolution_panel(ax, b, r_anchors, e_max: float, cov_b=None,
                      title: str = "Energy resolution") -> None:
    e_res = np.linspace(1.0, e_max, 300)
    rel = 100.0 * sigma_model(b, e_res) / e_res
    ax.plot(e_res, rel, "-", color="tab:orange", lw=1.5,
            label=r"$r(E) = \sqrt{b_0 + b_1 E + b_2 E^2}\,/\,E$")
    if cov_b is not None and np.all(np.isfinite(cov_b)):
        # Delta method: d(sigma/E)/db = [1, E, E^2] / (2 E sigma(E)).
        var = np.maximum(b[0] + b[1] * e_res + b[2] * e_res**2, 1e-12)
        grad = poly_basis(e_res, 2) / (2.0 * e_res * np.sqrt(var))[:, None]
        err = RESOL_BAND_SCALE * 100.0 * np.sqrt(
            np.clip(np.sum((grad @ cov_b) * grad, axis=1), 0, None))
        ax.fill_between(e_res, rel - err, rel + err, color="tab:orange",
                        alpha=0.15, lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                              f"{RESOL_BAND_SCALE:g}}}$)")
    ax.plot(RESOL_ENERGIES, 100.0 * r_anchors, "o", ms=5, mfc="none",
            color="tab:orange",
            label=("Fit resolution ("
                   + "/".join(f"{e:g}" for e in RESOL_ENERGIES) + " keV)"))
    ax.set_ylim(0.0, RESOL_YMAX)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$\sigma/E$ (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_fit(result, out_pdf: str) -> None:
    det = result.detail
    c, b = result.params_c, result.params_b
    n = len(det.datasets)

    fig, gs = _figure_grid(n)
    scale_note = (f"({n} datasets, global-fit calibration + resolution)"
                  if n > 1 else f"$s = {det.datasets[0].s:.4f}$")
    _title_panel(
        fig.add_subplot(gs[0]),
        f"KC761 {'global ' if n > 1 else ''}fit  |  "
        f"$\\chi^2/\\mathrm{{ndof}} = {result.chi2:.1f}/{result.ndof} "
        f"= {result.reduced_chi2:.2f}$  {scale_note}")

    for i, ds in enumerate(det.datasets):
        ax_spec, ax_pull = _dataset_row(fig, gs, i + 1)
        label = _cap(ds.label)
        if n > 1:
            spec_title = (f"{label}  [{ds.elow:g}-{ds.ehigh:g} keV]  "
                          f"$\\chi^2 = {ds.chi2:.1f}$, {ds.n_bins} bins, "
                          f"$s = {ds.s:.4f}$")
            pull_title = f"{label} relative residual"
        else:
            spec_title = None
            pull_title = "Relative residual"
        _spectrum_panel(ax_spec, ds, spec_title)
        _residual_panel(ax_pull, ds.mu, ds.d, ds.err, ds.m,
                        ds.elow, ds.ehigh, pull_title)

    cal_title = "Energy calibration" + (" (global)" if n > 1 else "")
    res_title = "Energy resolution" + (" (global)" if n > 1 else "")
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    _calibration_panel(ax_cal, c, result.params[CHANNELS],
                       x_max=det.n_channel_bins, cov_c=result.cov_c,
                       title=cal_title)
    _resolution_panel(ax_res, b, result.params[RESOLUTIONS],
                      e_max=float(poly3(c, det.channel_max)),
                      cov_b=result.cov_b, title=res_title)
    _parameter_panel(ax_params, _parameter_text(result))

    _save_fig(fig, out_pdf)
