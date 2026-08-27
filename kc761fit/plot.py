"""PDF figures of the fit result.

Layout (single and global fits share the same structure, with N dataset rows):

    row 0        [ title                                            ]
    row 1..N     [ spectrum_i          | relative residual_i        ]
    row N+1      [ calibration         | resolution | parameter list ]

The last row is a single three-column row: energy calibration, energy
resolution, and the parameter list (fitted channels / resolutions / scale(s)
together with the derived coefficients c0..c3 and a0..a2) side by side —
nothing is drawn on top of a spectrum.  The figure height grows with the
number of datasets so the panels stay readable.

Panels:
  1. spectrum      : calibrated data (points with errors, log scale) vs
                     best-fit model (resolution-smeared, scaled simulation),
                     plus the raw (unconvolved, scaled) simulation; x = energy;
  2. residual      : relative residual (data - model)/model vs energy;
  3. calibration   : fitted E(x) vs channel, with the fitted line positions
                     and a 1-sigma error band (from the coefficient errors);
  4. resolution    : fitted relative resolution sigma(E)/E (%) vs energy,
                     with the fitted resolution points (y capped at 15%) and
                     a 1-sigma error band.
"""

from __future__ import annotations
from pathlib import Path

from .resolution import RESOL_ENERGIES, sigma_model
from .params import PARAM_NAMES_A, PARAM_NAMES_C
from .calibration import CALIB_ENERGIES, poly3
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")


def _save_fig(fig, out_pdf: str) -> None:
    """Save the figure as a PDF, creating the output directory as needed.

    The output is enforced to be a PDF (a non-empty non-``.pdf`` suffix is
    replaced), so a missing ``-o out`` never silently produces a PNG.
    """
    out = Path(out_pdf)
    if not out.suffix.lower().endswith(".pdf"):
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)


def _cap(s: str) -> str:
    """Capitalize the first letter of a string (titles / axis labels)."""
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def _figure_grid(n_datasets: int):
    """Figure + row gridspec whose height grows with the dataset count.

    Row 0 is a full-width title row (axis-less); rows 1..N hold the dataset
    (spectrum | residual) pairs via nested 2-column gridspecs; row N+1 is a
    single three-column row holding calibration | resolution | parameter
    list via a nested 3-column gridspec.

    The title is a real grid row (not a floating ``fig.suptitle``), so it is
    packed directly above the first dataset row with no dead band and no
    risk of colliding with the first panel's title.
    """
    n_rows = n_datasets + 2
    fig = plt.figure(figsize=(15.0, 3.5 * n_datasets + 5.0))
    gs = fig.add_gridspec(
        n_rows, 1,
        height_ratios=[0.5] + [3.0] * n_datasets + [5.0],
        hspace=0.5,
    )
    return fig, gs


def _dataset_row(fig, gs, row: int):
    """Nested [spectrum | relative residual] axes pair for dataset row ``row``."""
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row],
                                    wspace=0.24)
    return (fig.add_subplot(inner[0, 0]),
            fig.add_subplot(inner[0, 1]))


def _footer_row(fig, gs, row: int):
    """Nested [calibration | resolution | parameter list] axes for the last
    row (the plots get slightly more width than the text table)."""
    inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row],
                                    wspace=0.3,
                                    width_ratios=[1.15, 1.15, 0.9])
    return (fig.add_subplot(inner[0, 0]),
            fig.add_subplot(inner[0, 1]),
            fig.add_subplot(inner[0, 2]))


def _title_panel(ax, txt: str) -> None:
    """Figure title in the dedicated top grid row (centered)."""
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=11)


def _parameter_text(result) -> str:
    """Parameter list lines: fitted params (incl. scale(s)) + derived c/a."""
    txt = "\n".join(
        f"{n} = {v: .6g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(result.names, result.params, result.errors)
    )
    txt += "\n" + "\n".join(
        f"{n} = {v: .6g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(PARAM_NAMES_C, result.params_c, result.errors_c)
    )
    txt += "\n" + "\n".join(
        f"{n} = {v: .6g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(PARAM_NAMES_A, result.params_a, result.errors_a)
    )
    return txt


def _parameter_panel(ax, txt: str) -> None:
    """Parameter list in the third column of the last (footer) row.

    The text block is centered in its panel.
    """
    ax.axis("off")
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, va="center", ha="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="#f8f8f8", ec="gray", alpha=0.9))


def _spectrum_panel(ax, mu, d, err, m, m_raw_unsmeared, grid_edges,
                    s_i, elow, ehigh, title: str | None) -> None:
    """One dataset's spectrum panel (log scale, x = energy).

    Data points are drawn at 60% opacity so the model curves stay readable
    underneath."""
    ax.plot(mu, m, "-", color="tab:red", lw=1.5,
            label="Best-fit model (smeared sim.)")
    ax.errorbar(mu, d, yerr=err, fmt="o", ms=1.5, lw=0.8, alpha=0.6,
                color="tab:blue", label="Data (-bkg, calibrated)")
    ax.stairs(s_i * m_raw_unsmeared, grid_edges,
              color="tab:gray", lw=0.8,
              label="Raw sim. (perfect res., scaled)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0)
    ax.set_xlim(elow, ehigh)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=8, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=9)


def _residual_panel(ax, mu, d, err, m, elow, ehigh, title: str) -> None:
    """One dataset's relative-residual panel (x = energy).

    The y range is auto-scaled to the residual spread (3 x the 95th
    percentile of |residual|, capped at +/-0.6) so a good fit's few-percent
    residuals are not compressed into the middle of the panel; the nominal
    +/-30% guide lines are drawn only when they fit in the range.
    """
    ok = m > 0
    rel = (d[ok] - m[ok]) / m[ok]
    ax.errorbar(mu[ok], rel, yerr=err[ok] / m[ok], fmt="o",
                ms=1.5, lw=0.8, color="tab:green")
    ax.axhline(0, color="k", lw=0.8)
    if len(rel):
        lim = float(min(0.6, max(0.05, 3.0 * np.percentile(np.abs(rel), 95))))
    else:
        lim = 0.6
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Relative residual $(d-m)/m$")
    ax.set_xlim(elow, ehigh)
    ax.set_ylim(-lim, lim)
    if lim >= 0.3:
        ax.axhline(0.3, color="tab:red", lw=0.6, ls=":")
        ax.axhline(-0.3, color="tab:red", lw=0.6, ls=":")
    ax.set_title(title, fontsize=9)


# Magnification applied to the 1-sigma error bands drawn in the calibration
# and resolution panels: the true 1-sigma bands are too narrow to be visible
# at the axis scale, so they are scaled up for display.  The two factors are
# independent.
CALIB_BAND_SCALE = 100.0   # calibration: drawn sigma_E(x) = SCALE * 1-sigma
RESOL_BAND_SCALE = 30.0    # resolution:  drawn sigma_r(E) = SCALE * 1-sigma


def _calibration_panel(ax, c, x_anchors, x_max: float = 2048.0,
                       cov_c=None, title: str = "Energy calibration"):
    """Fitted E(x) vs channel with the fitted line positions.

    With ``cov_c`` (the covariance of the calibration coefficients) the
    1-sigma error band of the curve is drawn as well, magnified by
    ``CALIB_BAND_SCALE`` so it is visible next to the curve:
    sigma_E(x) = sqrt(v^T cov_c v), v = [1, x, x^2, x^3].
    """
    x = np.linspace(0.0, x_max, 400)
    e = poly3(c, x)
    ax.plot(x, e, "-", color="tab:purple", lw=1.5,
            label="$E(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$")
    if cov_c is not None and np.all(np.isfinite(cov_c)):
        v = np.stack([np.ones_like(x), x, x**2, x**3], axis=1)
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


# Upper bound of the relative-resolution axis in percent: r = sigma/E is
# clipped at RESOL_YMAX so the diverging low-energy tail does not squash the
# fitted points.
RESOL_YMAX = 15.0


def _resolution_panel(ax, a, r_anchors, e_max: float, cov_a=None,
                      title: str = "Energy resolution"):
    """Fitted relative resolution r(E) = sigma(E)/E, in percent.

    The curve is shown from low energy up to ``e_max`` (the energy of the
    last channel under the fitted calibration); the y-axis is the relative
    resolution in percent, capped at ``RESOL_YMAX`` (15%).  With ``cov_a``
    (the covariance of the resolution coefficients) the 1-sigma error band
    of the curve is drawn as well, magnified by ``RESOL_BAND_SCALE`` so it is
    visible: sigma_r(E) = sqrt(w^T cov_a w), w = [1/E, 1/sqrt(E), 1].
    """
    # Start just above E = 0: r = sigma(0)/0 diverges there.
    e_res = np.linspace(1.0, e_max, 300)
    rel = 100.0 * sigma_model(a, e_res) / e_res  # percent
    ax.plot(e_res, rel, "-", color="tab:orange", lw=1.5,
            label=r"$r(E) = \sigma(E)/E = a_0/E + a_1/\sqrt{E} + a_2$")
    if cov_a is not None and np.all(np.isfinite(cov_a)):
        w = np.stack([1.0 / e_res, 1.0 / np.sqrt(e_res),
                      np.ones_like(e_res)], axis=1)
        err = RESOL_BAND_SCALE * 100.0 * np.sqrt(
            np.clip(np.sum((w @ cov_a) * w, axis=1), 0, None))
        ax.fill_between(e_res, rel - err, rel + err, color="tab:orange",
                        alpha=0.15, lw=0,
                        label=f"1$\\sigma$ band ($\\mathbf{{\\times "
                        f"{RESOL_BAND_SCALE:g}}}$)")
    ax.plot(RESOL_ENERGIES, 100.0 * r_anchors, "o", ms=5, mfc="none",
            color="tab:orange",
            label=f"Fit resolution ({'/'.join(f'{e:g}' for e in RESOL_ENERGIES)} keV)")
    ax.set_ylim(0.0, RESOL_YMAX)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$\sigma/E$ (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_fit(model, result, out_pdf: str) -> None:
    """PDF of a fit (single- or multi-dataset).

    Layout: one (spectrum | relative residual) row per dataset, then a single
    last row with the calibration curve, the resolution curve and the
    parameter list side by side (never on top of a spectrum).  The figure
    height grows with the number of datasets.  A single-dataset fit is the
    N = 1 case of the same layout.
    """
    det = result.detail
    p = result.params
    c, a = result.params_c, result.params_a
    space = model.space
    x_anchors = p[space.channels]
    r_anchors = p[space.resolutions]
    n = model.n_datasets
    models = model.models  # one per-dataset FitModel per row

    fig, gs = _figure_grid(n)
    if n > 1:
        title = (f"KC761 global fit  |  $\\chi^2/\\mathrm{{ndof}} = "
                 f"{result.chi2:.1f}/{result.ndof} = {result.reduced_chi2:.2f}$"
                 f"  ({n} datasets, global-fit calibration + resolution)")
    else:
        title = (f"KC761 fit  |  $\\chi^2/\\mathrm{{ndof}} = "
                 f"{result.chi2:.1f}/{result.ndof} = {result.reduced_chi2:.2f}$"
                 f"   $s = {det.datasets[0].s:.4f}$")
    _title_panel(fig.add_subplot(gs[0]), title)

    for i, (entry, m_i) in enumerate(zip(det.datasets, models)):
        ax_spec, ax_pull = _dataset_row(fig, gs, i + 1)
        label = _cap(entry.label)
        if n > 1:
            spec_title = (f"{label}  [{entry.elow:g}-{entry.ehigh:g} keV]  "
                          f"$\\chi^2 = {entry.chi2:.1f}$, "
                          f"{entry.n_bins} bins, $s = {entry.s:.4f}$")
            pull_title = f"{label} relative residual"
        else:
            spec_title = None
            pull_title = "Relative residual"
        _spectrum_panel(ax_spec, entry.mu, entry.d, entry.err, entry.m,
                        m_i.raw_model_counts(), entry.grid_edges, entry.s,
                        entry.elow, entry.ehigh, spec_title)
        _residual_panel(ax_pull, entry.mu, entry.d, entry.err, entry.m,
                        entry.elow, entry.ehigh, pull_title)

    # Last row: calibration | resolution | parameter list.  The calibration is
    # drawn over the channel range valid for *every* dataset; the resolution
    # curve extends to the energy of the last channel of the largest dataset.
    cal_title = "Energy calibration" if n == 1 else "Energy calibration (global)"
    res_title = "Energy resolution" if n == 1 else "Energy resolution (global)"
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    _calibration_panel(ax_cal, c, x_anchors, x_max=model.n_channel_bins,
                       cov_c=result.cov_c, title=cal_title)
    _resolution_panel(ax_res, a, r_anchors,
                      e_max=float(poly3(c, model.channel_max)),
                      cov_a=result.cov_a, title=res_title)
    _parameter_panel(ax_params, _parameter_text(result))

    _save_fig(fig, out_pdf)
