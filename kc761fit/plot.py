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
from .resolution import RESOL_ENERGIES, sigma_model
from .fitmodel import PARAM_NAMES_A, PARAM_NAMES_C
from .calibration import CALIB_ENERGIES, poly3
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")


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
        height_ratios=[0.0] + [3.0] * n_datasets + [5.0],
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


def _spectrum_panel(ax, mu, d, err, m, m_i, s_i, elow, ehigh,
                    title: str | None) -> None:
    """One dataset's spectrum panel (log scale, x = energy).

    Data points are drawn at 60% opacity so the model curves stay readable
    underneath."""
    ax.plot(mu, m, "-", color="tab:red", lw=1.5,
            label="Best-fit model (smeared sim.)")
    ax.errorbar(mu, d, yerr=err, fmt="o", ms=1.5, lw=0.8, alpha=0.6,
                color="tab:blue", label="Data (-bkg, calibrated)")
    ax.stairs(s_i * m_i.raw_model_counts(), m_i.grid_edges,
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
    """One dataset's relative-residual panel (x = energy)."""
    ok = m > 0
    rel = (d[ok] - m[ok]) / m[ok]
    ax.errorbar(mu[ok], rel, yerr=err[ok] / m[ok], fmt="o",
                ms=1.5, lw=0.8, color="tab:green")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.3, color="tab:red", lw=0.6, ls=":")
    ax.axhline(-0.3, color="tab:red", lw=0.6, ls=":")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Relative residual $(d-m)/m$")
    ax.set_xlim(elow, ehigh)
    ax.set_ylim(-0.6, 0.6)
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


def plot_fit(model, result, out_pdf: str, elow: float, ehigh: float) -> None:
    """PDF of a single-dataset fit.

    Layout: one (spectrum | relative residual) row, and a single last row
    with the calibration curve, the resolution curve and the parameter list
    side by side (never on top of the spectrum).
    """
    det = result.detail
    p = result.params
    c, a = result.params_c, result.params_a
    x_anchors = p[:4]
    r_anchors = p[4:7]

    fig, gs = _figure_grid(1)
    _title_panel(fig.add_subplot(gs[0]),
                 f"KC761 fit  |  $\\chi^2/\\mathrm{{ndof}} = {result.chi2:.1f}"
                 f"/{result.ndof} = {result.reduced_chi2:.2f}$   "
                 f"$s = {det['s']:.4f}$")
    ax_spec, ax_pull = _dataset_row(fig, gs, 1)
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, 2)

    _spectrum_panel(ax_spec, det["mu"], det["d"], det["err"], det["m"],
                    model, det["s"], elow, ehigh, title=None)
    _residual_panel(ax_pull, det["mu"], det["d"], det["err"], det["m"],
                    elow, ehigh, title="Relative residual")
    _calibration_panel(ax_cal, c, x_anchors, x_max=model.data.edges[-1],
                       cov_c=result.cov_c)
    _resolution_panel(ax_res, a, r_anchors,
                      e_max=float(poly3(c, model.data.edges[-1])),
                      cov_a=result.cov_a)
    _parameter_panel(ax_params, _parameter_text(result))

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_global_fit(model, result, out_pdf: str) -> None:
    """PDF of a global (multi-dataset) fit.

    One (spectrum | relative residual) row per dataset, then a single last
    row with the global-fit calibration, the global-fit resolution and the
    parameter list (global-fit params + derived coefficients + per-dataset
    scales) side by side.  The figure height grows with the number of
    datasets.
    """
    det = result.detail
    p = result.params
    c, a = result.params_c, result.params_a
    x_anchors = p[:4]
    r_anchors = p[4:7]
    n = model.n_datasets
    labels = model.labels

    fig, gs = _figure_grid(n)
    _title_panel(fig.add_subplot(gs[0]),
                 f"KC761 global fit  |  $\\chi^2/\\mathrm{{ndof}} = "
                 f"{result.chi2:.1f}/{result.ndof} = {result.reduced_chi2:.2f}$  "
                 f"({n} datasets, global-fit calibration + resolution)")

    for i, entry in enumerate(det["datasets"]):
        ax_spec, ax_pull = _dataset_row(fig, gs, i + 1)
        m_i = model.models[i]
        label = _cap(labels[i])
        title = (f"{label}  [{m_i.elow:g}-{m_i.ehigh:g} keV]  "
                 f"$\\chi^2 = {entry['chi2']:.1f}$, {entry['bins']} bins, "
                 f"$s = {entry['s']:.4f}$")
        _spectrum_panel(ax_spec, entry["mu"], entry["d"], entry["err"],
                        entry["m"], m_i, entry["s"], m_i.elow, m_i.ehigh,
                        title)
        _residual_panel(ax_pull, entry["mu"], entry["d"], entry["err"],
                        entry["m"], m_i.elow, m_i.ehigh,
                        f"{label} relative residual")

    # Last row: global-fit calibration | global-fit resolution | parameter
    # list.  The calibration is drawn over the channel range valid for *every*
    # dataset (min n_bins); the resolution curve extends to the energy of the
    # last channel of the largest dataset (max channel count).
    min_n_bins = min(m.data.n_bins for m in model.models)
    max_channel = max(m.data.edges[-1] for m in model.models)
    ax_cal, ax_res, ax_params = _footer_row(fig, gs, n + 1)
    _calibration_panel(ax_cal, c, x_anchors, x_max=min_n_bins,
                       cov_c=result.cov_c, title="Energy calibration (global)")
    _resolution_panel(ax_res, a, r_anchors,
                      e_max=float(poly3(c, max_channel)),
                      cov_a=result.cov_a, title="Energy resolution (global)")
    _parameter_panel(ax_params, _parameter_text(result))

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
