"""PDF figures of the fit result.

Four stacked panels:
  1. spectrum      : calibrated data (points with errors, log scale) vs
                     best-fit model (resolution-smeared, scaled simulation),
                     plus the raw (unconvolved, scaled) simulation; x = energy;
  2. residual      : relative residual (data - model)/model vs energy;
  3. calibration   : fitted E(x) vs channel, with the fitted line positions;
  4. resolution    : fitted sigma(E) vs energy, with the fitted sigma points.

The parameter text box lists the fitted channels / relative resolutions /
scale together with the derived coefficients c0..c3 and a0..a2.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .calibration import CAL_ENERGIES, poly3
from .fitmodel import PARAM_NAMES_A, PARAM_NAMES_C
from .resolution import RES_ENERGIES, sigma_model


def plot_fit(model, result, out_pdf: str, elow: float, ehigh: float) -> None:
    det = result.detail
    p = result.params
    c, a = result.params_c, result.params_a
    x_anchors = p[:4]
    r_anchors = p[4:7]

    fig, axes = plt.subplots(
        4, 1, figsize=(9, 13),
        gridspec_kw=dict(height_ratios=[3, 1, 1, 1], hspace=0.4),
    )
    ax_spec, ax_pull, ax_cal, ax_res = axes

    mu = det["mu"]
    d, err, m = det["d"], det["err"], det["m"]
    s = det["s"]

    # --- spectrum panel (log scale, x = energy) ---------------------------
    ax_spec.errorbar(mu, d, yerr=err, fmt="o", ms=1, lw=0.1,
                     color="tab:blue", label="data (sub-bkg, calibrated)")
    ax_spec.plot(mu, m, "-", color="tab:red", lw=1.5,
                 label="best-fit model (smeared sim)")
    ax_spec.stairs(s * model.raw_model_counts(), model.grid_edges,
                   color="tab:gray", lw=0.8,
                   label="raw sim (no resolution, scaled)")
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(bottom=1.0)
    ax_spec.set_xlabel("energy (keV)")
    ax_spec.set_ylabel("counts")
    ax_spec.set_xlim(elow, ehigh)
    ax_spec.legend(fontsize=8, loc="upper right")
    ax_spec.set_title(
        f"kc761 fit  |  $\\chi^2/\\mathrm{{ndof}} = {result.chi2:.1f}/{result.ndof}"
        f" = {result.reduced_chi2:.2f}$   $s = {s:.4f}$",
        fontsize=10,
    )

    # parameter text box (bottom-left): fitted + derived parameters
    txt = "\n".join(
        f"{n} = {v: .4g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(result.names, p, result.errors)
    )
    txt += "\n" + "\n".join(
        f"{n} = {v: .4g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(PARAM_NAMES_C, c, result.errors_c)
    )
    txt += "\n" + "\n".join(
        f"{n} = {v: .4g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(PARAM_NAMES_A, a, result.errors_a)
    )
    ax_spec.text(0.02, 0.02, txt, transform=ax_spec.transAxes, fontsize=7.5,
                 va="bottom", ha="left", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # --- relative-residual panel (x = energy) ------------------------------
    # Fractional deviation (data - model)/model; error bars are the relative
    # data errors.  Bins with a zero model (no data overlap) are skipped.
    ok = m > 0
    rel = (d[ok] - m[ok]) / m[ok]
    ax_pull.errorbar(mu[ok], rel, yerr=err[ok] / m[ok], fmt="o",
                     ms=1, lw=0.1, color="tab:green")
    ax_pull.axhline(0, color="k", lw=0.8)
    ax_pull.axhline(0.3, color="tab:red", lw=0.6, ls=":")
    ax_pull.axhline(-0.3, color="tab:red", lw=0.6, ls=":")
    ax_pull.set_xlabel("energy (keV)")
    ax_pull.set_ylabel("relative residual $(d-m)/m$")
    ax_pull.set_xlim(elow, ehigh)
    ax_pull.set_ylim(-0.6, 0.6)

    # --- calibration curve (x = channel) -----------------------------------
    x = np.linspace(model.data.edges[0], model.data.edges[-1], 400)
    ax_cal.plot(x, poly3(c, x), "-", color="tab:purple", lw=1.5,
                label="$E(x) = c_3 x^3 + c_2 x^2 + c_1 x + c_0$")
    ax_cal.plot(x_anchors, CAL_ENERGIES, "o", ms=5, mfc="none",
                color="tab:purple",
                label="fit line positions (60/609/1461/2614 keV)")
    ax_cal.set_xlabel("channel")
    ax_cal.set_ylabel("energy (keV)")
    ax_cal.set_title("energy calibration", fontsize=10)
    ax_cal.grid(alpha=0.3)
    ax_cal.legend(fontsize=8, loc="upper left")

    # --- resolution curve (x = energy) -------------------------------------
    # Show the full fitted resolution model over [0, 3000] keV so all three
    # fitted sigma points (60/1461/2614 keV) and the extrapolation are
    # visible regardless of the fit range [elow, ehigh].
    e_res = np.linspace(0.0, 3000.0, 300)
    sig = sigma_model(a, e_res)
    ax_res.plot(e_res, sig, "-", color="tab:orange", lw=1.5,
                label=r"$\sigma(E) = a_2 E + a_1 \sqrt{E} + a_0$")
    ax_res.plot(RES_ENERGIES, RES_ENERGIES * r_anchors, "o", ms=5, mfc="none",
                color="tab:orange",
                label=f"fit resolution ({'/'.join(f'{e:g}' for e in RES_ENERGIES)} keV)")
    ax_res.set_xlim(0.0, 3000.0)
    ax_res.set_xlabel("energy (keV)")
    ax_res.set_ylabel(r"$\sigma$ (keV)")
    ax_res.set_title("energy resolution", fontsize=10)
    ax_res.grid(alpha=0.3)
    ax_res.legend(fontsize=8, loc="upper left")

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
