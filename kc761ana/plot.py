"""PDF figures of the fit result.

Four stacked panels:
  1. spectrum      : calibrated data (points with errors, log scale) vs
                     best-fit model (resolution-smeared, scaled simulation),
                     plus the raw (unconvolved, scaled) simulation; x = energy;
  2. pulls         : (data - model) / error vs energy;
  3. calibration   : fitted E(x) vs channel;
  4. resolution    : fitted sigma(E) vs energy.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .calibrate import poly3
from .resolution import sigma_model


def plot_fit(model, result, out_pdf: str, elow: float, ehigh: float) -> None:
    det = result.detail
    p = result.params
    c, a = p[:4], p[4:7]

    fig, axes = plt.subplots(
        4, 1, figsize=(9, 13),
        gridspec_kw=dict(height_ratios=[3, 1, 1, 1], hspace=0.4),
    )
    ax_spec, ax_pull, ax_cal, ax_res = axes

    mu = det["mu"]
    d, err, m = det["d"], det["err"], det["m"]
    s = det["s"]

    # --- spectrum panel (log scale, x = energy) ---------------------------
    pos = d > 0  # log scale cannot show non-positive points
    ax_spec.errorbar(mu[pos], d[pos], yerr=err[pos], fmt="o", ms=3, lw=1,
                     color="tab:blue", label="data (sub-bkg, calibrated)")
    ax_spec.stairs(s * model.raw_model_counts(), model.grid_edges,
                   color="tab:gray", lw=0.8,
                   label="raw sim (no resolution, scaled)")
    ax_spec.plot(mu, m, "-", color="tab:red", lw=1.5,
                 label="best-fit model (smeared sim)")
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

    # parameter text box (bottom-left)
    names = result.names
    txt = "\n".join(
        f"{n} = {v: .4g} $\\pm$ {e_: .3g}"
        for n, v, e_ in zip(names, p, result.errors)
    )
    ax_spec.text(0.02, 0.02, txt, transform=ax_spec.transAxes, fontsize=8,
                 va="bottom", ha="left", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # --- pull panel (x = energy) ------------------------------------------
    pull = (d - m) / err
    ax_pull.errorbar(mu, pull, yerr=1.0, fmt="o", ms=2.5, lw=1, color="tab:green")
    ax_pull.axhline(0, color="k", lw=0.8)
    ax_pull.axhline(3, color="tab:red", lw=0.6, ls=":")
    ax_pull.axhline(-3, color="tab:red", lw=0.6, ls=":")
    ax_pull.set_xlabel("energy (keV)")
    ax_pull.set_ylabel("pull")
    ax_pull.set_xlim(elow, ehigh)
    ax_pull.set_ylim(-6, 6)

    # --- calibration curve (x = channel) -----------------------------------
    x = np.linspace(model.data.edges[0], model.data.edges[-1], 400)
    ax_cal.plot(x, poly3(c, x), "-", color="tab:purple", lw=1.5,
                label="calibration $E(x) = c_3 x^3 + c_2 x^2 + c_1 x + c_0$")
    ax_cal.set_xlabel("channel")
    ax_cal.set_ylabel("energy (keV)")
    ax_cal.set_title("energy calibration", fontsize=10)
    ax_cal.grid(alpha=0.3)
    ax_cal.legend(fontsize=8, loc="upper left")

    # --- resolution curve (x = energy) -------------------------------------
    e_res = np.linspace(max(elow, 1.0), ehigh, 300)
    ax_res.plot(e_res, sigma_model(a, e_res), "-", color="tab:orange", lw=1.5,
                label=r"$\sigma(E) = a_2 E + a_1 \sqrt{E} + a_0$")
    ax_res.set_xlabel("energy (keV)")
    ax_res.set_ylabel(r"$\sigma$ (keV)")
    ax_res.set_title("energy resolution", fontsize=10)
    ax_res.grid(alpha=0.3)
    ax_res.legend(fontsize=8, loc="upper left")

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
