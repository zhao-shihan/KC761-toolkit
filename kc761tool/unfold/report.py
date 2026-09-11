"""Text report for an unfold run.

The report is a plain newline-joined string with the frozen diagnostics
(F-UNF-4): window, alpha, degrees of freedom, band summary, the F-MODEL-5
clamp record and the scale-bound flags carried by the calibration product.
"""

from __future__ import annotations

import numpy as np

from kc761tool.schema.products import (
    UNFOLD_MODE_CALIB_ONLY,
    CalibProduct,
    SimProduct,
)
from kc761tool.unfold.types import UnfoldResult


def _band_summary(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    return (
        f"max={float(np.max(values)):.6g} median={float(np.median(values)):.6g} "
        f"min={float(np.min(values)):.6g}"
    )


def render_report(
    result: UnfoldResult,
    *,
    calib: CalibProduct | None = None,
    sim: SimProduct | None = None,
) -> str:
    """Render the unfold report as a newline-joined string."""
    lines: list[str] = ["=== KC761 unfold ===", f"mode: {result.mode}"]
    if result.mode == UNFOLD_MODE_CALIB_ONLY:
        lines.append(
            "calib_only: relabeled the channel axis to E(i +- 1/2); no solve")
        if result.unfolded is not None:
            values = np.asarray(result.unfolded.values, dtype=np.float64)
            lines.append(
                f"bins: {values.size}  total counts: {float(values.sum()):.6g}  "
                f"unit: {result.unfolded.axis.unit}"
            )
        return "\n".join(lines)

    assert result.settings is not None
    assert result.window is not None
    settings = result.settings
    window = result.window
    lines.append(
        f"window: [{window.energy_low_kev:.6g}, {window.energy_high_kev:.6g}] keV -> "
        f"channels [{window.channel_low}, {window.channel_high}] "
        f"(solver [{window.solve_low}, {window.solve_high}])"
    )
    lines.append(
        f"alpha = {settings.alpha!r}  difference_order = {settings.difference_order}  "
        f"pad_nsigma = {settings.pad_nsigma!r}  syst_frac = {settings.syst_frac!r}"
    )
    if result.unfolded is not None:
        total = int(result.unfolded.axis.n_bins)
        lines.append(
            f"reported primary bins: {total} "
            f"(fit rows {result.n_fit_rows}, kept columns {result.n_kept_columns}, "
            f"exact-zero pruned {result.n_pruned_columns})"
        )
    reduced = result.chi2 / result.dof if result.dof > 0 else float("nan")
    lines.append(
        f"chi2 = {result.chi2:.8g}  dof = {result.dof}  chi2/dof = {reduced:.6g}  "
        f"covariance_scale = {result.covariance_scale:g}"
    )
    lines.append(f"active solution bins: {result.n_active}")
    if result.sigma_total is not None:
        total_band = np.asarray(result.sigma_total.values, dtype=np.float64)
        stat_band = np.asarray(
            result.sigma_statistical.values, dtype=np.float64)
        syst_band = np.asarray(
            result.sigma_systematic.values, dtype=np.float64)
        lines.append(f"total band: {_band_summary(total_band)}")
        lines.append(f"stat band:  {_band_summary(stat_band)}")
        lines.append(f"syst band:  {_band_summary(syst_band)}")
    if result.certificate is not None:
        certificate = result.certificate
        lines.append(
            f"KKT: converged={certificate.converged} "
            f"max_neg_reduced_gradient={certificate.max_negative_reduced_gradient:.3g} "
            f"complementarity={certificate.complementarity:.3g} "
            f"iterations={certificate.iterations}"
        )
    if sim is not None:
        lines.append(
            f"simulation: mode={sim.mode_name} geometry={sim.geometry_name} "
            f"angular={sim.angular_distribution} seed={sim.seed} n_events={sim.n_events}"
        )
    if calib is not None:
        lines.append(
            f"calibration: fit_status={calib.fit_status} chi2/dof="
            f"{calib.chi2 / calib.dof if calib.dof > 0 else float('nan'):.6g}"
        )
        lines.append(
            f"resolution clamp (F-MODEL-5): count={calib.resol_clamp_count} "
            f"energy range=[{calib.resol_clamp_energy_low_kev:.6g}, "
            f"{calib.resol_clamp_energy_high_kev:.6g}] keV"
        )
        for label, flags in calib.scale_bound_flags:
            if any(flags):
                lines.append(
                    f"scale bound flag: {label} -> {tuple(int(f) for f in flags)}")
    return "\n".join(lines)


__all__ = ["render_report"]
