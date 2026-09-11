"""Text report of a calibration fit.

Prints the chi-square per degree of freedom, the reported parameters with their
F-CAL-5 uncertainties, the per-dataset Bezier scale (F-CAL-2) and the
F-MODEL-5 clamp summary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761.calib.model import DatasetDetail
from kc761.calib.types import FitResult
from kc761.core.model import PARAM_NAMES_REPORTED


def _reported_errors(param_cov: NDArray[np.float64]) -> NDArray[np.float64]:
    diagonal = np.diag(np.asarray(param_cov, dtype=np.float64))
    return np.sqrt(np.maximum(diagonal, 0.0))


def render_report(result: FitResult, details: tuple[DatasetDetail, ...] = ()) -> str:
    """Render the calibration fit report as a newline-joined string."""
    lines: list[str] = []
    lines.append("=== KC761 calibration ===")
    lines.append(f"status: {result.status}")
    if result.message:
        lines.append(f"message: {result.message}")
    reduced = result.chi2 / result.dof if result.dof > 0 else float("nan")
    lines.append(f"chi2/dof = {result.chi2:.8g} / {result.dof} = {reduced:.8g}")
    lines.append(f"covariance s^2 = {result.covariance_scale:.8g}")
    errors = _reported_errors(result.param_cov)
    lines.append("--- reported parameters (F-MODEL-2) ---")
    values = (*result.params_reported, *result.resol_params)
    for name, value, error in zip(PARAM_NAMES_REPORTED, values, errors, strict=True):
        lines.append(f"  {name} = {value: .10g} +- {error:.3g}")
    lines.append("--- per-dataset Bezier scale (F-CAL-2) ---")
    for scale in result.scales:
        s0, s1, s2, s3 = scale.params
        lines.append(
            f"  {scale.label}: s0={s0:.8g} s1={s1:.8g} s2={s2:.8g} s3={s3:.8g}"
            f"  (initial {scale.initial_scale:.8g})"
        )
    lines.append("--- scale boundary flags (s0,s1,s2,s3; D-103) ---")
    for label, flags in result.scale_bound_flags:
        lines.append(f"  {label}: {tuple(int(flag) for flag in flags)}")
    lines.append("--- resolution clamp (F-MODEL-5, export grid) ---")
    lines.append(f"  clamped points: {result.resol_clamp_count}")
    if result.resol_clamp_count:
        lines.append(
            f"  clamped energy range: [{result.resol_clamp_energy_low_kev:.8g}, "
            f"{result.resol_clamp_energy_high_kev:.8g}] keV"
        )
    if details:
        lines.append("--- per-dataset chi2 ---")
        for detail in details:
            lines.append(
                f"  {detail.label}: chi2={detail.chi2:.8g} over {detail.data_counts.size} bins"
            )
    return "\n".join(lines)


__all__ = ["render_report"]
