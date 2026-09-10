"""Calibration fit entry point (F-CAL-1..F-CAL-5, W3).

``run_fit`` is the public W6 entry: it builds the global model, runs the
single-stage bounded quasi-Newton fit with the analytic F-CAL-4 gradient,
certifies the result (F-MODEL-3/F-MODEL-5/F-COV-2), extracts the reported
covariance (F-CAL-5), and optionally writes the calibration product and the
figure.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from kc761.calib.covariance import calibration_covariance
from kc761.calib.model import N_CORE, CalibrationModel
from kc761.calib.plot import plot_fit
from kc761.calib.product import build_calib_product, channel_derived_edges_kev
from kc761.calib.report import render_report
from kc761.calib.scaling import N_SCALE
from kc761.calib.types import DatasetSpec, FitResult, FitSettings, ScaleResult
from kc761.core.covariance import verify_covariance_psd
from kc761.core.model import (
    InternalCalibration,
    internal_to_reported,
    resolution_clamped_mask,
    verify_energy_monotonicity,
    verify_resolution_positivity,
)
from kc761.errors import SolverError
from kc761.schema.io import build_provenance, write_product

STATUS_CONVERGED: Final = "converged"
STATUS_STOPPED: Final = "stopped-early"

_BOUND_FLAG_TOL: Final = 1e-6
"""Relative tolerance for recording a scale parameter as bound-limited (D-103)."""


def _scale_bound_flags(
    theta: NDArray[np.float64], model: CalibrationModel
) -> tuple[tuple[str, tuple[bool, bool, bool, bool]], ...]:
    """Per-dataset flags for scale parameters sitting on a bound (D-103).

    A bound hit means the scale is at (or numerically on) the boundary of its
    allowed range; for ``s0`` this is the (nearly) polynomial-stratum regime,
    which is allowed and recorded rather than rejected.
    """
    flags: list[tuple[str, tuple[bool, bool, bool, bool]]] = []
    for index, spec in enumerate(model.specs):
        start = N_CORE + N_SCALE * index
        values = theta[start : start + N_SCALE]
        bounds = model.bounds[start : start + N_SCALE]
        entry: list[bool] = []
        for value, (low, high) in zip(values, bounds, strict=True):
            tolerance = _BOUND_FLAG_TOL * max(1.0, abs(high - low))
            entry.append(bool(value <= low + tolerance or value >= high - tolerance))
        flags.append((spec.label, (entry[0], entry[1], entry[2], entry[3])))
    return tuple(flags)


def _unique_inputs(specs: Sequence[DatasetSpec]) -> list[str | Path]:
    seen: set[str] = set()
    paths: list[str | Path] = []
    for spec in specs:
        for candidate in (spec.data_path, spec.mc_path):
            if candidate is None:
                continue
            key = str(Path(candidate).resolve())
            if key not in seen:
                seen.add(key)
                paths.append(candidate)
    return paths


def run_fit(
    datasets: Sequence[DatasetSpec],
    *,
    channel_max: float | None = None,
    output: str | Path | None = None,
    command: str = "kc761 calib",
    arguments: Sequence[tuple[str, str]] = (),
    producer: str = "kc761-calib",
    force: bool = False,
    strict: bool = False,
    settings: FitSettings | None = None,
    plot: bool = True,
    plot_path: str | Path | None = None,
    plot_force: bool = False,
    extra_inputs: Sequence[str | Path] = (),
) -> FitResult:
    """Fit a shared calibration over one or more datasets (W3).

    The product is written only when ``output`` is given; ``plot=True`` writes
    a figure next to ``output`` (or to ``plot_path``). Non-convergence is
    recorded in ``fit_status`` and, outside strict mode, still writes the
    product; strict mode raises. Degenerate fits raise in every mode.
    ``extra_inputs`` are hashed into the product provenance in addition to the
    dataset paths; the W6 config mode passes the configuration file here
    (D-133).
    """
    specs = tuple(datasets)
    resolved_settings = settings if settings is not None else FitSettings()
    model = CalibrationModel(specs, channel_max=channel_max, settings=resolved_settings)

    lower = np.asarray([bound[0] for bound in model.bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in model.bounds], dtype=np.float64)
    if np.any(upper - lower <= 0.0):
        raise SolverError("calibration bounds must satisfy lo < hi for every parameter")

    def residual(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return model.residuals(theta)

    def residual_jacobian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        sigma = np.sqrt(model.variance(theta))
        residual_values = model.residuals(theta)
        variance = model.variance(theta)
        return -(model.jacobian(theta) / sigma[:, None]) - 0.5 * (
            model.variance_gradient(theta) * (residual_values / variance)[:, None]
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimization = optimize.least_squares(
            residual,
            model.x0,
            jac=residual_jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=resolved_settings.ftol,
            xtol=resolved_settings.xtol,
            gtol=resolved_settings.gtol,
            max_nfev=resolved_settings.maxiter,
        )
    theta = np.asarray(optimization.x, dtype=np.float64)
    if theta.size != model.n_free or not np.all(np.isfinite(theta)):
        raise SolverError(
            f"calibration optimizer returned an invalid point ({theta.size} entries, "
            f"finite={bool(np.all(np.isfinite(theta)))})"
        )

    clamp_events = sum("F-MODEL-5" in str(warning.message) for warning in caught)
    if clamp_events:
        warnings.warn(
            f"F-MODEL-5 clamp was active in {clamp_events} fit evaluation(s); "
            "the export-grid summary is recorded in the product meta (D-104)",
            RuntimeWarning,
            stacklevel=2,
        )

    core_internal = np.asarray(theta[:N_CORE], dtype=np.float64)
    core_calib = np.asarray(core_internal[:4], dtype=np.float64)
    resol = np.asarray(core_internal[4:], dtype=np.float64)
    calibration = InternalCalibration.from_array(core_calib)
    verify_energy_monotonicity(calibration, channel_max=model.channel_max, strict=strict)

    export_edges = channel_derived_edges_kev(core_calib, model.n_channels, model.channel_max)
    export_centers = 0.5 * (export_edges[:-1] + export_edges[1:])
    verify_resolution_positivity(export_centers, resol, strict=strict)
    clamped = resolution_clamped_mask(export_centers, resol)
    clamp_count = int(np.count_nonzero(clamped))
    if clamp_count:
        clamp_low = float(np.min(export_centers[clamped]))
        clamp_high = float(np.max(export_centers[clamped]))
    else:
        clamp_low = 0.0
        clamp_high = 0.0

    chi2 = model.evaluate(theta)
    dof = model.dof
    covariance = calibration_covariance(
        model.jacobian(theta),
        model.variance(theta),
        chi2=chi2,
        dof=dof,
        channel_max=model.channel_max,
    )
    verify_covariance_psd(covariance, strict=strict)

    success = bool(optimization.success)
    status = STATUS_CONVERGED if success else STATUS_STOPPED
    message = str(optimization.message)
    if strict and not success:
        raise SolverError(f"calibration fit did not converge in strict mode: {message}")

    scales = tuple(
        ScaleResult(
            label=spec.label,
            initial_scale=model.initial_scales[index],
            params=(
                float(theta[N_CORE + N_SCALE * index]),
                float(theta[N_CORE + N_SCALE * index + 1]),
                float(theta[N_CORE + N_SCALE * index + 2]),
                float(theta[N_CORE + N_SCALE * index + 3]),
            ),
        )
        for index, spec in enumerate(specs)
    )

    reported = internal_to_reported(calibration, channel_max=model.channel_max)
    scale_bound_flags = _scale_bound_flags(theta, model)
    details = model.dataset_details(theta)

    product = None
    product_path: Path | None = None
    if output is not None:
        provenance = build_provenance(
            producer=producer,
            command=command,
            arguments=tuple(arguments),
            inputs=[*_unique_inputs(specs), *extra_inputs],
        )
        product = build_calib_product(
            core_internal=core_calib,
            resol_params=resol,
            param_cov=covariance.matrix,
            chi2=chi2,
            dof=dof,
            covariance_scale=covariance.scale,
            fit_status=status,
            scales=tuple((scale.label, scale.params) for scale in scales),
            scale_bound_flags=scale_bound_flags,
            n_channels=model.n_channels,
            channel_max=model.channel_max,
            provenance=provenance,
            strict=strict,
        )
        product_path = write_product(product, output, force=force, strict=strict)

    result = FitResult(
        success=success,
        status=status,
        message=message,
        nfev=int(optimization.nfev),
        core_internal=core_internal,
        params_reported=(
            float(reported.c0),
            float(reported.c1),
            float(reported.c2),
            float(reported.c3),
        ),
        resol_params=(float(resol[0]), float(resol[1]), float(resol[2])),
        param_cov=np.asarray(covariance.matrix, dtype=np.float64),
        scales=scales,
        scale_bound_flags=scale_bound_flags,
        chi2=float(chi2),
        dof=int(dof),
        covariance_scale=float(covariance.scale),
        n_bins=model.n_bins,
        n_free=model.n_free,
        channel_max=float(model.channel_max),
        resol_clamp_count=clamp_count,
        resol_clamp_energy_low_kev=clamp_low,
        resol_clamp_energy_high_kev=clamp_high,
        product=product,
        product_path=product_path,
    )
    result = replace(result, report=render_report(result, details))
    if plot:
        target = plot_path
        if target is None and output is not None:
            target = Path(output).with_suffix(".pdf")
        if target is not None:
            result = replace(
                result, plot_path=plot_fit(result, details, path=target, force=plot_force)
            )
    return result


__all__ = ["STATUS_CONVERGED", "STATUS_STOPPED", "run_fit"]
