"""Unfold orchestration: full mode and ``calib_only`` (W4, F-UNF-4..F-UNF-6).

``run_unfold`` is the W6 entry for ``kc761 unfold``. It loads the data, calib
and simulation products, checks the axis contract (D-114), composes the
full-primary response, solves the padded window (F-UNF-3), propagates the
strict stat/syst bands (F-UNC-1..3) and writes the unfold product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from kc761.core.binning import ChannelGrid
from kc761.errors import ValidationError
from kc761.schema.axes import channel_axis, energy_axis
from kc761.schema.io import build_provenance, write_product
from kc761.schema.products import (
    META_ALPHA,
    META_CHANNEL_HIGH,
    META_CHANNEL_LOW,
    META_CHI2,
    META_COVARIANCE_SCALE,
    META_DIFFERENCE_ORDER,
    META_DOF,
    META_ENERGY_HIGH_KEV,
    META_ENERGY_LOW_KEV,
    META_PAD_NSIGMA,
    META_SYST_FRAC,
    SCHEMA_VERSION,
    UNFOLD_MODE_CALIB_ONLY,
    UNFOLD_MODE_FULL,
    CalibProduct,
    Histogram1D,
    SimProduct,
    SpectrumProduct,
    UnfoldProduct,
)
from kc761.unfold.compose import compose_from_products
from kc761.unfold.inputs import (
    check_data_channel_axis,
    check_recorded_input,
    coerce_calib,
    coerce_sim,
    coerce_spectrum,
    covariance_matrix,
    internal_calibration,
    primary_edges_kev,
    resolution_params,
)
from kc761.unfold.plot import plot_unfold
from kc761.unfold.report import render_report
from kc761.unfold.selection import select_window
from kc761.unfold.solve import solve_window
from kc761.unfold.types import (
    ComposeResult,
    UnfoldResult,
    UnfoldSettings,
)

_CALIB_ONLY_AXIS_NAME = "energy_kev"
_UNFOLDED_AXIS_NAME = "primary_energy_kev"
_COVARIANCE_SCALE = 1.0


def _settings_tuple(
    settings: UnfoldSettings,
    *,
    channel_low: int,
    channel_high: int,
    chi2: float,
    dof: int,
) -> tuple[tuple[str, str], ...]:
    """Build the 11 frozen unfold settings fields (docs/formats.md)."""
    assert settings.alpha is not None
    return (
        (META_ALPHA, repr(float(settings.alpha))),
        (META_DIFFERENCE_ORDER, str(int(settings.difference_order))),
        (META_ENERGY_LOW_KEV, repr(float(settings.energy_low_kev))),
        (META_ENERGY_HIGH_KEV, repr(float(settings.energy_high_kev))),
        (META_CHANNEL_LOW, str(int(channel_low))),
        (META_CHANNEL_HIGH, str(int(channel_high))),
        (META_PAD_NSIGMA, repr(float(settings.pad_nsigma))),
        (META_SYST_FRAC, repr(float(settings.syst_frac))),
        (META_CHI2, repr(float(chi2))),
        (META_DOF, str(int(dof))),
        (META_COVARIANCE_SCALE, repr(float(_COVARIANCE_SCALE))),
    )


def _provenance(
    *,
    producer: str,
    command: str,
    arguments: Sequence[tuple[str, str]],
    paths: Sequence[Path | None],
):
    return build_provenance(
        producer=producer,
        command=command,
        arguments=tuple(arguments),
        inputs=[path for path in paths if path is not None],
    )


def _run_calib_only(
    *,
    data: SpectrumProduct,
    calib: CalibProduct,
    output: str | Path | None,
    force: bool,
    strict: bool,
    producer: str,
    command: str,
    arguments: Sequence[tuple[str, str]],
    data_path: Path | None,
    calib_path: Path | None,
    plot: bool,
    plot_path: str | Path | None,
    plot_force: bool,
) -> UnfoldResult:
    """Relabel the channel axis to ``C.y = E(i +- 1/2)`` without unfolding."""
    edges = np.asarray(calib.deposition_to_channel.y.edges, dtype=np.float64)
    if edges.size != data.spectrum.axis.n_bins + 1:
        raise ValidationError(
            "calib_only: the deposition axis of C does not match the data spectrum bins"
        )
    spectrum = Histogram1D(
        axis=energy_axis(edges, name=_CALIB_ONLY_AXIS_NAME),
        values=np.asarray(data.spectrum.values, dtype=np.float64).copy(),
        variances=np.asarray(data.spectrum.variances, dtype=np.float64).copy(),
    )

    product: UnfoldProduct | None = None
    product_path: Path | None = None
    if output is not None:
        provenance = _provenance(
            producer=producer,
            command=command,
            arguments=arguments,
            paths=(data_path, calib_path),
        )
        product = UnfoldProduct(
            format_version=SCHEMA_VERSION,
            mode=UNFOLD_MODE_CALIB_ONLY,
            spectrum=spectrum,
            sigma_statistical=None,
            sigma_systematic=None,
            sigma_total=None,
            refolded=None,
            settings=(),
            provenance=provenance,
        )
        product_path = write_product(product, output, force=force, strict=strict)

    result = UnfoldResult(
        mode=UNFOLD_MODE_CALIB_ONLY,
        settings=None,
        window=None,
        unfolded=spectrum,
        sigma_statistical=None,
        sigma_systematic=None,
        sigma_total=None,
        refolded=None,
        data_window=None,
        chi2=0.0,
        dof=0,
        covariance_scale=_COVARIANCE_SCALE,
        n_fit_rows=0,
        n_active=0,
        n_kept_columns=0,
        n_pruned_columns=0,
        certificate=None,
        product=product,
        product_path=product_path,
        kept_columns=None,
        pruned_columns=None,
    )
    result = replace(result, report=render_report(result))
    if plot:
        target = plot_path
        if target is None and output is not None:
            target = Path(output).with_suffix(".pdf")
        if target is not None:
            result = replace(result, plot_path=plot_unfold(result, path=target, force=plot_force))
    return result


def run_unfold(
    data: str | Path | SpectrumProduct,
    calib: str | Path | CalibProduct,
    sim: str | Path | SimProduct | None = None,
    *,
    energy_low_kev: float,
    energy_high_kev: float,
    alpha: float | None = None,
    difference_order: int = 2,
    pad_nsigma: float = 5.0,
    syst_frac: float = 0.10,
    calib_only: bool = False,
    output: str | Path | None = None,
    force: bool = False,
    strict: bool = False,
    command: str = "kc761 unfold",
    arguments: Sequence[tuple[str, str]] = (),
    producer: str = "kc761-unfold",
    plot: bool = True,
    plot_path: str | Path | None = None,
    plot_force: bool = False,
) -> UnfoldResult:
    """Unfold a measured channel spectrum (F-SOLVE/F-UNC/F-UNF)."""
    data_product, data_path = coerce_spectrum(data, strict=strict)
    calib_product, calib_path = coerce_calib(calib, strict=strict)
    check_data_channel_axis(calib_product, data_product)

    if calib_only:
        return _run_calib_only(
            data=data_product,
            calib=calib_product,
            output=output,
            force=force,
            strict=strict,
            producer=producer,
            command=command,
            arguments=arguments,
            data_path=data_path,
            calib_path=calib_path,
            plot=plot,
            plot_path=plot_path,
            plot_force=plot_force,
        )

    if sim is None:
        raise ValidationError("--sim is required unless --calib-only is given")
    if alpha is None:
        raise ValidationError("--alpha is required for the full unfold")
    settings = UnfoldSettings(
        alpha=alpha,
        energy_low_kev=energy_low_kev,
        energy_high_kev=energy_high_kev,
        difference_order=difference_order,
        pad_nsigma=pad_nsigma,
        syst_frac=syst_frac,
    )
    sim_product, sim_path = coerce_sim(sim, strict=strict)
    check_recorded_input(sim_product, calib_path)

    calibration = internal_calibration(calib_product)
    resol_params = resolution_params(calib_product)
    param_cov = covariance_matrix(calib_product)
    channel_grid = ChannelGrid(calib_product.deposition_to_channel.x.n_bins)
    primary_edges = primary_edges_kev(sim_product)
    selection = select_window(
        calibration=calibration,
        resol_params=resol_params,
        channel_max=calib_product.channel_max,
        n_channels=channel_grid.n_channels,
        primary_edges_kev=primary_edges,
        energy_low_kev=energy_low_kev,
        energy_high_kev=energy_high_kev,
        pad_nsigma=pad_nsigma,
        strict=strict,
    )

    response, composed, _efficiency = compose_from_products(
        calib_product, sim_product, strict=strict
    )
    outcome = solve_window(
        composed=composed,
        response=response,
        selection=selection,
        data_values=np.asarray(data_product.spectrum.values, dtype=np.float64),
        data_variances=np.asarray(data_product.spectrum.variances, dtype=np.float64),
        deposition_counts=np.asarray(sim_product.primary_to_deposition.values, dtype=np.float64),
        column_totals=np.asarray(sim_product.primary_column_totals.values, dtype=np.float64),
        calibration=calibration,
        resol_params=resol_params,
        channel_grid=channel_grid,
        channel_max=calib_product.channel_max,
        param_cov=param_cov,
        regularization=settings.regularization(),
        syst_frac=syst_frac,
        strict=strict,
    )

    report_slice = slice(selection.report_low, selection.report_high + 1)
    axis = energy_axis(
        primary_edges[selection.report_low : selection.report_high + 2],
        name=_UNFOLDED_AXIS_NAME,
    )
    unfolded_values = outcome.mu_full[report_slice]
    stat_values = outcome.stat_full[report_slice]
    syst_values = outcome.syst_full[report_slice]
    total_values = outcome.total_full[report_slice]
    unfolded = Histogram1D(
        axis=axis, values=unfolded_values, variances=total_values**2
    )
    sigma_statistical = Histogram1D(
        axis=axis, values=stat_values, variances=stat_values**2
    )
    sigma_systematic = Histogram1D(
        axis=axis, values=syst_values, variances=syst_values**2
    )
    sigma_total = Histogram1D(
        axis=axis, values=total_values, variances=total_values**2
    )

    refolded_axis = channel_axis(channel_grid.n_channels).slice(
        selection.channel_low, selection.channel_high
    )
    refolded_full = np.asarray(composed.matrix @ outcome.mu_full, dtype=np.float64)
    refolded = Histogram1D(
        axis=refolded_axis,
        values=refolded_full[selection.channel_low : selection.channel_high + 1],
        variances=None,
    )
    data_window = Histogram1D(
        axis=refolded_axis,
        values=np.asarray(data_product.spectrum.values, dtype=np.float64)[
            selection.channel_low : selection.channel_high + 1
        ],
        variances=np.asarray(data_product.spectrum.variances, dtype=np.float64)[
            selection.channel_low : selection.channel_high + 1
        ],
    )

    settings_tuple = _settings_tuple(
        settings,
        channel_low=selection.channel_low,
        channel_high=selection.channel_high,
        chi2=outcome.chi2,
        dof=outcome.dof,
    )

    product: UnfoldProduct | None = None
    product_path: Path | None = None
    if output is not None:
        provenance = _provenance(
            producer=producer,
            command=command,
            arguments=arguments,
            paths=(data_path, calib_path, sim_path),
        )
        product = UnfoldProduct(
            format_version=SCHEMA_VERSION,
            mode=UNFOLD_MODE_FULL,
            spectrum=unfolded,
            sigma_statistical=sigma_statistical,
            sigma_systematic=sigma_systematic,
            sigma_total=sigma_total,
            refolded=refolded,
            settings=settings_tuple,
            provenance=provenance,
        )
        product_path = write_product(product, output, force=force, strict=strict)

    result = UnfoldResult(
        mode=UNFOLD_MODE_FULL,
        settings=settings,
        window=selection,
        unfolded=unfolded,
        sigma_statistical=sigma_statistical,
        sigma_systematic=sigma_systematic,
        sigma_total=sigma_total,
        refolded=refolded,
        data_window=data_window,
        chi2=outcome.chi2,
        dof=outcome.dof,
        covariance_scale=_COVARIANCE_SCALE,
        n_fit_rows=outcome.n_fit_rows,
        n_active=outcome.n_active,
        n_kept_columns=outcome.n_kept_columns,
        n_pruned_columns=outcome.n_pruned_columns,
        certificate=outcome.certificate,
        product=product,
        product_path=product_path,
        kept_columns=outcome.kept_columns,
        pruned_columns=outcome.pruned_columns,
    )
    result = replace(
        result,
        report=render_report(result, calib=calib_product, sim=sim_product),
    )
    if plot:
        target = plot_path
        if target is None and output is not None:
            target = Path(output).with_suffix(".pdf")
        if target is not None:
            result = replace(result, plot_path=plot_unfold(result, path=target, force=plot_force))
    return result


__all__ = ["ComposeResult", "run_unfold"]
