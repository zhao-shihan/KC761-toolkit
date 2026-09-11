"""Calibration product assembly (F-RESP-1/F-IO-1, D-101/D-106).

Builds the export response matrix on the channel-derived non-uniform axis
``E(i +- 1/2)`` (the axis construction, D-101) from the fitted core,
together with the reported parameters, the reported-basis covariance and the
F-MODEL-5 clamp summary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761tool.core._checks import as_float_array
from kc761tool.core.binning import ChannelGrid
from kc761tool.core.model import (
    N_REPORTED_PARAMS,
    InternalCalibration,
    energy_kev,
    internal_to_reported,
    resolution_clamped_mask,
)
from kc761tool.core.response import build_response_matrix
from kc761tool.errors import ValidationError
from kc761tool.schema.axes import (
    DEPOSITION_AXIS_NAME,
    channel_axis,
    energy_axis,
    reported_parameter_axis,
)
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    Histogram2D,
    Provenance,
)


def channel_derived_edges_kev(
    core_internal: NDArray[np.float64], n_channels: int, channel_max: float
) -> NDArray[np.float64]:
    """Export deposition axis ``E(i +- 1/2)`` for ``i = 0..n_channels`` (D-101).

    The single source of the channel-derived axis used by the exported
    response matrix; both the product builder and the strict positivity check
    go through it.
    """
    core = as_float_array("core_internal", core_internal, ndim=1)
    calibration = InternalCalibration.from_array(core)
    channel_edges = np.arange(int(n_channels) + 1, dtype=np.float64) - 0.5
    return np.asarray(
        energy_kev(channel_edges, calibration, channel_max=channel_max),
        dtype=np.float64,
    )


def build_calib_product(
    *,
    core_internal: NDArray[np.float64],
    resol_params: NDArray[np.float64],
    param_cov: NDArray[np.float64],
    chi2: float,
    dof: int,
    covariance_scale: float,
    fit_status: str,
    scales: tuple[tuple[str, tuple[float, float, float, float]], ...],
    scale_bound_flags: tuple[tuple[str, tuple[bool, bool, bool, bool]], ...],
    n_channels: int,
    channel_max: float,
    provenance: Provenance,
    strict: bool = False,
) -> CalibProduct:
    """Assemble the calibration product from a fitted core (D-101/D-106)."""
    core = as_float_array("core_internal", core_internal, ndim=1)
    if core.size != 4:
        raise ValidationError(
            f"core_internal must have 4 entries, got {core.size}")
    resol = as_float_array("resol_params", resol_params, ndim=1)
    if resol.size != 3:
        raise ValidationError(
            f"resol_params must have 3 entries, got {resol.size}")
    covariance = as_float_array("param_cov", param_cov, ndim=2)
    if covariance.shape != (N_REPORTED_PARAMS, N_REPORTED_PARAMS):
        raise ValidationError(f"param_cov must be 7x7, got {covariance.shape}")
    if int(n_channels) < 1:
        raise ValidationError(f"n_channels must be >= 1, got {n_channels!r}")

    calibration = InternalCalibration.from_array(core)
    energy_edges = channel_derived_edges_kev(
        core, int(n_channels), channel_max)
    response = build_response_matrix(
        energy_edges,
        calibration,
        resol,
        channel_grid=ChannelGrid(int(n_channels)),
        channel_max=channel_max,
        strict=strict,
    )
    centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    clamped = resolution_clamped_mask(centers, resol)
    clamp_count = int(np.count_nonzero(clamped))
    if clamp_count:
        clamp_low = float(np.min(centers[clamped]))
        clamp_high = float(np.max(centers[clamped]))
    else:
        clamp_low = 0.0
        clamp_high = 0.0

    reported = internal_to_reported(calibration, channel_max=channel_max)
    return CalibProduct(
        format_version=SCHEMA_VERSION,
        deposition_to_channel=Histogram2D(
            x=channel_axis(int(n_channels)),
            y=energy_axis(energy_edges, name=DEPOSITION_AXIS_NAME),
            values=np.asarray(response.matrix.toarray(), dtype=np.float64),
            variances=None,
        ),
        param_cov=Histogram2D(
            x=reported_parameter_axis(),
            y=reported_parameter_axis(),
            values=np.asarray(covariance, dtype=np.float64),
            variances=None,
        ),
        params_reported=(
            float(reported.c0),
            float(reported.c1),
            float(reported.c2),
            float(reported.c3),
        ),
        resol_params=(float(resol[0]), float(resol[1]), float(resol[2])),
        channel_max=float(channel_max),
        provenance=provenance,
        chi2=float(chi2),
        dof=int(dof),
        covariance_scale=float(covariance_scale),
        fit_status=str(fit_status),
        scales=scales,
        scale_bound_flags=scale_bound_flags,
        resol_clamp_count=clamp_count,
        resol_clamp_energy_low_kev=clamp_low,
        resol_clamp_energy_high_kev=clamp_high,
    )


__all__ = [
    "DEPOSITION_AXIS_NAME",
    "build_calib_product",
    "channel_derived_edges_kev",
]
