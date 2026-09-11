"""Calibration fit data types.

Small frozen containers shared by the forward model, the optimizer, the
covariance extraction and the report. Formula IDs: F-CAL-1..F-CAL-5
(``docs/derivations.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kc761.core.uncertainty import DEFAULT_SYST_FRAC
from kc761.errors import ValidationError
from kc761.schema.products import CalibProduct, Histogram1D


@dataclass(frozen=True)
class DatasetSpec:
    """One data/MC pair on a shared channel range ``[low, high]`` (0-based).

    ``data`` is the measured channel spectrum (counts + ``fSumw2``); ``mc`` is
    the source-mode simulation spectrum on the fixed uniform deposition axis
    (D-33/D-101). ``data_path``/``mc_path`` are optional input paths recorded
    in the provenance of the exported product.
    """

    label: str
    data: Histogram1D
    mc: Histogram1D
    channel_low: int
    channel_high: int
    syst_frac: float = DEFAULT_SYST_FRAC
    data_path: str | None = None
    mc_path: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValidationError("dataset label must be non-empty")


@dataclass(frozen=True)
class FitSettings:
    """Fixed, documented optimizer defaults (D-107; exposed by the CLI as
    ``calib --max-iter/--tolerance`` under D-145).

    The fit uses one bounded trust-region (reflective) Gauss-Newton stage with
    the analytic residual Jacobian; ``maxiter`` caps the function evaluations.
    """

    maxiter: int = 2000
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10

    def __post_init__(self) -> None:
        if self.maxiter < 1:
            raise ValidationError(f"maxiter must be >= 1, got {self.maxiter!r}")
        for name in ("ftol", "xtol", "gtol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValidationError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True)
class FitProgress:
    """Progress event emitted by :func:`kc761.calib.fit.run_fit` (D-168).

    ``nfev == 0`` marks the pre-fit summary event; positive values are
    time-cadenced progress reports and the final event. ``ms_per_eval`` is the
    mean wall time per objective evaluation so far.
    """

    nfev: int
    chi2: float
    dof: int
    reduced_chi2: float
    elapsed_s: float
    ms_per_eval: float
    n_free: int
    n_bins: int
    n_datasets: int


@dataclass(frozen=True)
class ScaleResult:
    """One dataset's fitted quadratic-Bezier scale (F-CAL-2)."""

    label: str
    initial_scale: float
    params: tuple[float, float, float, float]


@dataclass(frozen=True)
class FitResult:
    """Outcome of one calibration fit (F-CAL-1..F-CAL-5).

    ``core_internal`` is the fit-basis vector ``(c0, k1, k2, k3, b0, b1, b2)``;
    ``param_cov`` is the reported-basis 7x7 covariance (D-105);
    ``resol_clamp_*`` summarize the F-MODEL-5 clamp on the export grid.
    """

    success: bool
    status: str
    message: str
    nfev: int
    core_internal: NDArray[np.float64]
    params_reported: tuple[float, float, float, float]
    resol_params: tuple[float, float, float]
    param_cov: NDArray[np.float64]
    scales: tuple[ScaleResult, ...]
    scale_bound_flags: tuple[tuple[str, tuple[bool, bool, bool, bool]], ...]
    chi2: float
    dof: int
    covariance_scale: float
    n_bins: int
    n_free: int
    channel_max: float
    resol_clamp_count: int
    resol_clamp_energy_low_kev: float
    resol_clamp_energy_high_kev: float
    product: CalibProduct | None = None
    product_path: Path | None = None
    plot_path: Path | None = None
    report: str = ""
