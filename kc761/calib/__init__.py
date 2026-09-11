"""Calibration package.

User-facing orchestration for the energy/resolution fit, Bezier dataset
scaling, covariance estimation and calibration product export. Formula IDs
F-CAL-1..F-CAL-5 (``docs/derivations.md``). The CLI imports
:func:`kc761.calib.run_fit`.
"""

from __future__ import annotations

from kc761.calib.fit import STATUS_CONVERGED, STATUS_STOPPED, run_fit
from kc761.calib.model import (
    BOUNDS_CALIB,
    BOUNDS_RESOL,
    FIXED_DEPOSITION_BINS,
    FIXED_DEPOSITION_MAX_KEV,
    INIT_CALIB,
    INIT_RESOL,
    CalibrationModel,
    DatasetDetail,
    core_bounds,
    fixed_deposition_edges_kev,
    initial_core,
)
from kc761.calib.report import render_report
from kc761.calib.scaling import (
    N_SCALE,
    PARAM_NAMES_SCALE,
    scale_bounds,
    scale_curve,
    scale_curve_grad,
    scale_names,
)
from kc761.calib.types import (
    DEFAULT_SYST_FRAC,
    DatasetSpec,
    FitResult,
    FitSettings,
    ScaleResult,
)

__all__ = [
    "BOUNDS_CALIB",
    "BOUNDS_RESOL",
    "DEFAULT_SYST_FRAC",
    "FIXED_DEPOSITION_BINS",
    "FIXED_DEPOSITION_MAX_KEV",
    "INIT_CALIB",
    "INIT_RESOL",
    "N_SCALE",
    "PARAM_NAMES_SCALE",
    "STATUS_CONVERGED",
    "STATUS_STOPPED",
    "CalibrationModel",
    "DatasetDetail",
    "DatasetSpec",
    "FitResult",
    "FitSettings",
    "ScaleResult",
    "core_bounds",
    "fixed_deposition_edges_kev",
    "initial_core",
    "render_report",
    "run_fit",
    "scale_bounds",
    "scale_curve",
    "scale_curve_grad",
    "scale_names",
]
