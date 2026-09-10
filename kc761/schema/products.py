"""Product contracts: containers, object names and provenance.

The four product kinds and their object names are frozen here and described in
docs/formats.md. Changes go through the contract-change process (AGENTS.md).

Node classes hold numpy arrays and axes only; validation and IO live in
``kc761.schema.io`` (W2). Provenance is complete by contract (D-18): git
revision, dependency versions, input hashes and the full CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kc761.schema.axes import Axis

SCHEMA_VERSION = 1
META_NTUPLE_NAME = "meta"
"""Metadata object name; it is an RNTuple (D-12 as revised 2026-09-10)."""

OBJECT_NAMES: dict[str, tuple[str, ...]] = {
    "calib": ("deposition_to_channel", "param_cov", META_NTUPLE_NAME),
    "sim": ("primary_to_deposition", "primary_column_totals", META_NTUPLE_NAME),
    "compose": (
        "response_matrix",
        "deposition_to_channel",
        "primary_to_deposition",
        "primary_column_totals",
        "primary_efficiency",
        META_NTUPLE_NAME,
    ),
    "unfold": (
        "kc761_spectrum_unfolded",
        "sigma_statistical",
        "sigma_systematic",
        "sigma_total",
        "kc761_spectrum_refolded",
        META_NTUPLE_NAME,
    ),
    "unfold_calib_only": ("kc761_spectrum_calibrated", META_NTUPLE_NAME),
    "spectrum": ("kc761_spectrum", META_NTUPLE_NAME),
}


@dataclass(frozen=True)
class InputFingerprint:
    """One input file and its sha256 digest (D-18)."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Provenance:
    """Complete provenance record written into every ``meta`` tree (D-18)."""

    created_utc: str
    producer: str
    command: str
    arguments: tuple[tuple[str, str], ...]
    git_revision: str
    git_dirty: bool
    python_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    inputs: tuple[InputFingerprint, ...]


@dataclass(frozen=True)
class Histogram1D:
    """One-dimensional histogram: axis, values and optional variances."""

    axis: Axis
    values: NDArray[np.float64]
    variances: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class Histogram2D:
    """Two-dimensional histogram with x = output side, y = input side (D-20)."""

    x: Axis
    y: Axis
    values: NDArray[np.float64]
    variances: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class CalibProduct:
    """Calibration export: response matrix, covariance and reported parameters."""

    format_version: int
    deposition_to_channel: Histogram2D
    param_cov: Histogram2D
    params_reported: tuple[float, float, float, float]
    resol_params: tuple[float, float, float]
    channel_max: float
    provenance: Provenance


@dataclass(frozen=True)
class SimProduct:
    """Matrix-mode simulation export: counts plus per-column totals (D-35/D-36)."""

    format_version: int
    primary_to_deposition: Histogram2D
    primary_column_totals: Histogram1D
    mode: str
    seed: int
    n_events: int
    workers: int
    provenance: Provenance


@dataclass(frozen=True)
class ComposeProduct:
    """Inspection artifact for ``R = C . p_tilde . diag(eta)`` (D-43)."""

    format_version: int
    response_matrix: Histogram2D
    deposition_to_channel: Histogram2D
    primary_to_deposition: Histogram2D
    primary_column_totals: Histogram1D
    primary_efficiency: Histogram1D
    provenance: Provenance


@dataclass(frozen=True)
class UnfoldProduct:
    """Unfold (or calibration-only) export with strictly split bands (D-50)."""

    format_version: int
    mode: str  # "unfold" or "calib_only"
    spectrum: Histogram1D
    sigma_statistical: Histogram1D | None
    sigma_systematic: Histogram1D | None
    sigma_total: Histogram1D | None
    refolded: Histogram1D | None
    settings: tuple[tuple[str, str], ...]
    provenance: Provenance


Product = CalibProduct | SimProduct | ComposeProduct | UnfoldProduct
