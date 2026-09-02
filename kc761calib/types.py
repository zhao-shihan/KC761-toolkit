"""Typed evaluation and fit-result containers.

``FitDetail`` is the single source of truth for a fitted state; ``FitResult``
exposes it directly and derives convenience views (per-dataset chi2, scale
values and errors) as properties instead of storing copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetArrays:
    """Per-dataset arrays shared by the residual/chi2/detail evaluation paths."""

    data_counts: np.ndarray  # background-subtracted counts on the used bins
    total_errors: np.ndarray  # total per-bin 1-sigma uncertainty (stat + sys)
    model_counts: np.ndarray  # resolution-smeared sim, unscaled (used bins)
    bin_centers: np.ndarray  # energy centers of the used bins (keV)
    scale_lo: float  # lower scale ref energy at the current calibration (keV)
    scale_hi: float  # upper scale ref energy at the current calibration (keV)


@dataclass
class DatasetDetail:
    """Diagnostics for one dataset on its channel-range fit binning."""

    label: str
    channel_low: int  # first selected channel (0-based, inclusive)
    channel_high: int  # last selected channel (0-based, inclusive)
    scale_lo: float  # lower ref energy for the scale at the fitted calibration (keV)
    scale_hi: float  # upper ref energy for the scale at the fitted calibration (keV)
    bin_centers: np.ndarray  # energy centers of the used bins (keV)
    data_counts: np.ndarray  # background-subtracted counts per used bin
    total_errors: np.ndarray  # total per-bin uncertainty (stat + sys)
    model_prediction: np.ndarray  # best-fit, scaled smeared simulation
    unsmeared_sim: np.ndarray  # rebinned sim on the bins, unscaled
    scale_params: np.ndarray  # scale at the lower/upper ref energy
    chi2: float
    n_bins: int
    bin_edges: np.ndarray  # energy edges of the full selected range


@dataclass
class FitDetail:
    datasets: list[DatasetDetail]
    chi2: float
    ndof: int
    scale_params: np.ndarray  # per-dataset linear scale (s0,s1)
    channel_max: float = 0.0  # largest data channel edge across models
    valid: bool = True

    @property
    def bins_per_dataset(self) -> np.ndarray:
        return np.array([ds.n_bins for ds in self.datasets], dtype=int)


@dataclass
class FitResult:
    success: bool
    message: str
    nfev: int
    params: np.ndarray
    errors: np.ndarray
    names: list[str]
    chi2: float
    ndof: int
    reduced_chi2: float
    cov: np.ndarray
    calib_params: np.ndarray
    calib_errors: np.ndarray
    calib_cov: np.ndarray
    resol_params: np.ndarray
    resol_errors: np.ndarray
    resol_cov: np.ndarray
    detail: FitDetail | None = None

    @property
    def scales(self) -> np.ndarray:
        return self.detail.scale_params

    @property
    def scale_errors(self) -> np.ndarray:
        scale_slice = slice(len(self.params) - len(self.detail.scale_params),
                            len(self.params))
        return self.errors[scale_slice]

    @property
    def chi2_per_dataset(self) -> np.ndarray:
        return np.array([ds.chi2 for ds in self.detail.datasets], dtype=float)
