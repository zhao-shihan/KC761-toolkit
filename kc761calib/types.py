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
    weights: np.ndarray  # total per-bin 1-sigma uncertainty (stat + sys)
    model_counts: np.ndarray  # resolution-smeared sim, unscaled (used bins)
    bin_centers: np.ndarray  # energy centers of the used bins (keV)


@dataclass
class DatasetDetail:
    """Diagnostics for one dataset on its channel-range fit binning."""

    label: str
    channel_low: int  # first selected channel (0-based, inclusive)
    channel_high: int  # last selected channel (0-based, inclusive)
    scale_lo: float  # fixed lower ref energy for the scale (keV)
    scale_hi: float  # fixed upper ref energy for the scale (keV)
    bin_centers: np.ndarray  # energy centers of the used bins (keV)
    data_counts: np.ndarray  # background-subtracted counts per used bin
    data_errors: np.ndarray  # total per-bin uncertainty (stat + sys)
    model_prediction: np.ndarray  # best-fit, scaled smeared simulation
    smeared_sim: np.ndarray  # resolution-smeared simulation, unscaled
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
    calib_params: np.ndarray  # calibration parameters (c0, k1, k2, k3)
    resol_params: np.ndarray  # resolution parameters (b0..b5: sigma, tau)
    scale_params: np.ndarray  # per-dataset linear scale (s0,s1)
    channel_max: float = 0.0  # largest data channel edge across models
    n_channel_bins: int = 0  # smallest data channel count across models
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
    model: object = None
    detail: FitDetail | None = None

    @property
    def scales(self) -> np.ndarray:
        return self.detail.scale_params

    @property
    def scale_errors(self) -> np.ndarray:
        tail = slice(len(self.params) - len(self.detail.scale_params),
                     len(self.params))
        return self.errors[tail]

    @property
    def chi2_per_dataset(self) -> np.ndarray:
        return np.array([ds.chi2 for ds in self.detail.datasets], dtype=float)
