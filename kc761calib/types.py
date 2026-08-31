"""Typed evaluation and fit-result containers.

``FitDetail`` is the single source of truth for a fitted state; ``FitResult``
exposes it directly and derives convenience views (per-dataset chi2, scale
values and errors) as properties instead of storing copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetDetail:
    """Diagnostics for one dataset on its fit binning."""

    label: str
    elow: float                # lower fit-window bound (keV)
    ehigh: float               # upper fit-window bound (keV)
    bin_centers: np.ndarray    # centers of the fitted bins (keV)
    bin_counts: np.ndarray     # rebinned, background-subtracted counts per bin
    sigma: np.ndarray          # total per-bin uncertainty (stat + sys)
    smeared_model: np.ndarray  # best-fit, scaled smeared simulation
    smeared_sim: np.ndarray    # resolution-smeared simulation, unscaled
    raw_sim: np.ndarray  # perfect-resolution sim on the bins, unscaled
    scale_params: np.ndarray   # (s0, s1, s2, s3) scale-curve values
    chi2: float
    n_bins: int
    bin_edges: np.ndarray


@dataclass
class FitDetail:
    datasets: list[DatasetDetail]
    chi2: float
    ndof: int
    calib_params: np.ndarray      # calibration parameters (c0, p, q)
    resol_params: np.ndarray      # resolution parameters (b0, b1, b2)
    scale_params: np.ndarray      # per-dataset scale-curve (s0,s1,s2,s3)
    channel_max: float = 0.0      # largest data channel edge across models
    n_channel_bins: int = 0       # smallest data channel count across models
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
