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
    """Diagnostics for one dataset on its fit grid."""

    label: str
    elow: float
    ehigh: float
    mu: np.ndarray            # bin centers
    d: np.ndarray             # rebinned data counts
    err: np.ndarray           # total sigma used for weighting
    m: np.ndarray             # smeared model scaled by s
    m_raw: np.ndarray         # smeared model, unscaled
    sim_raw: np.ndarray       # unsmeared sim integrated on the grid, unscaled
    s: float
    chi2: float
    n_bins: int
    grid_edges: np.ndarray


@dataclass
class FitDetail:
    datasets: list[DatasetDetail]
    chi2: float
    ndof: int
    x: np.ndarray                 # channel anchors
    r: np.ndarray                 # resolution anchors
    s: np.ndarray                 # per-dataset scales
    c: np.ndarray                 # calibration coefficients
    b: np.ndarray                 # sigma^2 coefficients
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
    params_c: np.ndarray
    errors_c: np.ndarray
    cov_c: np.ndarray
    params_b: np.ndarray
    errors_b: np.ndarray
    cov_b: np.ndarray
    model: object = None
    detail: FitDetail | None = None

    @property
    def scales(self) -> np.ndarray:
        return self.detail.s

    @property
    def scale_errors(self) -> np.ndarray:
        tail = slice(len(self.params) - len(self.detail.s), len(self.params))
        return self.errors[tail]

    @property
    def chi2_per_dataset(self) -> np.ndarray:
        return np.array([ds.chi2 for ds in self.detail.datasets], dtype=float)
