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
    """Per-dataset arrays shared by the residual/chi2/detail evaluation paths.

    ``data_*``/``mc_errors``/``model_counts`` cover the usable bins (the
    frozen ``usable_mask`` or an explicit mask).
    """

    data_counts: np.ndarray  # background-subtracted counts on the used bins
    data_errors: np.ndarray  # data-side per-bin 1-sigma uncertainty (stat + sys)
    mc_errors: np.ndarray  # MC statistical 1-sigma of the unscaled smeared sim (used bins)
    model_counts: np.ndarray  # resolution-smeared sim per channel bin, unscaled (used bins)
    bin_centers: np.ndarray  # energy positions of the used channel bins (keV)
    channel_centers: np.ndarray  # channel numbers of the used bins


@dataclass
class DatasetDetail:
    """Diagnostics for one dataset on its channel-range fit binning.

    ``data_*``/``mc_errors``/``model_errors``/``combined_errors`` and
    ``model_prediction`` cover the usable bins; ``unsmeared_sim`` and
    ``unsmeared_sim_errors`` cover the full selected channel range
    (``bin_edges`` binning).
    """

    label: str
    channel_low: int  # first selected channel (0-based, inclusive)
    channel_high: int  # last selected channel (0-based, inclusive)
    bin_centers: np.ndarray  # energy positions of the used channel bins (keV)
    data_counts: np.ndarray  # background-subtracted counts per used bin
    data_errors: np.ndarray  # data-side per-bin uncertainty (stat + sys)
    mc_errors: np.ndarray  # MC statistical sigma of the unscaled smeared model (used bins)
    model_errors: np.ndarray  # MC statistical sigma of the scaled model prediction (used bins)
    combined_errors: np.ndarray  # per-bin sigma used in the chi2 pulls (stat + sys + model MC)
    model_prediction: np.ndarray  # best-fit, scaled smeared sim per channel bin
    unsmeared_sim: np.ndarray  # rebinned sim on the true-energy bins, unscaled (full range)
    unsmeared_sim_errors: np.ndarray  # MC statistical sigma of the rebinned sim, unscaled (full range)
    scale_params: np.ndarray  # (s0, s1, s2, s3) quadratic-Bezier scale
    chi2: float
    n_bins: int
    bin_edges: np.ndarray  # energy edges of the full selected range


@dataclass
class FitDetail:
    datasets: list[DatasetDetail]
    chi2: float
    ndof: int
    scale_params: np.ndarray  # per-dataset quadratic-Bezier scale (s0..s3)
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
