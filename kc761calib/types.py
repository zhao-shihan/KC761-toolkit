"""Typed evaluation and fit-result containers.

``FitDetail`` is the single source of truth for a fitted state; ``FitResult``
exposes it directly and derives convenience views (per-dataset chi2, scale
values and uncertainties) as properties instead of storing copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitparamspace import CORE


@dataclass
class DatasetArrays:
    """Per-dataset arrays shared by the residual/chi2/detail evaluation paths.

    ``data_*``/``mc_*``/``model_counts`` cover the fit window bins (all
    of them; empty bins are handled by the variance floor).
    """

    data_counts: np.ndarray  # background-subtracted counts on the used bins
    stat_uncertainties: np.ndarray  # data histogram per-bin errors (sumw2 sqrt)
    data_display_uncertainties: np.ndarray  # data band 1-sigma (stat + syst·data)
    mc_uncertainties: np.ndarray  # MC 1-sigma of the unscaled folded MC spectrum
    model_counts: np.ndarray  # folded MC spectrum, unscaled (used bins)
    bin_centers: np.ndarray  # energy positions of the used channels (keV)
    channel_centers: np.ndarray  # channel numbers of the used bins


@dataclass
class DatasetDetail:
    """Diagnostics for one dataset on its channel-range fit binning.

    ``data_*``/``mc_*``/``model_mc_uncertainties``/``fit_sigma`` and
    ``model_prediction`` cover the fit window bins; ``raw_mc_counts``
    and ``raw_mc_uncertainties`` cover the same full selected channel
    range (``bin_edges`` binning).
    """

    label: str
    channel_low: int  # first selected channel (0-based, inclusive)
    channel_high: int  # last selected channel (0-based, inclusive)
    bin_centers: np.ndarray  # energy positions of the used channels (keV)
    data_counts: np.ndarray  # background-subtracted counts per used bin
    data_display_uncertainties: np.ndarray  # data band 1-sigma (stat + syst·data)
    mc_uncertainties: np.ndarray  # MC 1-sigma of the unscaled folded MC spectrum
    model_mc_uncertainties: np.ndarray  # MC 1-sigma of the scaled model
    fit_sigma: np.ndarray  # chi-square denominator (stat + syst·data + model MC)
    model_prediction: np.ndarray  # best-fit, scaled folded MC spectrum
    raw_mc_counts: np.ndarray  # rebinned MC spectrum, unscaled (full range)
    raw_mc_uncertainties: np.ndarray  # rebinned MC 1-sigma, unscaled
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
    uncertainties: np.ndarray
    names: list[str]
    chi2: float
    ndof: int
    reduced_chi2: float
    cov: np.ndarray
    calib_params: np.ndarray
    calib_uncertainties: np.ndarray
    calib_cov: np.ndarray
    resol_params: np.ndarray
    resol_uncertainties: np.ndarray
    resol_cov: np.ndarray
    detail: FitDetail | None = None
    # Fit exit state: "converged" (optimizer converged normally),
    # "stopped-early" (optimizer reported failure but the result is
    # non-degenerate), or "degenerate" (invalid coverage/chi2; the
    # fitted parameters are not meaningful).  Products are written for
    # all three states; only "converged" exits 0.
    status: str = "converged"
    # Per-core-parameter flags from the profile covariance: a side made
    # infeasible by a fit bound (one-sided crossing).  Length 7, aligned
    # with the core parameters.
    bound_limited: np.ndarray | None = None
    # Asymmetric 1-sigma widths of the 7 core parameters from the
    # dchi2 = 1 profile crossings (metadata; the covariance itself uses
    # the symmetrized widths).  Bound-side infeasibility reports that
    # side as 0.
    err_lo: np.ndarray | None = None
    err_hi: np.ndarray | None = None

    @property
    def scales(self) -> np.ndarray:
        return self.detail.scale_params

    @property
    def core_cov(self) -> np.ndarray:
        """7x7 covariance of the shared core parameters ``(c0, k1, k2, k3, b0,
        b1, b2)``, including the calib-resol cross block."""
        return np.asarray(self.cov[CORE, CORE], dtype=float)

    @property
    def chi2_per_dataset(self) -> np.ndarray:
        return np.array([ds.chi2 for ds in self.detail.datasets], dtype=float)
