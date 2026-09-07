"""Typed containers for kc761unfold.

``CalibrationFile`` is the validated snapshot of a kc761calib export or
a kc761sim composite-response file; ``UnfoldSettings`` the
regularization/error-model configuration; and ``UnfoldResult`` the
single result structure consumed by the export and plot paths in both
the unfold and calibration-only modes, so the two modes share one
output pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

DEFAULT_SYST_FRAC = 0.10
DEFAULT_ALPHA = 0.1
DEFAULT_RESOL_FRAC = 0.5
DEFAULT_MASK_P0 = 5.0
DEFAULT_MASK_FLOOR = 0.05
DEFAULT_K = 1
DEFAULT_SNIP_ITER = 24


@dataclass
class CalibrationFile:
    """Validated kc761calib export or kc761sim composite-response file.

    ``matrix`` is the thresholded response on the selected channel
    subrange ``[channel_low, channel_high]``: entry ``[i, j]`` is the
    probability that a count in energy bin ``channel_low + j`` is
    detected in channel bin ``channel_low + i``.  The energy axis is the
    energy-deposition axis for kc761calib exports (the calibration image
    of the channel bins) and the true primary-gamma-energy axis for
    kc761sim matrix-mode composite files; both are read identically.
    ``energy_edges``/``centers``/``widths`` are the corresponding slice
    of the full energy binning.  ``param_cov`` is the 7x7 covariance of
    ``(c0, c1, c2, c3, b0, b1, b2)`` in the reported basis.
    """

    n_channels: int  # full channel count of the calibration file
    channel_low: int
    channel_high: int
    energy_edges_full: np.ndarray  # full range, n_channels + 1
    energy_edges: np.ndarray  # subrange slice, n_bins + 1
    centers: np.ndarray  # subrange, n_bins
    widths: np.ndarray  # subrange, n_bins
    matrix: sparse.csr_matrix  # subrange n_bins x n_bins
    #: Composite files only: conditional transport p_tilde[dep, primary]
    #: (thresholded CSR); None for calibration files.  Used by the
    #: systematic-error propagation to rebuild the response's parameter
    #: dependence, R(q) = C(q) p_tilde diag(eta).
    transport: sparse.csr_matrix | None
    calib_coeffs: np.ndarray  # (c0, c1, c2, c3)
    resol_params: np.ndarray  # (b0, b1, b2)
    param_cov: np.ndarray  # 7x7 reported basis

    @property
    def n_bins(self) -> int:
        return self.channel_high - self.channel_low + 1


@dataclass
class UnfoldSettings:
    """Hybrid regularized unfolding configuration; every field is CLI-exposed.

    The working range is given by the energy window ``energy_low`` ..
    ``energy_high`` (keV); ``channel_low``/``channel_high`` are the
    derived channel bins (the bins whose centers fall inside the energy
    window).  ``alpha`` is the dimensionless regularization strength.
    ``resol_frac`` is the resolution-fraction of the presentation floor:
    the unfolded spectrum is parameterized as mu = S nu with S a
    Gaussian smoother of width ``resol_frac`` times the analytic
    resolution model, so presented features are at least
    ``resol_frac * sigma`` wide (0 disables the floor and unfolds mu
    directly).
    """

    alpha: float = DEFAULT_ALPHA  # fixed regularization strength
    resol_frac: float = DEFAULT_RESOL_FRAC  # 0 = no resolution floor
    mask_p0: float = DEFAULT_MASK_P0  # SNIP peak-significance threshold
    mask_floor: float = DEFAULT_MASK_FLOOR  # minimum regularization on peaks
    k: int = DEFAULT_K  # difference order of the density penalty
    snip_iter: int = DEFAULT_SNIP_ITER  # SNIP clipping iterations
    syst_frac: float = DEFAULT_SYST_FRAC  # fractional data-side systematic
    energy_low: float = 0.0  # requested lower energy bound (keV)
    energy_high: float = 0.0  # requested upper energy bound (keV)
    channel_low: int = 0  # derived lower channel bin
    channel_high: int = 0  # derived upper channel bin


@dataclass
class UnfoldResult:
    """One output spectrum: the single structure behind ROOT export/plot.

    Unfold mode (``calib_only=False``): ``counts`` is the unfolded
    mu_hat with the analytic per-bin total/statistical/systematic
    errors and the full n x n covariance pair; ``refolded`` carries the
    refolded prediction for the residual panel.  Calibration-only mode
    (``calib_only=True``): ``counts`` are the raw counts relabeled onto
    the energy axis; ``sigma_stat`` is the stored subtraction error,
    ``sigma_syst`` combines the fractional systematic with the
    calibration-model error propagated vertically through the spectrum
    derivative (``sigma_calib``); covariance matrices and diagnostics
    are None.
    """

    calib_only: bool
    n_bins: int
    channel_low: int
    channel_high: int
    energy_edges: np.ndarray  # n_bins + 1
    centers: np.ndarray  # n_bins
    counts: np.ndarray  # n_bins
    sigma_total: np.ndarray  # n_bins, total-error band
    sigma_stat: np.ndarray  # n_bins, statistical part
    sigma_syst: np.ndarray  # n_bins, systematic band
    sigma_calib: np.ndarray  # n_bins, calibration-model vertical term
    stat_cov: np.ndarray | None  # n_bins x n_bins
    syst_cov: np.ndarray | None  # n_bins x n_bins
    refolded: np.ndarray | None  # n_bins
    chi2: float | None
    ndof: int | None
    pen_cost: float | None
    n_iter: int | None
    data_counts: np.ndarray  # n_bins, the calibrated-spectrum layer
    data_sigma_total: np.ndarray  # n_bins, total-error band of that layer
    data_sigma_syst: np.ndarray  # n_bins, systematic band of that layer
    settings: UnfoldSettings
