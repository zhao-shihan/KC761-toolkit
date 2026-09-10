"""Typed containers for kc761unfold.

``CalibrationFile`` is the validated snapshot of a kc761calib export;
``UnfoldSettings`` the regularization/uncertainty-model configuration;
and ``UnfoldResult`` the single result structure consumed by the export
and plot paths in both the unfold and calibration-only modes, so the two
modes share one output pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

DEFAULT_ALPHA = 0.01
DEFAULT_K = 2
DEFAULT_SNIP_ITER = 24
DEFAULT_MASK_Z0 = 3.0
DEFAULT_MASK_FLOOR = 0.05
DEFAULT_SYST_FRAC = 0.10


@dataclass
class CalibrationFile:
    """Validated snapshot of a kc761calib export on the selected channel
    subrange.

    ``to_channel`` is the thresholded deposition-to-channel matrix on
    ``[channel_low, channel_high]``; ``energy_edges``/``centers``/
    ``widths`` are the corresponding slice of the full binning;
    ``param_cov`` is the 7x7 covariance of ``(c0, c1, c2, c3, b0, b1,
    b2)`` in the reported basis.  The private cache field is
    per-snapshot bookkeeping (``init=False``), so
    :func:`dataclasses.replace` of a snapshot starts with an empty cache
    instead of inheriting results computed for the original parameters.
    """

    n_channels: int  # full channel count of the calibration file
    channel_low: int
    channel_high: int
    energy_edges_full: np.ndarray  # full range, n_channels + 1
    energy_edges: np.ndarray  # subrange slice, n_bins + 1
    centers: np.ndarray  # subrange, n_bins
    widths: np.ndarray  # subrange, n_bins
    to_channel: sparse.csr_matrix  # subrange n_bins x n_bins
    calib_coeffs: np.ndarray  # (c0, c1, c2, c3)
    resol_params: np.ndarray  # (b0, b1, b2)
    param_cov: np.ndarray  # 7x7 reported basis
    # Session-local slice cache (see kc761unfold.reader.slice_calibration),
    # keyed by (channel_low, channel_high); private bookkeeping only.
    slice_cache: dict = field(default_factory=dict, repr=False,
                               compare=False, init=False)

    @property
    def n_bins(self) -> int:
        return self.channel_high - self.channel_low + 1


@dataclass
class UnfoldSettings:
    """Hybrid regularized unfolding configuration; every field is CLI-exposed.

    The working range is the energy window ``energy_low`` .. ``energy_high``
    (keV); ``channel_low``/``channel_high`` are the derived channels
    (their centers fall inside the window).  Fields are grouped by role:
    the penalty (``alpha``, ``k``), the SNIP peak mask (``snip_iter``,
    ``mask_z0``, ``mask_floor``) and the uncertainty model
    (``syst_frac``).
    """

    # working range
    energy_low: float = 0.0  # requested lower energy bound (keV)
    energy_high: float = 0.0  # requested upper energy bound (keV)
    channel_low: int = 0  # derived lower channel
    channel_high: int = 0  # derived upper channel
    # penalty
    alpha: float = DEFAULT_ALPHA  # regularization strength
    k: int = DEFAULT_K  # difference order of the density penalty
    # window padding (in local resolution widths, clipped to the
    # detector range)
    pad_nsigma: float = 5.0
    # SNIP peak mask
    snip_iter: int = DEFAULT_SNIP_ITER  # SNIP clipping iterations
    mask_z0: float = DEFAULT_MASK_Z0  # peak-significance scale (z-score)
    mask_floor: float = DEFAULT_MASK_FLOOR  # minimum regularization on peaks
    # uncertainty model
    syst_frac: float = DEFAULT_SYST_FRAC  # fractional data-side systematic


@dataclass
class UnfoldResult:
    """One output spectrum: the single structure behind ROOT export/plot.

    Unfold mode (``calib_only=False``): ``counts`` is the unfolded mu_hat
    with the analytic per-bin total/statistical/systematic uncertainties
    and the refolded
    prediction for the residual panel.  Calibration-only mode
    (``calib_only=True``): ``counts`` are the raw counts relabeled onto
    the energy axis, ``sigma_calib`` is the calibration-model uncertainty
    propagated vertically through the spectrum derivative, and the
    bands and diagnostics are None.
    """

    calib_only: bool
    n_bins: int
    channel_low: int
    channel_high: int
    energy_edges: np.ndarray  # n_bins + 1
    centers: np.ndarray  # n_bins
    counts: np.ndarray  # n_bins
    sigma_total: np.ndarray  # n_bins, total-uncertainty band
    sigma_stat: np.ndarray  # n_bins, statistical part
    sigma_syst: np.ndarray  # n_bins, systematic band
    sigma_calib: np.ndarray  # n_bins, calibration-model vertical term
    refolded: np.ndarray | None  # n_bins
    chi2: float | None
    ndof: int | None
    pen_cost: float | None
    n_iter: int | None
    fit_sigma: np.ndarray | None  # n_bins, chi-square denominator sigma
    data_counts: np.ndarray  # n_bins, the calibrated-spectrum layer
    data_sigma_total: np.ndarray  # n_bins, total-uncertainty band of layer
    data_sigma_syst: np.ndarray  # n_bins, systematic band of that layer
    settings: UnfoldSettings
    converged: bool | None = None  # solver status (None in calib-only mode)
