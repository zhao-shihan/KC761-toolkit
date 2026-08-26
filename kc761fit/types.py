"""Typed evaluation and fit-result containers.

These dataclasses replace the ad-hoc ``detail`` dictionaries previously
returned by ``FitModel.detail`` / ``GlobalFitModel.detail``.  The field names
and value semantics match the old dictionaries, so the consumers (plotting,
the CLI, the fitter) only changed their access style
(``det["s"]`` -> ``det.s``).

A fit is always N datasets (a single-dataset fit has N = 1): ``FitDetail``
and ``FitResult`` carry per-dataset arrays, so downstream code does not
branch on the single-vs-global distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetDetail:
    """Masked evaluation of one dataset of a (possibly global) fit.

    ``d`` / ``err`` are the calibrated, exactly-rebinned data counts and
    errors; ``m_raw`` the un-scaled resolution-smeared simulation; ``m`` the
    scaled model (``s * m_raw``); ``mu`` the grid-bin centers.  All arrays are
    masked to the bins with positive error (``mask``).  ``grid_edges`` and
    ``m_raw_unsmeared`` (raw, un-convolved simulation on the grid) are carried
    so the plotting layer does not need to reach into the model internals.
    """

    d: np.ndarray
    err: np.ndarray
    m_raw: np.ndarray
    m: np.ndarray
    s: float
    mu: np.ndarray
    chi2: float
    n_bins: int
    mask: np.ndarray
    grid_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    m_raw_unsmeared: np.ndarray = field(default_factory=lambda: np.array([]))
    label: str = ""
    elow: float = 0.0
    ehigh: float = 0.0
    model: object = None


@dataclass
class FitDetail:
    """Masked evaluation of a fit model at one parameter point.

    ``datasets`` holds one :class:`DatasetDetail` per dataset (exactly one for
    a single-dataset fit); ``s`` is the per-dataset scale array.  ``chi2`` is
    the total data chi^2, ``pen`` the soft monotonicity penalty (zero for a
    physically ordered point), ``ndof`` the degrees of freedom.  ``valid``
    marks a degenerate point (insufficient data coverage): ``chi2`` is
    ``inf`` and ``datasets`` is empty.
    """

    chi2: float
    ndof: int
    pen: float
    c: np.ndarray
    a: np.ndarray
    x: np.ndarray
    r: np.ndarray
    s: np.ndarray
    datasets: list[DatasetDetail] = field(default_factory=list)
    chi2_per_dataset: np.ndarray = field(default_factory=lambda: np.array([]))
    bins_per_dataset: np.ndarray = field(default_factory=lambda: np.array([]))
    valid: bool = True

    @property
    def mask(self) -> np.ndarray | None:
        """Concatenated bin mask over all datasets (None when degenerate)."""
        if not self.datasets:
            return None
        return np.concatenate([ds.mask for ds in self.datasets])


@dataclass
class FitResult:
    """Result of a completed fit (single- or multi-dataset).

    ``params`` / ``errors`` / ``names`` are in the fit parameter space
    ``[x60..x2614, r60..r2614, s0..s_{N-1}]``; ``params_c`` / ``params_a`` are
    the derived calibration (c0..c3) and resolution (a0..a2) coefficients with
    their errors and covariances propagated through the linear maps.
    ``scales`` / ``scale_errors`` / ``chi2_per_dataset`` / ``bins_per_dataset``
    hold the per-dataset values (length 1 for a single-dataset fit).  ``model``
    is the (final, rebuilt) model whose grid the result was evaluated on;
    ``detail`` the masked evaluation at the fitted parameters.
    """

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
    params_c: np.ndarray | None = None
    errors_c: np.ndarray | None = None
    cov_c: np.ndarray | None = None
    params_a: np.ndarray | None = None
    errors_a: np.ndarray | None = None
    cov_a: np.ndarray | None = None
    model: object = None
    detail: FitDetail | None = None
    scales: np.ndarray | None = None
    scale_errors: np.ndarray | None = None
    chi2_per_dataset: np.ndarray | None = None
    bins_per_dataset: np.ndarray | None = None
