"""Typed evaluation and fit-result containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetDetail:
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
    label: str = ""
    elow: float = 0.0
    ehigh: float = 0.0


@dataclass
class FitDetail:
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
        if not self.datasets:
            return None
        return np.concatenate([ds.mask for ds in self.datasets])


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
