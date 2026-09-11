"""Orchestration types for the unfolding pipeline.

These containers are the interface between the CLI thin wrappers and the
``kc761/unfold`` implementation. Numerics and product contracts stay in
``kc761.core`` and ``kc761.schema``; this module only groups the pieces that a
caller receives back (settings, selection, results).

Formula IDs (docs/derivations.md): F-UNF-1 .. F-UNF-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kc761.core.response import ComposedResponse, ResponseMatrix
from kc761.core.solver import (
    DEFAULT_DIFFERENCE_ORDER,
    DEFAULT_SNIP_FLOOR,
    DEFAULT_SNIP_MAX_ITERATIONS,
    DEFAULT_SNIP_PROTECT_SIGMA,
    DEFAULT_SNIP_THRESHOLD_SIGMA,
    KktCertificate,
    RegularizationSpec,
    SnipSettings,
)
from kc761.core.uncertainty import DEFAULT_SYST_FRAC
from kc761.errors import ValidationError
from kc761.schema.products import (
    ComposeProduct,
    Histogram1D,
    UnfoldProduct,
)


@dataclass(frozen=True)
class UnfoldSettings:
    """Validated unfold settings (F-UNF-1/F-UNF-2/D-112).

    ``alpha`` and ``difference_order`` build the F-SOLVE-1
    :class:`kc761.core.solver.RegularizationSpec`; the remaining fields are the
    frozen window and weight configuration. ``alpha`` is ``None`` only for the
    ``calib_only`` path, which does not solve a QP.
    """

    energy_low_kev: float
    energy_high_kev: float
    alpha: float | None
    difference_order: int = DEFAULT_DIFFERENCE_ORDER
    pad_nsigma: float = 5.0
    syst_frac: float = DEFAULT_SYST_FRAC
    snip_enabled: bool = True
    snip_threshold_sigma: float = DEFAULT_SNIP_THRESHOLD_SIGMA
    snip_protect_sigma: float = DEFAULT_SNIP_PROTECT_SIGMA
    snip_floor: float = DEFAULT_SNIP_FLOOR
    snip_iterations: int | None = None
    snip_max_iterations: int = DEFAULT_SNIP_MAX_ITERATIONS

    def __post_init__(self) -> None:
        if not np.isfinite(self.energy_low_kev) or not np.isfinite(self.energy_high_kev):
            raise ValidationError("energy window bounds must be finite")
        if not self.energy_low_kev < self.energy_high_kev:
            raise ValidationError(
                f"energy_low_kev must be < energy_high_kev, got "
                f"{self.energy_low_kev!r} and {self.energy_high_kev!r}"
            )
        if not np.isfinite(self.pad_nsigma) or self.pad_nsigma < 0.0:
            raise ValidationError(f"pad_nsigma must be finite and >= 0, got {self.pad_nsigma!r}")
        if not np.isfinite(self.syst_frac) or self.syst_frac < 0.0:
            raise ValidationError(f"syst_frac must be finite and >= 0, got {self.syst_frac!r}")
        if self.alpha is not None and (not np.isfinite(self.alpha) or self.alpha <= 0.0):
            raise ValidationError(f"alpha must be positive and finite, got {self.alpha!r}")
        if self.difference_order not in (1, 2):
            raise ValidationError(
                f"difference_order must be 1 or 2, got {self.difference_order!r}"
            )
        self.snip_settings()

    def snip_settings(self) -> SnipSettings:
        """Validated F-SOLVE-4/5 SNIP configuration (D-156/D-157)."""
        return SnipSettings(
            enabled=bool(self.snip_enabled),
            threshold_sigma=float(self.snip_threshold_sigma),
            protect_sigma=float(self.snip_protect_sigma),
            floor=float(self.snip_floor),
            iterations=self.snip_iterations,
            max_iterations=int(self.snip_max_iterations),
        )

    def regularization(self) -> RegularizationSpec:
        """Build the F-SOLVE-1 regularization spec (alpha is required)."""
        if self.alpha is None:
            raise ValidationError("calib_only settings carry no alpha to regularize with")
        return RegularizationSpec(
            alpha=self.alpha, difference_order=self.difference_order
        )


@dataclass(frozen=True)
class WindowSelection:
    """Energy window resolved to channel rows and primary bins (F-UNF-1).

    ``channel_low``/``channel_high`` are the reported data rows; ``solve_low``/
    ``solve_high`` are the padded solver rows (F-BIN-3). ``report_low``/
    ``report_high`` index the **full** primary axis (inclusive) and select the
    bins whose centers lie in the requested energy window.
    """

    energy_low_kev: float
    energy_high_kev: float
    channel_low: int
    channel_high: int
    solve_low: int
    solve_high: int
    report_low: int
    report_high: int

    @property
    def n_fit_rows(self) -> int:
        return self.solve_high - self.solve_low + 1

    @property
    def n_report_bins(self) -> int:
        return self.report_high - self.report_low + 1


@dataclass(frozen=True)
class ComposeResult:
    """Result of :func:`kc761.unfold.compose.run_compose` (F-RESP-2)."""

    response: ResponseMatrix
    composed: ComposedResponse
    efficiency: Histogram1D
    product: ComposeProduct | None
    product_path: Path | None
    calib_path: Path | None = None
    sim_path: Path | None = None


@dataclass(frozen=True)
class UnfoldResult:
    """Result of :func:`kc761.unfold.unfold.run_unfold` (F-UNF-4/F-UNF-5).

    The histogram fields hold the reported products on the reported primary
    axis; ``refolded`` and ``data_window`` are on the reported channel window.
    ``calib_only`` results carry only ``unfolded``.
    """

    mode: str
    settings: UnfoldSettings | None
    window: WindowSelection | None
    unfolded: Histogram1D | None
    sigma_statistical: Histogram1D | None
    sigma_systematic: Histogram1D | None
    sigma_total: Histogram1D | None
    refolded: Histogram1D | None
    data_window: Histogram1D | None
    chi2: float
    dof: int
    covariance_scale: float
    n_fit_rows: int
    n_active: int
    n_kept_columns: int
    n_pruned_columns: int
    certificate: KktCertificate | None
    report: str = ""
    product: UnfoldProduct | None = None
    product_path: Path | None = None
    plot_path: Path | None = None
    kept_columns: NDArray[np.int64] | None = None
    pruned_columns: NDArray[np.int64] | None = None

    @property
    def mu(self) -> NDArray[np.float64] | None:
        """Reported primary spectrum values (convenience accessor)."""
        return None if self.unfolded is None else self.unfolded.values


__all__ = [
    "ComposeResult",
    "UnfoldResult",
    "UnfoldSettings",
    "WindowSelection",
]
