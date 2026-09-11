"""Window selection and data-side fit weights (F-UNF-1/F-UNF-2).

* F-UNF-1: the requested energy window is mapped to channel rows through
  ``E(ch)`` at the channel centers and to reported primary bins through the
  primary-bin centers; the solver range is padded with the local resolution
  (F-BIN-3).
* F-UNF-2: the unfold fit weights are `sigma_fit**2 = max(stat, 1) +
  (syst_frac * data)**2`; the simulation-MC term is excluded (D-112).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761.core._checks import as_float_array, require_same_length
from kc761.core.binning import EnergyGrid, working_window
from kc761.core.model import InternalCalibration, energy_kev
from kc761.errors import ValidationError
from kc761.unfold.types import WindowSelection


def fit_sigma(
    stat_variance: NDArray[np.float64],
    data: NDArray[np.float64],
    syst_frac: float,
) -> NDArray[np.float64]:
    """Data-side unfold fit uncertainty (F-UNF-2/D-112)."""
    stat = as_float_array("stat_variance", stat_variance, ndim=1)
    counts = as_float_array("data", data, ndim=1)
    require_same_length("data/stat_variance", counts, stat)
    if not np.isfinite(syst_frac) or syst_frac < 0.0:
        raise ValidationError(f"syst_frac must be finite and >= 0, got {syst_frac!r}")
    floor = np.maximum(stat, 1.0)
    systematic = (syst_frac * counts) ** 2
    return np.sqrt(floor + systematic)


def select_window(
    *,
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    channel_max: float,
    n_channels: int,
    primary_edges_kev: NDArray[np.float64],
    energy_low_kev: float,
    energy_high_kev: float,
    pad_nsigma: float,
    strict: bool = False,
) -> WindowSelection:
    """Resolve an energy window to channel rows and reported primary bins.

    The primary axis is the simulation ``G.y`` axis; the channel map uses
    ``E(ch)`` at the channel centers. See F-UNF-1 for the exact conventions.
    """
    primary = EnergyGrid(edges_kev=np.asarray(primary_edges_kev, dtype=np.float64))
    primary_edges = primary.edges_kev
    if not np.isfinite(energy_low_kev) or not np.isfinite(energy_high_kev):
        raise ValidationError("energy window bounds must be finite")
    if not energy_low_kev < energy_high_kev:
        raise ValidationError(
            f"energy_low_kev must be < energy_high_kev, got "
            f"{energy_low_kev!r} and {energy_high_kev!r}"
        )
    if energy_low_kev < primary_edges[0] or energy_high_kev > primary_edges[-1]:
        raise ValidationError(
            f"energy window [{energy_low_kev!r}, {energy_high_kev!r}] keV is outside "
            f"the primary axis range [{primary_edges[0]!r}, {primary_edges[-1]!r}] keV"
        )

    channels = np.arange(int(n_channels), dtype=np.float64)
    channel_centers = energy_kev(channels, calibration, channel_max=channel_max)
    if not np.all(np.diff(channel_centers) > 0.0):
        raise ValidationError(
            "the calibration is not strictly increasing across the acquisition; "
            "the energy window cannot be mapped to channels"
        )
    low = int(np.searchsorted(channel_centers, energy_low_kev, side="left"))
    high = int(np.searchsorted(channel_centers, energy_high_kev, side="right")) - 1
    if low > high or low >= int(n_channels) or high < 0:
        # D-146 revision of D-111: a window entirely outside the channel energy
        # range is an error, never a silent degenerate single-channel fit.
        raise ValidationError(
            f"energy window [{energy_low_kev!r}, {energy_high_kev!r}] keV lies "
            f"entirely outside the channel energy range "
            f"[{channel_centers[0]!r}, {channel_centers[-1]!r}] keV"
        )
    channel_low = low
    channel_high = high

    solve_low, solve_high = working_window(
        channel_low,
        channel_high,
        pad_nsigma=pad_nsigma,
        calibration=calibration,
        resol_params=np.asarray(resol_params, dtype=np.float64),
        channel_max=channel_max,
        n_channels=int(n_channels),
        strict=strict,
    )

    primary_centers = 0.5 * (primary_edges[:-1] + primary_edges[1:])
    inside = (primary_centers >= energy_low_kev) & (primary_centers <= energy_high_kev)
    indices = np.flatnonzero(inside)
    if indices.size == 0:
        raise ValidationError(
            f"energy window [{energy_low_kev!r}, {energy_high_kev!r}] keV contains no "
            "primary bin center"
        )
    return WindowSelection(
        energy_low_kev=float(energy_low_kev),
        energy_high_kev=float(energy_high_kev),
        channel_low=channel_low,
        channel_high=channel_high,
        solve_low=solve_low,
        solve_high=solve_high,
        report_low=int(indices[0]),
        report_high=int(indices[-1]),
    )


__all__ = ["fit_sigma", "select_window"]
