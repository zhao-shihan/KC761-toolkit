"""Channel and energy grids, and the padded working window.

Formula IDs (docs/derivations.md): F-BIN-1 .. F-BIN-4.

Frozen decisions (docs/plan.md):

* F-BIN-1: the channel axis is uniform with edges ``-0.5 .. n - 0.5``; energy
  axes are variable, strictly increasing, in keV.
* F-BIN-2 (revised, D-79): the response is evaluated over the full deposition
  axis and the requested channel rows; there is **no** parameter-dependent bin
  selection. The support taper makes contributions beyond ``n_sigma`` exactly
  zero, so chi2 is continuous in the fit parameters by construction and no
  frozen bin subset is needed.
* F-BIN-3: the solver works on ``[chlo - pad, chhi + pad]`` and only
  ``[chlo, chhi]`` is reported (D-44). The pad is ``pad_nsigma`` local
  resolution widths converted to channels with the calibrated local slope.
* D-52: ``MAX_CHANNELS`` is the validated support limit; larger inputs fail
  fast instead of being silently attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761tool.core._checks import as_float_array, require_positive
from kc761tool.core.model import (
    InternalCalibration,
    energy_kev,
    resolution_sigma_kev,
)
from kc761tool.errors import ValidationError

MAX_CHANNELS = 4096
"""Validated support limit (D-52); it may be lowered with a measured justification."""


def dense_matrix_bytes(n_channels: int) -> int:
    """Bytes of one dense ``n x n`` float64 matrix (D-52 memory estimate)."""
    size = int(n_channels)
    return size * size * 8


def channels_limit_message(n_channels: int) -> str:
    """Fail-fast message for an over-limit channel count, with a footprint.

    D-52 requires the rejection to carry a memory estimate instead of failing
    later with a bare allocation error.
    """
    gib = dense_matrix_bytes(n_channels) / 2**30
    return (
        f"n_channels={int(n_channels)} exceeds the supported maximum {MAX_CHANNELS} "
        f"(D-52); one dense {int(n_channels)}x{int(n_channels)} float64 matrix "
        f"alone needs {gib:.2f} GiB"
    )


SOURCE_MODE_DEPOSITION_MAX_KEV: Final = 4096.0
SOURCE_MODE_DEPOSITION_BINS: Final = 4096
"""Fixed uniform *source-mode* Monte-Carlo axis: ``0..4096 keV / 4096 bins``.

This is the axis of the source-mode pulse spectrum (D-33/D-120) and of the
parameter-independent fit-time ``C_fit`` (D-101). The matrix-mode primary and
deposition axes are the calibration product's ``C.y`` (D-121 revised), so this
axis is *not* used by the matrix ``G``. It has a single implementation here;
``calib.model`` re-exports the former ``FIXED_DEPOSITION_*`` names as aliases.
"""


def source_mode_deposition_edges_kev() -> NDArray[np.float64]:
    """Return the fixed uniform source-mode deposition edges in keV (F-BIN-4)."""
    return np.linspace(0.0, SOURCE_MODE_DEPOSITION_MAX_KEV, SOURCE_MODE_DEPOSITION_BINS + 1)


@dataclass(frozen=True)
class ChannelGrid:
    """Uniform channel axis ``-0.5 .. n_channels - 0.5`` (F-BIN-1)."""

    n_channels: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_channels, int):
            raise ValidationError(f"n_channels must be an int, got {self.n_channels!r}")
        if self.n_channels < 1:
            raise ValidationError(f"n_channels must be >= 1, got {self.n_channels!r}")
        if self.n_channels > MAX_CHANNELS:
            raise ValidationError(channels_limit_message(self.n_channels))

    def edges(self) -> NDArray[np.float64]:
        return np.arange(-0.5, self.n_channels + 0.5, 1.0)

    def centers(self) -> NDArray[np.float64]:
        return np.arange(0.0, float(self.n_channels), 1.0)


@dataclass(frozen=True)
class EnergyGrid:
    """Variable energy axis in keV with strictly increasing edges (F-BIN-1)."""

    edges_kev: NDArray[np.float64]

    def __post_init__(self) -> None:
        edges = as_float_array("edges_kev", self.edges_kev, ndim=1)
        if edges.size < 2:
            raise ValidationError(f"energy edges need at least 2 entries, got {edges.size}")
        if not np.all(np.diff(edges) > 0.0):
            raise ValidationError("energy edges must be strictly increasing")
        object.__setattr__(self, "edges_kev", edges)

    @property
    def n_bins(self) -> int:
        return int(self.edges_kev.size) - 1

    def centers_kev(self) -> NDArray[np.float64]:
        return 0.5 * (self.edges_kev[:-1] + self.edges_kev[1:])

    def widths_kev(self) -> NDArray[np.float64]:
        return np.diff(self.edges_kev)


def working_window(
    channel_low: int,
    channel_high: int,
    *,
    pad_nsigma: float,
    calibration: InternalCalibration,
    resol_params: NDArray[np.float64],
    channel_max: float,
    n_channels: int,
    strict: bool = False,
) -> tuple[int, int]:
    """Solver range ``[chlo - pad, chhi + pad]`` clipped to the detector (F-BIN-3).

    ``pad`` is measured in local resolution widths; the conversion to channels
    uses the calibrated local channel width at each window edge. The reported
    window stays ``[channel_low, channel_high]`` (D-44).
    """
    ChannelGrid(n_channels)
    channel_max = require_positive("channel_max", channel_max)
    if not np.isfinite(pad_nsigma) or pad_nsigma < 0.0:
        raise ValidationError(f"pad_nsigma must be finite and >= 0, got {pad_nsigma!r}")
    if not 0 <= channel_low <= channel_high < n_channels:
        raise ValidationError(
            f"window [{channel_low}, {channel_high}] outside [0, {n_channels - 1}]"
        )

    edge_channels = np.array(
        [
            channel_low - 0.5,
            channel_low + 0.5,
            channel_high - 0.5,
            channel_high + 0.5,
        ],
        dtype=np.float64,
    )
    edge_energies = energy_kev(edge_channels, calibration, channel_max=channel_max)
    width_low = float(edge_energies[1] - edge_energies[0])
    width_high = float(edge_energies[3] - edge_energies[2])
    if width_low <= 0.0 or width_high <= 0.0:
        raise ValidationError(
            "calibration is not increasing across the requested window "
            f"(widths {width_low!r}, {width_high!r})"
        )
    edge_sigma = resolution_sigma_kev(
        np.array([edge_energies[0], edge_energies[3]]), resol_params, strict=strict
    )
    pad_low = int(np.ceil(pad_nsigma * float(edge_sigma[0]) / width_low))
    pad_high = int(np.ceil(pad_nsigma * float(edge_sigma[1]) / width_high))
    low = max(0, channel_low - pad_low)
    high = min(n_channels - 1, channel_high + pad_high)
    return low, high
