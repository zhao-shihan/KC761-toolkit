"""Flat fit-vector layout q = [*channels(4), *resolutions(3), *scales(n)].

This module is the single source of truth for parameter names, bounds and
slices; the anchor energies themselves live in calibration/resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .response import BOUNDS_R, CALIB_ENERGIES, INIT_R, INIT_X, RESOL_ENERGIES

DEFAULT_SYS_FRAC = 0.05

BOUNDS_S = [(1e-3, 1e3)]

N_CHANNELS = len(CALIB_ENERGIES)
N_RESOLUTIONS = len(RESOL_ENERGIES)
N_CORE = N_CHANNELS + N_RESOLUTIONS

CHANNELS = slice(0, N_CHANNELS)
RESOLUTIONS = slice(N_CHANNELS, N_CORE)

PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_B = ["b0", "b1", "b2"]

_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_name(label: str, index: int) -> str:
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_")
    return f"s_{clean}" if clean else f"s{index}"


def anchor_names() -> list[str]:
    return ([f"x{e:g}" for e in CALIB_ENERGIES]
            + [f"r{e:g}" for e in RESOL_ENERGIES])


@dataclass(frozen=True)
class Space:
    """Layout bookkeeping for one fit: scale labels plus the channel bound.

    Channel anchors are bounded to the shared data-channel range so that a
    monotonically increasing calibration cannot run off the detector axis.
    """

    scale_labels: tuple[str, ...]
    channel_bound: float

    @property
    def n_scales(self) -> int:
        return len(self.scale_labels)

    @property
    def size(self) -> int:
        return N_CORE + self.n_scales

    @property
    def scales(self) -> slice:
        return slice(N_CORE, self.size)

    @property
    def names(self) -> list[str]:
        return (anchor_names()
                + [scale_name(l, i) for i, l in enumerate(self.scale_labels)])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return ([(0.0, float(self.channel_bound))] * N_CHANNELS
                + BOUNDS_R + BOUNDS_S * self.n_scales)

    def x0(self, init_scales: np.ndarray | list[float]) -> np.ndarray:
        return np.concatenate([INIT_X, INIT_R,
                               np.asarray(init_scales, dtype=float)])


_MISSING = object()


def broadcast(value, n: int, name: str, default=_MISSING):
    """Expand None/scalar/1-or-n sequence into a length-n list."""
    if value is None:
        if default is _MISSING:
            raise ValueError(f"{name}: value is None (expected a scalar or "
                             f"sequence of length 1 or {n})")
        return [default] * n
    if np.isscalar(value) or isinstance(value, (str, bytes)):
        return [value] * n
    seq = list(value)
    if len(seq) == n:
        return seq
    if len(seq) == 1:
        return seq * n
    raise ValueError(f"{name}: expected 1 or {n} values, got {len(seq)}")
