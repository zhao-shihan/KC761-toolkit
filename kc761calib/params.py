"""Flat fit-vector layout q = [*calib(3), *resol(3), *scales(n)]."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .response import BOUNDS_RESOL, BOUNDS_CALIB, INIT_CALIB, INIT_RESOL

DEFAULT_SYS_FRAC = 0.05

BOUNDS_SCALE = [(1e-3, 1e3)]

N_CALIB = 4
N_RESOL = 3
N_CORE = N_CALIB + N_RESOL

CALIB = slice(0, N_CALIB)
CALIB_K = slice(1, N_CALIB)  # k1, k2, k3 (the slope parameters, excluding c0)
RESOL = slice(N_CALIB, N_CORE)

PARAM_NAMES_CORE = ["c0", "k1", "k2", "k3", "b0", "b1", "b2"]
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_K = ["k1", "k2", "k3"]
PARAM_NAMES_B = ["b0", "b1", "b2"]

_SCALE_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def scale_name(label: str, index: int) -> str:
    clean = _SCALE_CLEAN.sub("_", str(label)).strip("_")
    return f"s_{clean}" if clean else f"s{index}"


@dataclass(frozen=True)
class Space:
    """Layout bookkeeping for one fit: scale labels only.

    The calibration/resolution parameters have fixed bounds in response, so the
    space only has to know how many per-dataset scales there are.
    """

    scale_labels: tuple[str, ...]

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
        return (PARAM_NAMES_CORE
                + [scale_name(l, i) for i, l in enumerate(self.scale_labels)])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return BOUNDS_CALIB + BOUNDS_RESOL + BOUNDS_SCALE * self.n_scales

    def x0(self, init_scales: np.ndarray | list[float]) -> np.ndarray:
        return np.concatenate([INIT_CALIB, INIT_RESOL,
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
