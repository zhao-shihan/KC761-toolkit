"""Flat fit-vector parameter space q = [*calib(4), *resol(3), *(s0,s1,s2) per dataset].

The per-block counts and parameter names live with their block (``response`` for
calibration/resolution, ``scaling`` for the scale); this module only lays them out
in the flat vector and provides ``FitParamSpace``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .response import (BOUNDS_RESOL, BOUNDS_CALIB, INIT_CALIB, INIT_RESOL,
                       N_CALIB, N_RESOL, PARAM_NAMES_CORE)
from .scaling import BOUNDS_SCALE, N_SCALE, scale_names

N_CORE = N_CALIB + N_RESOL
CALIB = slice(0, N_CALIB)
CALIB_K = slice(1, N_CALIB)  # k1, k2, k3 (the slope parameters, excluding c0)
RESOL = slice(N_CALIB, N_CORE)


@dataclass(frozen=True)
class FitParamSpace:
    scale_labels: tuple[str, ...]
    s2_bounds: tuple[tuple[float, float], ...]

    @property
    def n_scales(self) -> int:
        return len(self.scale_labels)

    @property
    def size(self) -> int:
        return N_CORE + N_SCALE * self.n_scales

    @property
    def scales(self) -> slice:
        return slice(N_CORE, self.size)

    def scale_slice(self, i: int) -> slice:
        return slice(N_CORE + N_SCALE * i, N_CORE + N_SCALE * (i + 1))

    @property
    def names(self) -> list[str]:
        return (PARAM_NAMES_CORE
                + [n for i, l in enumerate(self.scale_labels)
                   for n in scale_names(l, i)])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        bounds = [*BOUNDS_CALIB, *BOUNDS_RESOL]
        for i in range(self.n_scales):
            bounds.extend(
                (BOUNDS_SCALE[0], BOUNDS_SCALE[1], self.s2_bounds[i]))
        return bounds

    def x0(self, init_scales: np.ndarray | list[float]) -> np.ndarray:
        scale_block = np.zeros(N_SCALE * self.n_scales)
        scale_block[::N_SCALE] = np.asarray(init_scales, dtype=float)
        midpoints = [0.5 * (lo + hi) for (lo, hi) in self.s2_bounds]
        scale_block[2::N_SCALE] = np.asarray(midpoints, dtype=float)
        return np.concatenate([INIT_CALIB, INIT_RESOL, scale_block])
