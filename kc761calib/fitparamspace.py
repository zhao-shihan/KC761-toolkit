"""Flat fit-vector parameter space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .response import (BOUNDS_CALIB, BOUNDS_RESOL, INIT_CALIB, INIT_RESOL,
                       N_CALIB, N_RESOL, PARAM_NAMES_CORE)
from .scaling import N_SCALE, scale_bounds, scale_names

N_CORE = N_CALIB + N_RESOL
CORE = slice(0, N_CORE)  # (c0, k1, k2, k3, b0, b1, b2), the shared core block
CALIB = slice(0, N_CALIB)
CALIB_K = slice(1, N_CALIB)  # k1, k2, k3 (the slope parameters, excluding c0)
RESOL = slice(N_CALIB, N_CORE)


@dataclass(frozen=True)
class FitParamSpace:
    scale_labels: tuple[str, ...]
    init_scales: tuple[float, ...]
    channel_windows: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not (len(self.init_scales) == len(self.scale_labels)
                == len(self.channel_windows)):
            raise ValueError(
                "scale_labels, init_scales and channel_windows must provide "
                "one entry per dataset "
                f"({len(self.scale_labels)} labels, {len(self.init_scales)} "
                f"scales, {len(self.channel_windows)} windows)")

    @property
    def n_datasets(self) -> int:
        return len(self.scale_labels)

    @property
    def size(self) -> int:
        return N_CORE + N_SCALE * self.n_datasets

    @property
    def scale_block(self) -> slice:
        return slice(N_CORE, self.size)

    def scale(self, i: int) -> slice:
        return slice(N_CORE + N_SCALE * i, N_CORE + N_SCALE * (i + 1))

    @property
    def names(self) -> list[str]:
        return (PARAM_NAMES_CORE
                + [n for i, l in enumerate(self.scale_labels)
                   for n in scale_names(l, i)])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        bounds = [*BOUNDS_CALIB, *BOUNDS_RESOL]
        for i in range(self.n_datasets):
            ch_lo, ch_hi = self.channel_windows[i]
            bounds.extend(scale_bounds(
                float(self.init_scales[i]), ch_lo, ch_hi))
        return bounds

    def x0(self) -> np.ndarray:
        # Flat start: s1 = s2 = s3 = initial_scale (constant scale across the
        # window) with the control channel s0 at the window midpoint.
        scale_blocks = [
            [0.5 * (ch_lo + ch_hi)] + [float(s)] * (N_SCALE - 1)
            for s, (ch_lo, ch_hi) in zip(self.init_scales, self.channel_windows)]
        return np.concatenate([INIT_CALIB, INIT_RESOL,
                               np.asarray(scale_blocks, dtype=float).ravel()])
