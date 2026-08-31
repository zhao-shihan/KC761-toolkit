"""Flat fit-vector parameter space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .response import (BOUNDS_RESOL, BOUNDS_CALIB, INIT_CALIB, INIT_RESOL,
                       N_CALIB, N_RESOL, PARAM_NAMES_CORE)
from .scaling import N_SCALE, scale_bounds, scale_names

N_CORE = N_CALIB + N_RESOL
CALIB = slice(0, N_CALIB)
CALIB_K = slice(1, N_CALIB)  # k1, k2, k3 (the slope parameters, excluding c0)
RESOL = slice(N_CALIB, N_CORE)


@dataclass(frozen=True)
class FitParamSpace:
    scale_labels: tuple[str, ...]
    init_scales: tuple[float, ...]

    def __post_init__(self):
        if len(self.init_scales) != len(self.scale_labels):
            raise ValueError(
                "init_scales must provide one value per dataset "
                f"({len(self.scale_labels)} datasets, got {len(self.init_scales)})")

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
            bounds.extend(scale_bounds(float(self.init_scales[i])))
        return bounds

    def x0(self) -> np.ndarray:
        # Flat start: all four knot values equal the constant initial scale.
        scale_block = np.repeat(np.asarray(self.init_scales, dtype=float), N_SCALE)
        return np.concatenate([INIT_CALIB, INIT_RESOL, scale_block])
