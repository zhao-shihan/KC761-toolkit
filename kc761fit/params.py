"""Fit parameter-space layout and shared fit configuration constants."""

from __future__ import annotations

import numpy as np

from .calibration import CALIB_ENERGIES, INIT_X
from .resolution import BOUNDS_R, INIT_R, RESOL_ENERGIES

DEFAULT_SYS_FRAC = 0.05

BOUNDS_S = [(1e-3, 1e3)]

PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_A = ["a0", "a1", "a2"]


class ParamBlock:
    def __init__(self, name: str, names: list[str] | tuple[str, ...],
                 init: np.ndarray | list[float],
                 bounds: list[tuple[float, float]]):
        self.name = name
        self.names = tuple(names)
        self.init = np.asarray(init, dtype=float)
        self.bounds = list(bounds)
        if not (len(self.names) == len(self.init) == len(self.bounds)):
            raise ValueError(
                f"block '{name}': names/init/bounds lengths "
                f"{len(self.names)}/{len(self.init)}/{len(self.bounds)} "
                "must match")
        self.start: int = 0

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def stop(self) -> int:
        return self.start + self.n

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


class ParameterSpace:
    def __init__(self, blocks: list[ParamBlock]):
        self.blocks = tuple(blocks)
        start = 0
        for b in self.blocks:
            b.start = start
            start += b.n
        self._size = start

    @classmethod
    def from_anchors(cls, n_channels: float) -> "ParameterSpace":
        calib_names = [f"x{e:g}" for e in CALIB_ENERGIES]
        resol_names = [f"r{e:g}" for e in RESOL_ENERGIES]
        return cls([
            ParamBlock("channels", calib_names, INIT_X,
                       [(0.0, float(n_channels))] * len(calib_names)),
            ParamBlock("resolutions", resol_names, INIT_R, BOUNDS_R),
        ])

    def with_scales(self, n: int, names: list[str] | None = None,
                    init: np.ndarray | list[float] | None = None,
                    bounds: list[tuple[float, float]] | None = None
                    ) -> "ParameterSpace":
        if n < 0:
            raise ValueError(f"n_scales must be >= 0, got {n}")
        if names is None:
            names = [f"s{i}" for i in range(n)]
        if init is None:
            init = [1.0] * n
        if bounds is None:
            bounds = BOUNDS_S * n
        return ParameterSpace([*self.blocks,
                               ParamBlock("scales", names, init, bounds)])

    @property
    def size(self) -> int:
        return self._size

    def block(self, name: str) -> ParamBlock:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"no parameter block named {name!r}")

    @property
    def names(self) -> list[str]:
        return [n for b in self.blocks for n in b.names]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [x for b in self.blocks for x in b.bounds]

    @property
    def x0(self) -> np.ndarray:
        return np.concatenate([b.init for b in self.blocks])

    @property
    def scale_start(self) -> int:
        for b in self.blocks:
            if b.name == "scales":
                return b.start
        return self.size

    @property
    def channels(self) -> slice:
        return self.block("channels").slice

    @property
    def resolutions(self) -> slice:
        return self.block("resolutions").slice

    @property
    def scales(self) -> slice:
        return self.block("scales").slice


_MISSING = object()


def broadcast(value, n: int, name: str, default=_MISSING):
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
