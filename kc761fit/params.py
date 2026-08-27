"""Fit parameter-space layout and shared fit configuration constants.

The optimizer works on a flat parameter vector

    q = [x60..x2614, r60..r2614, s0..s_{N-1}]

(channel positions of the calibration lines, relative resolutions, and the
normalization scale(s)).  This module owns that layout: the parameter blocks,
their names / initial values / bounds, and the derived flat-vector accessors.
It replaces the previous module-level ``IDX_CHANNELS`` / ``IDX_RESOL`` /
``IDX_SCALE`` slices, whose meaning differed between the single- and
multi-dataset models.

A :class:`ParameterSpace` is built with :meth:`ParameterSpace.from_anchors`
(the 4 calibration channels + 3 relative resolutions of the reference lines)
and extended per model with :meth:`ParameterSpace.with_scales`.
"""

from __future__ import annotations

import numpy as np

from .calibration import CALIB_ENERGIES, INIT_X
from .resolution import BOUNDS_R, INIT_R, RESOL_ENERGIES

# Default per-bin fractional systematic error (dimensionless): 5%.  Added in
# quadrature to the statistical errors proportional to the bin counts, so the
# chi^2 weights are not dominated by the highest-statistics peaks alone.
# CLI ``--sys`` overrides it per run.
DEFAULT_SYS_FRAC = 0.05

# Bounds of the normalization scale (dimensionless).
BOUNDS_S = [(1e-3, 1e3)]

# Names of the derived coefficients reported on output (from the fitted
# channels / resolutions).
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_A = ["a0", "a1", "a2"]


class ParamBlock:
    """One contiguous group of fit parameters (e.g. the 4 calibration
    channels): names, initial values and box bounds."""

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
        # Flat-vector index where the block starts; assigned by
        # ParameterSpace.__init__.
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
    """Ordered list of parameter blocks with flat-vector layout.

    Blocks are laid out in order (channels, resolutions, scales).  Each block
    knows its ``start`` / ``stop`` / ``slice`` into the flat parameter vector.
    """

    def __init__(self, blocks: list[ParamBlock]):
        self.blocks = tuple(blocks)
        start = 0
        for b in self.blocks:
            b.start = start
            start += b.n
        self._size = start

    # -- construction ------------------------------------------------------
    @classmethod
    def from_anchors(cls, n_channels: float) -> "ParameterSpace":
        """Channels (4 reference-line positions) + resolutions (3 relative
        widths), with the constants from calibration.py / resolution.py.

        ``n_channels`` is the per-channel upper fit bound (the data's channel
        count); the resolution block uses ``BOUNDS_R``.
        """
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
        """New space with a scale block of ``n`` parameters appended."""
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

    # -- layout ------------------------------------------------------------
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
        """Flat-vector index of the first scale parameter (or ``size`` when
        the space has no scale block)."""
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
    """Broadcast ``value`` (scalar / length-1 / length-n) to length ``n``.

    A ``None`` ``value`` is replaced by ``default`` (``* n``) when provided,
    otherwise it is an error.  A wrong-length sequence raises ValueError.
    """
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
