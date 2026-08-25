"""Reparameterisation of the calibration and resolution models.

The fitter works with well-scaled *internal* parameters so that steps,
bounds and finite differences have comparable magnitude; the reported
parameters are always the original (un-reparameterised) ones.

Calibration  E(x) = c3 x^3 + c2 x^2 + c1 x + c0   (x in channels)
------------  with u = x / N, N = number of channels (u ~ [0, 1]):

        E = b0 + b1 u + b2 u^2 + b3 u^3
        b0 = c0,  b1 = c1 N,  b2 = c2 N^2,  b3 = c3 N^3

    All internal b parameters are in keV with comparable magnitude
    (the quadratic / cubic terms contribute b2, b3 at the top of the range).

Resolution  sigma(E) = a2 E + a1 sqrt(E) + a0   (E in keV)
-----------  with E_ref a reference energy (RES_E_REF = 662 keV, the Cs-137
             line) and s = E / E_ref:

        sigma = g0 + g1 sqrt(s) + g2 s
        g0 = a0,  g1 = a1 sqrt(E_ref),  g2 = a2 E_ref

    All internal g parameters are in keV (the contributions of the sqrt
    and linear terms evaluated at E = E_ref).

Both transforms are linear maps with positive coefficients, so:
  * bounds map as [lo, hi] -> [to_internal(lo), to_internal(hi)];
  * parameter errors map by the diagonal scale factors, and
    cov_original = diag(scale) . cov_internal . diag(scale).
"""

from __future__ import annotations

import numpy as np

# Reference energy (keV) for the resolution reparameterisation: the
# Cs-137 662 keV gamma line.
RES_E_REF = 662.0


class CalibTransform:
    """Map between original calibration (c0..c3) and internal (b0..b3)."""

    def __init__(self, n_channels: int):
        self.n = float(n_channels)
        # orig_param = scale[i] * internal_param[i]
        self.scale = np.array([1.0, 1.0 / self.n, 1.0 / self.n**2, 1.0 / self.n**3])

    def to_internal(self, c) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        return np.array([c[0], c[1] * self.n, c[2] * self.n**2, c[3] * self.n**3])

    def from_internal(self, b) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return np.array([b[0], b[1] / self.n, b[2] / self.n**2, b[3] / self.n**3])


class ResTransform:
    """Map between original resolution (a0..a2) and internal (g0..g2)."""

    def __init__(self, e_ref: float):
        self.er = float(e_ref)
        self.scale = np.array([1.0, 1.0 / np.sqrt(self.er), 1.0 / self.er])

    def to_internal(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return np.array([a[0], a[1] * np.sqrt(self.er), a[2] * self.er])

    def from_internal(self, g) -> np.ndarray:
        g = np.asarray(g, dtype=float)
        return np.array([g[0], g[1] / np.sqrt(self.er), g[2] / self.er])
