"""Detector response: energy calibration E(channel) and resolution sigma^2(E).

The calibration is cubic in the channel number and is parameterized by the
intercept plus three slopes ``(c0, k1, k2, k3)``:

   E(channel) = c0 + k1 channel + (4 k2 - 3 k1 - k3)/(2 channel_max) channel^2
                 + 2 (k1 - 2 k2 + k3)/(3 channel_max^2) channel^3,
   E'(0) = k1,   E'(channel_max/2) = k2,   E'(channel_max) = k3.

Monotonicity on [0, channel_max] is guaranteed by the fit bounds (no separate
check).  Reported coefficients are the plain cubic form ``(c0, c1, c2, c3)`` with
``c1 = k1``, ``c2 = (4 k2 - 3 k1 - k3)/(2 channel_max)`` and
``c3 = 2 (k1 - 2 k2 + k3)/(3 channel_max^2)`` via the constant, invertible map
``c0k1k2k3_to_c0c1c2c3``.

Resolution is a pure Gaussian with energy-dependent standard deviation
``sigma(E)``.  ``sigma^2`` is a quadratic Bernstein polynomial in
``t = E / RESOL_E_REF`` (``RESOL_E_REF = 2000`` keV), equivalently a
quadratic Bezier curve with control values ``(b0^2, b1^2, b2^2)``:

   sigma^2(t) = (1-t)^2 b0^2 + 2(1-t)t b1^2 + t^2 b2^2

The Bernstein basis is a non-negative partition of unity only on
``t`` in [0, 1], so ``t`` is clamped at 0 before evaluation: for
``E <= 0`` the resolution saturates at ``b0``, keeping the variance
non-negative at the low-energy edge and preventing a degenerate delta
response there.  Above ``RESOL_E_REF`` the polynomial continues unclamped;
``MIN_SIGMA`` remains as numerical safety.

Folding the Gaussian response into histograms is done by the extended
binning and sparse response matrix in :mod:`kc761calib.folding`, which maps
true-energy bins to detected channel bins; its fused assembly kernel calls
:func:`gaussian_pdf` directly.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .util import _bernstein_basis

# --------------------------------------------------------------------------
# initial values and fit bounds

N_CALIB = 4  # (c0, k1, k2, k3)
INIT_CALIB = np.array([-180.0, 1.5, 2.5, 3.5])
BOUNDS_CALIB = [(-300.0, -100.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

N_RESOL = 3  # (b0, b1, b2)
RESOL_E_REF = 2000.0  # keV, reference energy
MIN_SIGMA = 0.001  # keV, sigma floor (numerical safety for negative-variance corners)
INIT_RESOL = np.array([2.0, 20.0, 40.0])
BOUNDS_RESOL = [(0.0, 10.0), (0.0, 80.0), (0.0, 100.0)]

PARAM_NAMES_CORE = ["c0", "k1", "k2", "k3", "b0", "b1", "b2"]
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_K = ["k1", "k2", "k3"]
PARAM_NAMES_B = ["b0", "b1", "b2"]


# --------------------------------------------------------------------------
# energy calibration E(ch)

def poly_basis(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Monomial basis vector [1, x, ..., x^degree] on the last axis."""
    x = np.asarray(x, dtype=float)
    return np.stack([x**k for k in range(degree + 1)], axis=-1)


@numba.njit(inline="always", cache=True)
def calib_model(calib_params, channel, channel_max):
    """Cubic E(channel) from an intercept and three slopes.

    ``calib_params = [c0, k1, k2, k3]`` where ``k1 = E'(0)``, ``k2 = E'(channel_max/2)``
    and ``k3 = E'(channel_max)``; ``channel_max`` is the maximum channel number of the
    acquisition, a fixed constant of the data.  ``calib_params`` is a float64
    array of length 4 and ``channel`` a float64 scalar or array.
    """
    c0 = calib_params[0]
    k1 = calib_params[1]
    k2 = calib_params[2]
    k3 = calib_params[3]
    return (c0 + k1 * channel
            + (4 * k2 - 3 * k1 - k3) / (2.0 * channel_max) * channel**2
            + 2.0 * (k1 - 2.0 * k2 + k3) / (3.0 * channel_max**2) * channel**3)


def c0k1k2k3_to_c0c1c2c3(calib_params: np.ndarray | list[float], channel_max: float) -> np.ndarray:
    """Cubic coefficients [c0, c1, c2, c3] from the calibration parameters.

    ``calib_params = [c0, k1, k2, k3]`` where ``k1 = E'(0)``, ``k2 = E'(channel_max/2)``
    and ``k3 = E'(channel_max)``: ``c1 = k1``, ``c2 = (4 k2 - 3 k1 - k3)/(2 channel_max)`` and
    ``c3 = 2 (k1 - 2 k2 + k3)/(3 channel_max^2)``.
    """
    c0, k1, k2, k3 = np.asarray(calib_params, dtype=float)
    c1 = k1
    c2 = (4 * k2 - 3 * k1 - k3) / (2.0 * channel_max)
    c3 = 2.0 * (k1 - 2.0 * k2 + k3) / (3.0 * channel_max**2)
    return np.array([c0, c1, c2, c3])


def jac_c0k1k2k3(channel_max: float) -> np.ndarray:
    """Constant Jacobian d(c0, c1, c2, c3)/d(c0, k1, k2, k3)."""
    inv2x = 1.0 / (2.0 * channel_max)
    inv3x2 = 1.0 / (3.0 * channel_max**2)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -3.0 * inv2x, 2.0 / channel_max, -inv2x],
        [0.0, 2.0 * inv3x2, -4.0 * inv3x2, 2.0 * inv3x2],
    ])


def reported_calib(calib_params: np.ndarray | list[float], calib_cov: np.ndarray,
                   channel_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reported (c0, c1, c2, c3) values, errors and covariance from raw params.

    Useful only for display: internally the parameterization works with
    ``(c0, k1, k2, k3)``, and the plain cubic coefficients are recovered on
    demand via the constant map ``c0k1k2k3_to_c0c1c2c3``.
    """
    c = c0k1k2k3_to_c0c1c2c3(calib_params, channel_max)
    jac = jac_c0k1k2k3(channel_max)
    cov = jac @ np.asarray(calib_cov, dtype=float) @ jac.T
    err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return c, err, cov

# --------------------------------------------------------------------------
# resolution: Gaussian sigma(E)


@numba.njit(inline="always", cache=True)
def gaussian_pdf(d, sigma):
    """Normal (Gaussian) probability density at offset ``d`` for ``sigma > 0``.

    ``exp(-d^2 / (2 sigma^2)) / (sqrt(2 pi) sigma)``; the normalization uses
    ``math`` constants, which numba constant-folds inside the kernel.
    """
    return math.exp(-0.5 * (d / sigma)**2) / (math.sqrt(2.0 * math.pi) * sigma)


@numba.njit(cache=True)
def resol_sigma_model(resol_params, energy):
    """sigma(E) from the resolution parameters ``resol_params = [b0, b1, b2]`` (in keV).

    ``sigma^2`` is the quadratic Bernstein polynomial in ``t = E / RESOL_E_REF``
    with coefficients ``(b0^2, b1^2, b2^2)``, i.e. a quadratic Bezier curve
    with those coefficients as control values.  ``t`` is clamped at 0 before
    evaluation, so ``sigma^2`` is a non-negative convex combination of the
    squared coefficients for ``0 <= t <= 1`` -- the Bernstein basis is not
    non-negative for negative ``t``, and clamping is what keeps the variance
    from going negative there.  Physically this saturates the resolution at
    the low-energy edge: ``sigma(E <= 0) = b0``.  Above ``RESOL_E_REF`` the
    polynomial continues unclamped, so ``sigma`` keeps growing with energy as
    before and the plotted model band stays consistent with it.  The
    ``MIN_SIGMA`` floor is numerical safety only.  ``resol_params`` is a
    float64 array of length 3 and ``energy`` a float64 scalar or array.
    """
    t = np.clip(np.asarray(energy), 0.0, np.inf) / RESOL_E_REF
    var = np.dot(_bernstein_basis(t, 2), resol_params ** 2)
    return np.sqrt(np.maximum(var, MIN_SIGMA**2))
