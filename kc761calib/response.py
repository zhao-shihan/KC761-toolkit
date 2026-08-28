"""Detector response: energy calibration E(x) and resolution sigma^2(E).

The calibration is cubic in the channel number and is parameterized by the
intercept plus three slopes ``(c0, k1, k2, k3)``:

   E(x) = c0 + k1 x + (4 k2 - 3 k1 - k3)/(2 x_max) x^2
                 + 2 (k1 - 2 k2 + k3)/(3 x_max^2) x^3,
   E'(0) = k1,   E'(x_max/2) = k2,   E'(x_max) = k3.

Monotonicity on [0, x_max] is guaranteed by the fit bounds (no separate check).
Reported coefficients are the plain cubic form ``(c0, c1, c2, c3)`` with
``c1 = k1``, ``c2 = (4 k2 - 3 k1 - k3)/(2 x_max)`` and
``c3 = 2 (k1 - 2 k2 + k3)/(3 x_max^2)`` via the constant, invertible map
``c0k1k2k3_to_c0c1c2c3``.

Resolution is parametrized directly by ``sigma^2(E) = b0^2 + b1^2 E + b2^2 E^2``;
the squared coefficients keep sigma(E) real and positive for any real
``(b0, b1, b2)``, so no extra positivity constraint is needed.
"""

from __future__ import annotations

import os

import numba
import numpy as np

# --------------------------------------------------------------------------
# initial values and fit bounds

N_CALIB = 4  # (c0, k1, k2, k3)
INIT_CALIB = np.array([-180.0, 1.5, 2.5, 3.5])
BOUNDS_CALIB = [(-200.0, -160.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

N_RESOL = 3  # (b0, b1, b2)
INIT_RESOL = np.array([2.0, 1.0, 0.0])
BOUNDS_RESOL = [(0.0, 10.0), (0.025, 2.5), (0.0, 0.1)]

PARAM_NAMES_CORE = ["c0", "k1", "k2", "k3", "b0", "b1", "b2"]
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_K = ["k1", "k2", "k3"]
PARAM_NAMES_B = ["b0", "b1", "b2"]


# --------------------------------------------------------------------------
# energy calibration E(x)

def poly_basis(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Design matrix [1, x, ..., x^degree]."""
    x = np.asarray(x, dtype=float)
    return np.stack([x**k for k in range(degree + 1)], axis=-1)


def calib_model(calib_params: np.ndarray | list[float], x: np.ndarray | float,
                x_max: float) -> np.ndarray:
    """Cubic E(x) from an intercept and three slopes.

    ``calib_params = [c0, k1, k2, k3]`` where ``k1 = E'(0)``, ``k2 = E'(x_max/2)``
    and ``k3 = E'(x_max)``; ``x_max`` is the maximum channel number of the
    acquisition, a fixed constant of the data.
    """
    c0, k1, k2, k3 = np.asarray(calib_params, dtype=float)
    x = np.asarray(x, dtype=float)
    return (c0 + k1 * x
            + (4 * k2 - 3 * k1 - k3) / (2.0 * x_max) * x**2
            + 2.0 * (k1 - 2.0 * k2 + k3) / (3.0 * x_max**2) * x**3)


def c0k1k2k3_to_c0c1c2c3(calib_params: np.ndarray | list[float], x_max: float) -> np.ndarray:
    """Cubic coefficients [c0, c1, c2, c3] from the calibration parameters.

    ``calib_params = [c0, k1, k2, k3]`` where ``k1 = E'(0)``, ``k2 = E'(x_max/2)``
    and ``k3 = E'(x_max)``: ``c1 = k1``, ``c2 = (4 k2 - 3 k1 - k3)/(2 x_max)`` and
    ``c3 = 2 (k1 - 2 k2 + k3)/(3 x_max^2)``.
    """
    c0, k1, k2, k3 = np.asarray(calib_params, dtype=float)
    c1 = k1
    c2 = (4 * k2 - 3 * k1 - k3) / (2.0 * x_max)
    c3 = 2.0 * (k1 - 2.0 * k2 + k3) / (3.0 * x_max**2)
    return np.array([c0, c1, c2, c3])


def jac_c0k1k2k3(x_max: float) -> np.ndarray:
    """Constant Jacobian d(c0, c1, c2, c3)/d(c0, k1, k2, k3)."""
    inv2x = 1.0 / (2.0 * x_max)
    inv3x2 = 1.0 / (3.0 * x_max**2)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -3.0 * inv2x, 2.0 / x_max, -inv2x],
        [0.0, 2.0 * inv3x2, -4.0 * inv3x2, 2.0 * inv3x2],
    ])


def reported_calib(calib_params: np.ndarray | list[float], cov_calib: np.ndarray,
                   x_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reported (c0, c1, c2, c3) values, errors and covariance from raw params.

    Useful only for display: internally the parameterization works with
    ``(c0, k1, k2, k3)``, and the plain cubic coefficients are recovered on
    demand via the constant map ``c0k1k2k3_to_c0c1c2c3``.
    """
    c = c0k1k2k3_to_c0c1c2c3(calib_params, x_max)
    jac = jac_c0k1k2k3(x_max)
    cov = jac @ np.asarray(cov_calib, dtype=float) @ jac.T
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    return c, err, cov

# --------------------------------------------------------------------------
# resolution sigma^2(E) and histogram convolution


if "NUMBA_NUM_THREADS" not in os.environ:
    numba.set_num_threads(min(numba.get_num_threads(), 16))

_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


@numba.njit(parallel=True)
def _smear_kernel(ec, f, sig_j, mu, sig_i, widths, nsigma):
    out = np.empty(mu.shape[0], dtype=np.float64)
    for i in numba.prange(mu.shape[0]):
        lo = np.searchsorted(ec, mu[i] - nsigma * sig_i[i])
        hi = np.searchsorted(ec, mu[i] + nsigma * sig_i[i])
        s = 0.0
        for j in range(lo, hi):
            if sig_j[j] <= 0.0:
                continue
            d = (mu[i] - ec[j]) / sig_j[j]
            s += f[j] * np.exp(-0.5 * d * d) * _SQRT_2PI_INV / sig_j[j]
        out[i] = widths[i] * s
    return out


def resol_model(b: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    """sigma(E) = sqrt(b0^2 + b1^2 E + b2^2 E^2).

    The squared coefficients keep the variance non-negative for any real ``b``;
    clamped at zero only as a numerical safety.
    """
    b0, b1, b2 = np.asarray(b, dtype=float)
    e = np.asarray(e, dtype=float)
    var = b0**2 + (b1**2 + b2**2 * e) * e
    return np.sqrt(np.maximum(var, 0.0))


MIN_SIGMA = 0.001  # 1eV


def smear_on_bins(sim_counts: np.ndarray, sim_edges: np.ndarray,
                  t_lo: np.ndarray, t_hi: np.ndarray,
                  b: np.ndarray | list[float], nsigma: float = 4.0) -> np.ndarray:
    """Convolve sim counts with an energy-dependent Gaussian onto target bins."""
    b = np.asarray(b, dtype=float)
    mu = 0.5 * (t_lo + t_hi)
    sig_i = np.maximum(resol_model(b, mu), MIN_SIGMA)

    sim_centers = 0.5 * (sim_edges[:-1] + sim_edges[1:])

    pad = nsigma * np.max(sig_i) if sig_i.size else 0.0
    jsel = (sim_centers >= mu.min() - pad) & (sim_centers <= mu.max() + pad)
    js = np.where(jsel)[0]
    if js.size == 0:
        return np.zeros_like(mu)

    ec = sim_centers[js]
    f = sim_counts[js]
    sig_j = np.maximum(resol_model(b, ec), MIN_SIGMA)

    return _smear_kernel(ec, f, sig_j, mu, sig_i, t_hi - t_lo, nsigma)


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, b: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], b, nsigma)
