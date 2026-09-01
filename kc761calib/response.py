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

Resolution is an exponentially modified Gaussian (EMG): a Gaussian core with
standard deviation ``sigma`` convolved with a one-sided exponential of mean
``tau``, giving full-energy peaks a high-energy tail.  Both parameters are
quadratic Bezier curves in ``t = E / RESOL_T_SCALE`` (``RESOL_T_SCALE = 2000``
keV):

   sigma^2(t) = (1-t)^2 b0^2 + 2(1-t)t b1^2 + t^2 b2^2
   tau(t)     = relu((1-t)^2 b3 + 2(1-t)t b4 + t^2 b5)

``sigma`` keeps squared control values, so it is real and non-negative for any
real ``(b0, b1, b2)``.  ``tau`` is a direct Bezier curve floored at ``MIN_TAU``
(0.001 keV), so it is positive without any control-value constraint.  The EMG
mean is ``tau`` (not zero), so a smeared peak shifts to higher energy by
``~tau(E)``; this is the intended tail effect.
"""

from __future__ import annotations

import math
import os

import numba
import numpy as np

from .util import bezier2_basis

# --------------------------------------------------------------------------
# initial values and fit bounds

N_CALIB = 4  # (c0, k1, k2, k3)
INIT_CALIB = np.array([-180.0, 1.5, 2.5, 3.5])
BOUNDS_CALIB = [(-300.0, -100.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

N_RESOL = 6  # (b0, b1, b2) sigma; (b3, b4, b5) tau
RESOL_T_SCALE = 2000.0  # keV
MIN_SIGMA = 0.001  # keV, sigma floor (numerical safety)
MIN_TAU = 0.001  # keV, tau floor
INIT_RESOL = np.array([2.0, 20.0, 40.0,
                       0.0, 0.0, 50.0])
BOUNDS_RESOL = [(0.0, 10.0), (0.0, 80.0), (0.0, 100.0),
                (-10.0, 10.0), (-50.0, 50.0), (0.0, 200.0)]

PARAM_NAMES_CORE = ["c0", "k1", "k2", "k3", "b0", "b1", "b2", "b3", "b4", "b5"]
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_K = ["k1", "k2", "k3"]
PARAM_NAMES_B = ["b0", "b1", "b2", "b3", "b4", "b5"]


# --------------------------------------------------------------------------
# energy calibration E(x)

def poly_basis(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Design matrix [1, x, ..., x^degree]."""
    x = np.asarray(x, dtype=float)
    return np.stack([x**k for k in range(degree + 1)], axis=-1)


def calib_model(calib_params: np.ndarray | list[float], channel: np.ndarray | float,
                channel_max: float) -> np.ndarray:
    """Cubic E(channel) from an intercept and three slopes.

    ``calib_params = [c0, k1, k2, k3]`` where ``k1 = E'(0)``, ``k2 = E'(channel_max/2)``
    and ``k3 = E'(channel_max)``; ``channel_max`` is the maximum channel number of the
    acquisition, a fixed constant of the data.
    """
    c0, k1, k2, k3 = np.asarray(calib_params, dtype=float)
    channel = np.asarray(channel, dtype=float)
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
# resolution EMG (sigma, tau) and histogram convolution


if "NUMBA_NUM_THREADS" not in os.environ:
    numba.set_num_threads(numba.get_num_threads())


@numba.njit(inline="always")
def _erfcx(z):
    """Scaled complementary error function ``erfcx(z) = exp(z^2) erfc(z)`` for ``z >= 0``.

    The exact ``exp(z^2) * erfc(z)`` is used below ``z = 8`` (no overflow); above
    it the asymptotic series ``sqrt(pi) erfcx(z) = 1/z - 1/(2z^3) + 3/(4z^5)
    - 15/(8z^7) + 105/(16z^9)`` is used, accurate to ``~945/(32 z^10)`` relative.
    """
    if z < 8.0:
        return math.exp(z * z) * math.erfc(z)
    invz = 1.0 / z
    z2 = invz * invz
    return invz / math.sqrt(math.pi) * (
        1.0 + z2 * (-0.5 + z2 * (0.75 + z2 * (-1.875 + z2 * 6.5625))))


@numba.njit(inline="always")
def _emg_density(d, sigma, tau):
    """EMG probability density at offset ``d``, stable for all ``sigma, tau > 0``.

    Uses the erfcx-scaled form so the ``sigma >> tau`` (near-Gaussian) limit does
    not overflow; the ``z < 0`` branch switches to the exponential-tail form.
    """
    z = (sigma / tau - d / sigma) / math.sqrt(2.0)
    if z >= 0.0:
        return 0.5 / tau * math.exp(-0.5 * (d / sigma) * (d / sigma)) * _erfcx(z)
    t1 = math.exp(-d / tau + 0.5 * (sigma / tau) * (sigma / tau)) / tau
    t2 = 0.5 / tau * math.exp(-0.5 * (d / sigma) * (d / sigma)) * _erfcx(-z)
    return t1 - t2


@numba.njit(parallel=True)
def _smear_kernel(ec, f, sig_j, tau_j, mu, sig_i, tau_i, widths, nsigma, ntail):
    """EMG blur of sim counts ``f`` at centers ``ec`` onto target centers ``mu``.

    Each sim center contributes ``f[j] * emg_density(mu[i] - ec[j])`` (the EMG
    density is normalized to unit integral).  Only centers within the asymmetric
    window ``[-nsigma*sigma_i, max(nsigma*sigma_i, ntail*tau_i)]`` contribute;
    ``searchsorted`` finds that contiguous span in the sorted ``ec`` grid.
    """
    out = np.empty(mu.shape[0], dtype=np.float64)
    for i in numba.prange(mu.shape[0]):
        r_lo = nsigma * sig_i[i]
        t_lo = ntail * tau_i[i]
        if t_lo > r_lo:
            r_lo = t_lo
        lo = np.searchsorted(ec, mu[i] - r_lo)
        hi = np.searchsorted(ec, mu[i] + nsigma * sig_i[i])
        s = 0.0
        for j in range(lo, hi):
            s += f[j] * _emg_density(mu[i] - ec[j], sig_j[j], tau_j[j])
        out[i] = widths[i] * s
    return out


def resol_sigma_model(resol_params: np.ndarray | list[float], energy: np.ndarray | float) -> np.ndarray:
    """sigma(E) from the sigma Bezier control values ``resol_params[0:3]`` (in keV).

    ``resol_params`` is the full 6-vector ``[b0, ..., b5]``; only the first three control
    values are used, and ``t = E / RESOL_T_SCALE``.  The squared control values
    keep the variance non-negative; clamped at zero only as a numerical safety.
    """
    resol_params = np.asarray(resol_params, dtype=float)
    energy = np.asarray(energy, dtype=float)
    var = bezier2_basis(energy / RESOL_T_SCALE) @ (resol_params[:3] ** 2)
    return np.sqrt(np.maximum(var, MIN_SIGMA**2))


def resol_tau_model(resol_params: np.ndarray | list[float], energy: np.ndarray | float) -> np.ndarray:
    """tau(E) from the tau Bezier control values ``resol_params[3:6]`` (in keV).

    The control values are used directly (no squaring); the curve is floored at
    ``MIN_TAU`` to keep the EMG mean positive.
    """
    resol_params = np.asarray(resol_params, dtype=float)
    energy = np.asarray(energy, dtype=float)
    tau = bezier2_basis(energy / RESOL_T_SCALE) @ resol_params[3:6]
    return np.maximum(tau, MIN_TAU)


def smear_on_bins(sim_counts: np.ndarray, sim_edges: np.ndarray,
                  t_lo: np.ndarray, t_hi: np.ndarray,
                  resol_params: np.ndarray | list[float], nsigma: float = 4.0, ntail: float = 10.0,
                  sim_centers: np.ndarray | None = None) -> np.ndarray:
    """Convolve sim counts with an energy-dependent EMG onto target bins.

    The target bins ``[t_lo, t_hi]`` may be non-uniform: each output bin is
    ``width_i * density(mu_i - E_sim)`` with ``width_i = t_hi[i] - t_lo[i]``.
    """
    resol_params = np.asarray(resol_params, dtype=float)
    if sim_centers is None:
        sim_centers = 0.5 * (sim_edges[:-1] + sim_edges[1:])
    mu = 0.5 * (t_lo + t_hi)
    sig_i = resol_sigma_model(resol_params, mu)
    tau_i = resol_tau_model(resol_params, mu)

    pad_lo = max(nsigma * np.max(sig_i), ntail * np.max(tau_i)) if sig_i.size else 0.0
    pad_hi = nsigma * np.max(sig_i) if sig_i.size else 0.0
    jsel = (sim_centers >= mu.min() -
            pad_lo) & (sim_centers <= mu.max() + pad_hi)
    js = np.where(jsel)[0]
    if js.size == 0:
        return np.zeros_like(mu)

    ec = sim_centers[js]
    f = sim_counts[js]
    sig_j = resol_sigma_model(resol_params, ec)
    tau_j = resol_tau_model(resol_params, ec)

    return _smear_kernel(
        ec, f, sig_j, tau_j, mu, sig_i, tau_i, t_hi - t_lo, nsigma, ntail)


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, resol_params: np.ndarray | list[float],
          nsigma: float = 4.0, ntail: float = 10.0,
          sim_centers: np.ndarray | None = None) -> np.ndarray:
    return smear_on_bins(
        sim_counts, sim_edges,
        target_edges[:-1], target_edges[1:], resol_params, nsigma, ntail,
        sim_centers=sim_centers)
