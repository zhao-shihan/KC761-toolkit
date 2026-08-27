"""Detector response: energy calibration E(x) and resolution sigma^2(E).

Both halves of the response are parameterized through their values at fixed
anchor energies and recovered by exact interpolation through those anchors:

  - E(x) = c3 x^3 + c2 x^2 + c1 x + c0 through the channel positions of the
    lines in ``CALIB_ENERGIES``;
  - sigma^2(E) = b0 + b1 E + b2 E^2 (noise / Fano / constant term added in
    quadrature) through the relative resolutions at ``RESOL_ENERGIES``.

Non-negative b coefficients guarantee a positive sigma(E) and a decreasing
sigma/E, and E(x) monotonicity is checked exactly on the channel axis, so
the fit gate can reject infeasible states with chi2 = inf instead of using
soft penalties.  The fit keeps the interpretable anchor values as free
parameters; ``channels_to_c`` and ``resol_to_b`` map them to coefficients.
"""

from __future__ import annotations

import os

import numba
import numpy as np

# --------------------------------------------------------------------------
# anchors and shared solver tolerance


CALIB_ENERGIES = np.array([60.0, 609.0, 1461.0, 2614.0])
INIT_X = np.array([160.0, 500.0, 900.0, 1350.0])

RESOL_ENERGIES = np.array([60.0, 609.0, 2614.0])
# Defaults must imply non-negative b coefficients so that the initial point
# is admissible (b0 >= 0 needs r60 >= ~0.148 given the other two anchors).
INIT_R = np.array([0.15, 0.05, 0.03])
BOUNDS_R = [(0.001, 0.5)] * 3

MAX_COND = 1e14
SIGMA_FLOOR = 0.001


# --------------------------------------------------------------------------
# energy calibration E(x)


def poly_basis(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Design matrix [1, x, ..., x^degree]; matches poly3/sigma_model terms."""
    x = np.asarray(x, dtype=float)
    return np.stack([x**k for k in range(degree + 1)], axis=-1)


def _solve_exact(m: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Solve m·theta = y exactly, guarded against degenerate systems."""
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(y))):
        return None
    try:
        if np.linalg.cond(m) > MAX_COND:
            return None
        return np.linalg.solve(m, y)
    except np.linalg.LinAlgError:
        return None


def _nan(n: int, jacobian: bool = False):
    theta = np.full(n, np.nan)
    if jacobian:
        return theta, np.full((n, n), np.nan)
    return theta


def poly3(c: np.ndarray | list[float], x: np.ndarray | float) -> np.ndarray:
    c0, c1, c2, c3 = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    return c3 * x**3 + c2 * x**2 + c1 * x + c0


def channels_to_c(x: np.ndarray | list[float],
                  energies: np.ndarray | list[float] = CALIB_ENERGIES,
                  jacobian: bool = False,
                  ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Polynomial coefficients interpolating ``energies`` at channels ``x``.

    Returns NaN coefficients when the system is not solvable.  The optional
    Jacobian is dC/dX of the implicit relation V(x) c(anchors) = energies.
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(energies, dtype=float)
    v = poly_basis(x, 3)
    c = _solve_exact(v, e)
    if c is None:
        return _nan(4, jacobian)
    if not jacobian:
        return c
    try:
        v_inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return _nan(4, jacobian)
    jac = np.empty((4, 4))
    dv = np.zeros_like(v)
    for j in range(4):
        dv[:] = 0.0
        dv[j] = [0.0, 1.0, 2.0 * x[j], 3.0 * x[j] ** 2]
        jac[:, j] = -v_inv @ (dv @ c)
    return c, jac


def poly3_is_increasing(c: np.ndarray | list[float], x_max: float) -> bool:
    """True if E(x) is strictly increasing over channels [0, x_max].

    Checked exactly via the minimum of E'(x) on the interval; a non-finite
    coefficient vector is rejected.
    """
    c = np.asarray(c, dtype=float)
    if not np.isfinite(c).all():
        return False
    _, c1, c2, c3 = c
    xs = [0.0, float(x_max)]
    if c3 != 0.0:
        xv = -c2 / (3.0 * c3)
        if 0.0 < xv < x_max:
            xs.append(xv)
    xs = np.asarray(xs)
    slope = c1 + 2.0 * c2 * xs + 3.0 * c3 * xs**2
    return bool(np.all(slope > 0.0))


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


def sigma_model(b: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    """sigma(E) from variance coefficients; clamped at zero for safety."""
    b0, b1, b2 = np.asarray(b, dtype=float)
    e = np.asarray(e, dtype=float)
    var = b0 + b1 * e + b2 * e**2
    return np.sqrt(np.maximum(var, 0.0))


def coeffs_ok(b: np.ndarray | list[float]) -> bool:
    """Admissible sigma^2 polynomial: finite and everywhere non-negative."""
    b = np.asarray(b, dtype=float)
    return bool(np.all(np.isfinite(b)) and np.all(b >= 0.0))


def resol_to_b(r: np.ndarray | list[float],
               energies: np.ndarray | list[float] = RESOL_ENERGIES,
               jacobian: bool = False,
               ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """sigma^2 coefficients from anchor resolutions r_i = sigma(E_i)/E_i.

    Solves the linear system b0 + b1 E + b2 E^2 = (r E)^2 through the three
    anchors.  The Jacobian is db/dR with columns M^-1 * 2 r_i E_i^2 e_i.
    """
    e = np.asarray(energies, dtype=float)
    r = np.asarray(r, dtype=float)
    m = poly_basis(e, 2)
    y = (r * e) ** 2
    b = _solve_exact(m, y)
    if b is None:
        return _nan(3, jacobian)
    if not jacobian:
        return b
    try:
        jac = np.linalg.solve(m, np.diag(2.0 * r * e * e))
    except np.linalg.LinAlgError:
        return _nan(3, jacobian)
    return b, jac


def smear_on_bins(sim_counts: np.ndarray, sim_edges: np.ndarray,
                  t_lo: np.ndarray, t_hi: np.ndarray,
                  b: np.ndarray | list[float], nsigma: float = 4.0) -> np.ndarray:
    """Convolve sim counts with an energy-dependent Gaussian onto target bins."""
    b = np.asarray(b, dtype=float)
    mu = 0.5 * (t_lo + t_hi)
    sig_i = np.maximum(sigma_model(b, mu), SIGMA_FLOOR)

    sim_centers = 0.5 * (sim_edges[:-1] + sim_edges[1:])

    pad = nsigma * np.max(sig_i) if sig_i.size else 0.0
    jsel = (sim_centers >= mu.min() - pad) & (sim_centers <= mu.max() + pad)
    js = np.where(jsel)[0]
    if js.size == 0:
        return np.zeros_like(mu)

    ec = sim_centers[js]
    f = sim_counts[js]
    sig_j = np.maximum(sigma_model(b, ec), SIGMA_FLOOR)

    return _smear_kernel(ec, f, sig_j, mu, sig_i, t_hi - t_lo, nsigma)


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, b: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], b, nsigma)
