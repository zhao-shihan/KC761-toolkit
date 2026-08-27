"""Gaussian detector resolution model and histogram convolution."""

from __future__ import annotations

import os

import numpy as np

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    numba = None
    _HAVE_NUMBA = False

RESOL_ENERGIES = np.array([60.0, 609.0, 2614.0])
INIT_R = np.array([0.1, 0.05, 0.03])
BOUNDS_R = [(0.001, 0.5)] * 3

SIGMA_FLOOR = 0.001
_MAX_COND = 1e14
MONOTONICITY_PENALTY = 10.0


if _HAVE_NUMBA and "NUMBA_NUM_THREADS" not in os.environ:
    numba.set_num_threads(min(numba.get_num_threads(), 16))

_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


if _HAVE_NUMBA:
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


def monotonicity_penalty(r: np.ndarray | list[float]) -> float:
    gaps = np.diff(np.asarray(r, dtype=float))
    viol = np.maximum(gaps, 0.0)
    return MONOTONICITY_PENALTY * 1e4 * float(np.sum(viol * viol))


def sigma_model(a: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    a0, a1, a2 = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    return a2 * e + a1 * np.sqrt(np.maximum(e, 0.0)) + a0


def _nan_a(jacobian: bool):
    a = np.full(3, np.nan)
    if jacobian:
        return a, np.full((3, 3), np.nan)
    return a


def resol_to_a(r: np.ndarray | list[float],
               energies: np.ndarray | list[float] = RESOL_ENERGIES,
               jacobian: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    e = np.asarray(energies, dtype=float)
    r = np.asarray(r, dtype=float)
    m = np.stack([np.ones_like(e), np.sqrt(e), e],
                 axis=1)
    if not np.isfinite(r).all():
        return _nan_a(jacobian)
    try:
        if np.linalg.cond(m) > _MAX_COND:
            return _nan_a(jacobian)
        a = np.linalg.solve(m, r * e)
    except np.linalg.LinAlgError:
        return _nan_a(jacobian)
    if not jacobian:
        return a
    try:
        m_inv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return _nan_a(jacobian)
    jac = m_inv * e[None, :]
    return a, jac


def smear_on_bins(sim_counts: np.ndarray, sim_edges: np.ndarray,
                  t_lo: np.ndarray, t_hi: np.ndarray,
                  a: np.ndarray | list[float], nsigma: float = 4.0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    mu = 0.5 * (t_lo + t_hi)
    sig_i = np.maximum(sigma_model(a, mu), SIGMA_FLOOR)

    sim_centers = 0.5 * (sim_edges[:-1] + sim_edges[1:])

    pad = nsigma * np.max(sig_i) if sig_i.size else 0.0
    jsel = (sim_centers >= mu.min() - pad) & (sim_centers <= mu.max() + pad)
    js = np.where(jsel)[0]
    if js.size == 0:
        return np.zeros_like(mu)

    ec = sim_centers[js]
    f = sim_counts[js]
    sig_j = np.maximum(sigma_model(a, ec), SIGMA_FLOOR)

    if _HAVE_NUMBA:
        return _smear_kernel(ec, f, sig_j, mu, sig_i, t_hi - t_lo, nsigma)

    j_lo = np.searchsorted(ec, mu - nsigma * sig_i)
    j_hi = np.searchsorted(ec, mu + nsigma * sig_i)
    width = int(np.max(j_hi - j_lo)) + 1
    cols = np.arange(width, dtype=np.intp)
    idx = j_lo[:, None] + cols[None, :]
    idx = np.clip(idx, 0, len(ec) - 1)
    ok = idx < j_hi[:, None]

    diff = (mu[:, None] - ec[idx]) / sig_j[idx]
    g = np.exp(-0.5 * diff * diff) * _SQRT_2PI_INV / sig_j[idx]
    g = np.where(ok, g, 0.0)

    return (t_hi - t_lo) * np.einsum("ij,ij->i", g, f[idx])


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, a: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], a, nsigma)
