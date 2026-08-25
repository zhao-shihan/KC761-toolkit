"""Gaussian detector resolution model and histogram convolution.

Resolution model:
    sigma(E) = a2 E + a1 sqrt(E) + a0      (keV, E in keV)

Fit parameterisation
--------------------
Instead of fitting the polynomial coefficients a0..a2 directly, the fit works
in terms of the *relative* resolution r(E) = sigma(E)/E at the reference
energies 60 keV, 1461 keV, 2614 keV.  Each is a dimensionless width (the
FWHM/2.355 fraction at that line), bounded between 0 and 1, and they must
*decrease* with energy (resolution improves as energy rises).  The
coefficients are recovered by solving the 3x3 linear system
r_i E_i = a0 + a1 sqrt(E_i) + a2 E_i; they are only needed for the forward
model and the final report (``res_to_a``), and the relative resolutions are
recovered from a given a by evaluation (``a_to_res``).

Convolution algorithm (normalised sliding window):
    The intrinsic simulation histogram f is convolved with a Gaussian of
    width sigma(E).  For every target bin i (centre mu_i) only the
    simulation bins within nsigma * sigma(mu_i) are used:

        g_ij = exp(-(mu_i - E_j)^2 / (2 sigma_j^2))
               / (sigma_j sqrt(2 pi)) * dE_j

    with E_j the simulation bin centre, sigma_j = sigma(E_j) its true-energy
    width.  The result is the smeared density sum_j f_j G(mu_i - E_j)
    multiplied by the target bin width; counts of simulation bins near the
    edge of the fitted energy range leak out naturally (the physical
    behaviour), and the fit's normalisation scale absorbs the overall
    factor.  With simulation bins much narrower than sigma (the usual case)
    this is a very accurate approximation of the exact histogram
    convolution, at a fraction of the cost of building the full kernel
    matrix.
"""

from __future__ import annotations

import os

import numpy as np

try:  # numba is optional; without it the numpy convolution fallback is used.
    import numba
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    numba = None
    _HAVE_NUMBA = False

if _HAVE_NUMBA and "NUMBA_NUM_THREADS" not in os.environ:
    # The convolution kernel is small (one thread per handful of grid bins),
    # so many threads only add scheduling overhead; cap the default at 16.
    numba.set_num_threads(min(numba.get_num_threads(), 16))

# 1 / sqrt(2 pi), captured by the convolution kernel.
_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


if _HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _smear_kernel(ec, f, sig_j, mu, sig_i, widths, nsigma):
        """Resolution-smeared density for every target bin i, parallel over i.

            out[i] = width_i * sum_j f_j G(mu_i - E_j),  j within nsigma*sig_i,
            G(x) = exp(-x^2/2) / (sig_j sqrt(2 pi)).

        A sliding window over the simulation bins (no large intermediate
        matrix, no per-call allocation) makes this far faster than the
        vectorised numpy version; ``prange`` splits the target bins across
        threads (NUMBA_NUM_THREADS controls the count).
        """
        out = np.empty(mu.shape[0], dtype=np.float64)
        for i in numba.prange(mu.shape[0]):
            lo = np.searchsorted(ec, mu[i] - nsigma * sig_i[i])
            hi = np.searchsorted(ec, mu[i] + nsigma * sig_i[i])
            s = 0.0
            for j in range(lo, hi):
                d = (mu[i] - ec[j]) / sig_j[j]
                s += f[j] * np.exp(-0.5 * d * d) * _SQRT_2PI_INV / sig_j[j]
            out[i] = widths[i] * s
        return out

# Reference energies (keV) whose relative resolutions parameterise the fit.
RES_ENERGIES = np.array([60.0, 1461.0, 2614.0])

# Default relative resolutions r = sigma/E at RES_ENERGIES (initial fit values)
DEFAULT_R = np.array([0.1, 0.03, 0.02])

# Fit bounds for the relative resolutions (dimensionless, must stay < 1).
BOUNDS_R = [(0.0, 1.0)] * 3

# Soft monotonicity-penalty strength: chi^2 units per (relative-resolution
# unit)^2 of ordering violation.  The relative resolutions are O(0.03), so a
# reversal of 0.01 (one "resolution unit") costs ~10 chi^2 units, growing
# quadratically — a gentle prior that discourages resolution worsening with
# energy without fighting the data where it is unconstrained (e.g. anchors
# far outside the fitted energy range), while keeping the objective finite
# and smooth for the derivative-free optimiser.
MONOTONICITY_PENALTY = 1e5


def ordering_slack(r) -> float:
    """Constraint slack (>= 0 for strictly decreasing resolutions): the
    minimum of -gap, where gap = r_{i+1} - r_i (resolution improves as energy
    rises, so r60 > r1461 > r2614)."""
    gaps = np.diff(np.asarray(r, dtype=float))
    return float(-np.max(gaps))


def monotonicity_penalty(r) -> float:
    """Soft, continuously rising penalty for non-decreasing resolutions.

    Zero when the relative resolutions are strictly decreasing
    (r60 > r1461 > r2614); otherwise grows quadratically with each reversal
    (``MONOTONICITY_PENALTY * violation^2``), so the objective stays finite
    and rises continuously as the violation deepens.  Add to chi^2 rather
    than returning inf: a physically ordered resolution has zero penalty, so
    the minimum is not biased.
    """
    gaps = np.diff(np.asarray(r, dtype=float))
    viol = np.maximum(gaps, 0.0)
    return MONOTONICITY_PENALTY * float(np.sum(viol * viol))


def sigma_model(a: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    """Gaussian sigma (keV) at energy e (keV): a2 E + a1 sqrt(E) + a0."""
    a0, a1, a2 = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    return a2 * e + a1 * np.sqrt(np.maximum(e, 0.0)) + a0


def a_to_res(a, energies=RES_ENERGIES) -> np.ndarray:
    """Relative resolution r = sigma(E)/E at the reference energies."""
    e = np.asarray(energies, dtype=float)
    return sigma_model(a, e) / e


def res_to_a(r, energies=RES_ENERGIES, jacobian: bool = False):
    """Resolution coefficients a = [a0, a1, a2] whose relative widths at the
    reference ``energies`` equal ``r`` (linear solve on r E = sigma(E)).

    With ``jacobian=True`` also returns the 3x3 Jacobian da/dr, used to
    propagate the resolution-fit uncertainties to the reported coefficients.
    """
    e = np.asarray(energies, dtype=float)
    r = np.asarray(r, dtype=float)
    m = np.stack([np.ones_like(e), np.sqrt(e), e], axis=1)  # columns a0,a1,a2
    a = np.linalg.solve(m, r * e)
    if not jacobian:
        return a
    m_inv = np.linalg.inv(m)
    jac = m_inv * e[None, :]  # da/dr_i = M^-1[:, i] * E_i
    return a, jac


def smear_on_bins(sim_counts: np.ndarray, sim_edges: np.ndarray,
                  t_lo: np.ndarray, t_hi: np.ndarray,
                  a: np.ndarray | list[float], nsigma: float = 4.0) -> np.ndarray:
    """Convolve the simulation onto arbitrary target bins.

    Parameters
    ----------
    sim_counts : (N_sim,) intrinsic simulation counts per bin.
    sim_edges  : (N_sim + 1,) simulation bin edges (keV).
    t_lo, t_hi : (M,) lower / upper edges (keV) of the target bins.
    a          : resolution parameters (a0, a1, a2).
    nsigma     : sliding-window half-width in units of sigma.

    Returns
    -------
    (M,) array with the resolution-smeared counts on the target bins
    (count-conserving: sum of the output equals sum of the input).

    The convolution is the dominant cost of the fit; with numba available it
    runs as a JIT-compiled parallel loop over target bins (``prange``), which
    is ~10-30x faster than the vectorised numpy fallback.  Thread count can
    be tuned via the ``NUMBA_NUM_THREADS`` environment variable.
    """
    a = np.asarray(a, dtype=float)
    mu = 0.5 * (t_lo + t_hi)
    sig_i = sigma_model(a, mu)

    sim_centers = 0.5 * (sim_edges[:-1] + sim_edges[1:])

    pad = nsigma * np.max(sig_i) if sig_i.size else 0.0
    jsel = (sim_centers >= mu.min() - pad) & (sim_centers <= mu.max() + pad)
    js = np.where(jsel)[0]
    if js.size == 0:
        return np.zeros_like(mu)

    ec = sim_centers[js]
    f = sim_counts[js]
    sig_j = sigma_model(a, ec)
    if np.any(sig_i <= 0) or np.any(sig_j <= 0):
        # Non-positive resolution (e.g. parameters stepped outside their
        # bounds during finite differences): degenerate, model is zero.
        return np.zeros_like(mu)

    if _HAVE_NUMBA:
        return _smear_kernel(ec, f, sig_j, mu, sig_i, t_hi - t_lo, nsigma)

    # Vectorised numpy fallback: sliding window per target bin with
    # simulation bins within nsigma * sig_i.
    j_lo = np.searchsorted(ec, mu - nsigma * sig_i)
    j_hi = np.searchsorted(ec, mu + nsigma * sig_i)
    width = int(np.max(j_hi - j_lo)) + 1
    cols = np.arange(width, dtype=np.intp)
    idx = j_lo[:, None] + cols[None, :]
    ok = idx < j_hi[:, None]
    idx = np.clip(idx, 0, len(ec) - 1)

    # Gaussian kernel (float64: avoid numerical noise that could degrade the
    # optimiser's step selection).
    diff = (mu[:, None] - ec[idx]) / sig_j[idx]
    g = np.exp(-0.5 * diff * diff) / (np.sqrt(2.0 * np.pi) * sig_j[idx])
    g = np.where(ok, g, 0.0)

    # Smeared *density* sum_j f_j G(mu_i - E_j), times the target bin width.
    return (t_hi - t_lo) * np.einsum("ij,ij->i", g, f[idx])


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, a: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    """Convolve the simulation onto a contiguous histogram with ``target_edges``."""
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], a, nsigma)
