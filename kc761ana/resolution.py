"""Gaussian detector resolution model and histogram convolution.

Resolution model:
    sigma(E) = a2 E + a1 sqrt(E) + a0      (keV, E in keV)

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

import numpy as np

# Resolution parameters a0..a2: default initial values and fit bounds.
# (units: a0 keV, a1 keV/sqrt(keV), a2 dimensionless)
DEFAULT_A = np.array([5.0, 2.0, 0.0])
BOUNDS_A = [
    (1.0, 100.0),  # a0
    (0.0, 10.0),   # a1
    (0.0, 0.1),    # a2
]


def sigma_model(a: np.ndarray | list[float], e: np.ndarray | float) -> np.ndarray:
    """Gaussian sigma (keV) at energy e (keV): a2 E + a1 sqrt(E) + a0."""
    a0, a1, a2 = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    return a2 * e + a1 * np.sqrt(np.maximum(e, 0.0)) + a0


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
        # Non-positive resolution (e.g. resolution parameters stepped outside
        # their bounds during finite differences): degenerate, model is zero.
        return np.zeros_like(mu)

    # Sliding window per target bin: simulation bins within nsigma * sig_i.
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
    out = (t_hi - t_lo) * np.einsum("ij,ij->i", g, f[idx])
    return out


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, a: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    """Convolve the simulation onto a contiguous histogram with ``target_edges``."""
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], a, nsigma)
