"""Gaussian detector resolution model and histogram convolution.

Resolution model:
    sigma(E) = a2 E + a1 sqrt(E) + a0      (keV, E in keV)

Fit parameterization
-------------------
Instead of fitting the polynomial coefficients a0..a2 directly, the fit works
in terms of the *relative* resolution r(E) = sigma(E)/E at the reference
energies 60 keV, 609 keV, 2614 keV.  Each is a dimensionless width (the
FWHM/2.355 fraction at that line), bounded within ``BOUNDS_R`` (0.001-0.5),
and the three must *decrease* with energy (resolution improves as energy
rises).  The coefficients are recovered by solving the 3x3 linear system
r_i E_i = a0 + a1 sqrt(E_i) + a2 E_i; they are only needed for the forward
model and the final report (``resol_to_a``).

Convolution algorithm (normalized sliding window)
-------------------------------------------------
The intrinsic simulation histogram f is convolved with a Gaussian of width
sigma(E).  For every target bin i (center mu_i) only the simulation bins
within nsigma * sigma(mu_i) are used:

    g_ij = exp(-(mu_i - E_j)^2 / (2 sigma_j^2))
           / (sigma_j sqrt(2 pi)) * dE_j

with E_j the simulation bin center and sigma_j = sigma(E_j) its true-energy
width.  The result is the smeared density sum_j f_j G(mu_i - E_j) multiplied
by the target bin width; counts of simulation bins near the edge of the
fitted energy range leak out naturally (the physical behavior), and the fit's
normalization scale absorbs the overall factor.  With simulation bins much
narrower than sigma (the usual case) this is a very accurate approximation of
the exact histogram convolution, at a fraction of the cost of building the
full kernel matrix.
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

# Reference energies (keV) whose relative resolutions parameterize the fit.
RESOL_ENERGIES = np.array([60.0, 609.0, 2614.0])

# Initial relative resolutions r = sigma/E at RESOL_ENERGIES (fit start values).
INIT_R = np.array([0.1, 0.05, 0.03])

# Fit bounds for the relative resolutions (dimensionless, must stay < 1).
BOUNDS_R = [(0.001, 0.5)] * 3

# Floor (keV) applied to the resolution sigma in the convolution (1 eV).
# With the default parameters sigma(E) goes negative below ~5.8 keV (an
# unphysical low-energy tail of the resolution quadratic).  Clipping sigma
# to exactly 0 would give zero-width Gaussian kernels (empty convolution
# windows -> model 0 on those bins, an irreducible scale-independent chi^2
# contribution); flooring at a small physical width keeps every sim bin
# contributing with a finite (narrow) Gaussian.
SIGMA_FLOOR = 0.001

# Condition-number ceiling for the 3x3 / 4x4 reference-energy systems solved
# in ``resol_to_a`` / ``channels_to_c``: above this the coefficients are
# numerically meaningless (near-coincident reference channels) and NaN is
# returned instead.
_MAX_COND = 1e14

# Soft monotonicity-penalty strength: chi^2 units per (relative-resolution
# unit)^2 of ordering violation.  The relative resolutions are O(0.03), so a
# reversal of 0.01 (one "resolution unit") costs ~10 chi^2 units, growing
# quadratically — a gentle prior that discourages resolution worsening with
# energy without fighting the data where it is unconstrained (e.g. anchors far
# outside the fitted energy range), while keeping the objective finite and
# smooth for the derivative-free optimizer.
MONOTONICITY_PENALTY = 1e5


if _HAVE_NUMBA and "NUMBA_NUM_THREADS" not in os.environ:
    # The convolution kernel is small (one thread per handful of grid bins),
    # so many threads only add scheduling overhead; cap the default at 8.
    numba.set_num_threads(min(numba.get_num_threads(), 8))

# 1 / sqrt(2 pi), captured by the convolution kernel.
_SQRT_2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


if _HAVE_NUMBA:
    @numba.njit(parallel=True)
    def _smear_kernel(ec, f, sig_j, mu, sig_i, widths, nsigma):
        """Resolution-smeared density for every target bin i, parallel over i.

            out[i] = width_i * sum_j f_j G(mu_i - E_j),  j within nsigma*sig_i,
            G(x) = exp(-x^2/2) / (sig_j sqrt(2 pi)).

        A sliding window over the simulation bins (no large intermediate
        matrix, no per-call allocation) makes this far faster than the
        vectorized numpy version; ``prange`` splits the target bins across
        threads (NUMBA_NUM_THREADS controls the count).
        """
        out = np.empty(mu.shape[0], dtype=np.float64)
        for i in numba.prange(mu.shape[0]):
            lo = np.searchsorted(ec, mu[i] - nsigma * sig_i[i])
            hi = np.searchsorted(ec, mu[i] + nsigma * sig_i[i])
            s = 0.0
            for j in range(lo, hi):
                # Defensive: sig_j is floored at SIGMA_FLOOR by the caller,
                # so this only triggers if a zero-width sim bin slips through.
                if sig_j[j] <= 0.0:
                    continue
                d = (mu[i] - ec[j]) / sig_j[j]
                s += f[j] * np.exp(-0.5 * d * d) * _SQRT_2PI_INV / sig_j[j]
            out[i] = widths[i] * s
        return out


def monotonicity_penalty(r: np.ndarray | list[float]) -> float:
    """Soft, continuously rising penalty for non-decreasing resolutions.

    Zero when the relative resolutions are strictly decreasing
    (r60 > r609 > r2614); otherwise grows quadratically with each reversal
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


def _nan_a(jacobian: bool):
    """NaN resolution coefficients (singular / ill-conditioned reference
    system), with a NaN Jacobian when requested."""
    a = np.full(3, np.nan)
    if jacobian:
        return a, np.full((3, 3), np.nan)
    return a


def resol_to_a(r: np.ndarray | list[float],
               energies: np.ndarray | list[float] = RESOL_ENERGIES,
               jacobian: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Resolution coefficients a = [a0, a1, a2] whose relative widths at the
    reference ``energies`` equal ``r`` (linear solve on r E = sigma(E)).

    With ``jacobian=True`` also returns the 3x3 Jacobian da/dr, used to
    propagate the resolution-fit uncertainties to the reported coefficients.

    Coincident / near-coincident reference channels make the 3x3 system
    singular or ill-conditioned; instead of raising or returning wildly
    oscillating coefficients the result is NaN (the forward models' isfinite
    guards turn it into an inf objective).
    """
    e = np.asarray(energies, dtype=float)
    r = np.asarray(r, dtype=float)
    m = np.stack([np.ones_like(e), np.sqrt(e), e],
                 axis=1)  # columns a0, a1, a2
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
    (count-conserving up to the finite nsigma window: sum of the output
    equals the fraction of the input within the window).

    Unphysical negative resolution sigma(E) (the low-energy tail of the
    resolution quadratic at the default parameters) is floored at
    ``SIGMA_FLOOR`` (1 eV): such simulation bins keep contributing with a
    narrow finite Gaussian instead of being excluded entirely.  With a
    strictly positive sigma the smearing is count-conserving everywhere (the
    only approximation is the nsigma window truncation of the Gaussian
    tails).

    The convolution is the dominant cost of the fit; with numba available it
    runs as a JIT-compiled parallel loop over target bins (``prange``), which
    is ~10-30x faster than the vectorized numpy fallback.  Thread count can
    be tuned via the ``NUMBA_NUM_THREADS`` environment variable.
    """
    a = np.asarray(a, dtype=float)
    mu = 0.5 * (t_lo + t_hi)
    # Negative resolutions (the unphysical low-energy tail of the resolution
    # quadratic) are floored at SIGMA_FLOOR (1 eV) instead of being clipped to
    # 0: a zero sigma would give an empty convolution window (model 0 on that
    # target bin) and an irreducible, scale-independent chi^2 contribution.
    # With a strictly positive floor every bin convolves with a finite
    # (narrow) Gaussian and the smearing stays count-conserving.
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

    # Vectorized numpy fallback: sliding window per target bin with
    # simulation bins within nsigma * sig_i.  sig_j is floored at SIGMA_FLOOR
    # by the caller, so no simulation bin is ever excluded for a zero width.
    j_lo = np.searchsorted(ec, mu - nsigma * sig_i)
    j_hi = np.searchsorted(ec, mu + nsigma * sig_i)
    width = int(np.max(j_hi - j_lo)) + 1
    cols = np.arange(width, dtype=np.intp)
    idx = j_lo[:, None] + cols[None, :]
    idx = np.clip(idx, 0, len(ec) - 1)
    ok = idx < j_hi[:, None]

    # Gaussian kernel (float64: avoid numerical noise that could degrade the
    # optimizer's step selection).  sig_j is strictly positive after the
    # caller's floor, so the kernel stays finite everywhere.
    diff = (mu[:, None] - ec[idx]) / sig_j[idx]
    g = np.exp(-0.5 * diff * diff) * _SQRT_2PI_INV / sig_j[idx]
    g = np.where(ok, g, 0.0)

    # Smeared *density* sum_j f_j G(mu_i - E_j), times the target bin width.
    return (t_hi - t_lo) * np.einsum("ij,ij->i", g, f[idx])


def smear(sim_counts: np.ndarray, sim_edges: np.ndarray,
          target_edges: np.ndarray, a: np.ndarray | list[float],
          nsigma: float = 4.0) -> np.ndarray:
    """Convolve the simulation onto a contiguous histogram with ``target_edges``."""
    return smear_on_bins(sim_counts, sim_edges,
                         target_edges[:-1], target_edges[1:], a, nsigma)
