"""Energy calibration: cubic E(x) = c3 x^3 + c2 x^2 + c1 x + c0  (x = channel).

The calibrated experimental histogram is obtained by mapping every channel
bin edge through E(x).  Because the map is 1:1 and monotonic, the calibrated
data histogram keeps the *same* per-bin counts and errors — only the axis is
relabelled:

    calibrated bin j covers [E(x_j), E(x_{j+1})]   (x_j = channel edges)

This is the correct, count-conserving transform (equivalent to an exact
rebin of the channel histogram onto the energy grid E(channel edges)).

Fit parameterisation
--------------------
Instead of fitting the polynomial coefficients c0..c3 directly (their allowed
ranges are awkward and hard to constrain), the fit works in terms of the four
*channel positions* of the reference gamma lines

    60 keV (Am-241), 609 keV (Bi-214), 1461 keV (K-40), 2614 keV (Tl-208)

Each is simply a channel number between 0 and the number of MCA channels and
they must be strictly increasing with energy.  The polynomial coefficients
are recovered from the channels by solving the 4x4 Vandermonde system
E_i = poly3(c, x_i) (``channels_to_c``), which is only needed for the forward
model and for the final report.
"""

from __future__ import annotations

import numpy as np

# Reference energies (keV) whose channel positions parameterise the
# calibration: the principal gamma lines of the KC761 sources.
CAL_ENERGIES = np.array([60.0, 609.0, 1461.0, 2614.0])

# Default channel positions of the reference lines (initial fit values),
# typical for the KC761 MCA (~1.49 keV/channel, 2048 channels).
DEFAULT_X = np.array([150.0, 470.0, 890.0, 1360.0])

# Soft monotonicity-penalty strength: chi^2 units per (channel)^2 of
# ordering violation.  A 1-channel reversal of the calibration-line order
# costs this many chi^2 units, growing quadratically — finite and smooth, so
# the derivative-free optimiser can converge near the ordering boundary
# instead of hitting a hard inf wall.
MONOTONICITY_PENALTY = 10.0


def poly3(c: np.ndarray | list[float], x: np.ndarray | float) -> np.ndarray:
    """Evaluate the calibration polynomial E(x) = c3 x^3 + c2 x^2 + c1 x + c0."""
    c0, c1, c2, c3 = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    return c3 * x**3 + c2 * x**2 + c1 * x + c0


def channels_to_c(x, energies=CAL_ENERGIES, jacobian: bool = False):
    """Calibration coefficients c = [c0..c3] whose curve passes through the
    channel positions ``x`` at the reference ``energies`` (Vandermonde solve).

    With ``jacobian=True`` also returns the 4x4 Jacobian dc/dx, used to
    propagate the channel-fit uncertainties to the reported coefficients.
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(energies, dtype=float)
    v = np.stack([np.ones_like(x), x, x**2, x**3], axis=1)
    c = np.linalg.solve(v, e)
    if not jacobian:
        return c
    v_inv = np.linalg.inv(v)
    jac = np.empty((4, 4))
    dv = np.zeros_like(v)
    for j in range(4):
        dv[:] = 0.0
        dv[j] = [0.0, 1.0, 2.0 * x[j], 3.0 * x[j] ** 2]
        jac[:, j] = -v_inv @ (dv @ c)  # d(E = V c)/dx_j -> V dc = -dV c
    return c, jac


def ordering_slack(x) -> float:
    """Constraint slack (>= 0 for strictly increasing channels): the minimum
    gap between consecutive calibration-line channel positions."""
    return float(np.min(np.diff(np.asarray(x, dtype=float))))


def monotonicity_penalty(x) -> float:
    """Soft, continuously rising penalty for non-increasing channels.

    Zero when the calibration-line channels are strictly increasing
    (x60 < x609 < x1461 < x2614); otherwise grows quadratically with each
    reversal (``MONOTONICITY_PENALTY * violation^2``), so the objective stays
    finite and rises continuously as the violation deepens.  Add to chi^2
    rather than returning inf: a physically ordered calibration has zero
    penalty, so the minimum is not biased.
    """
    gaps = np.diff(np.asarray(x, dtype=float))
    viol = np.maximum(-gaps, 0.0)
    return MONOTONICITY_PENALTY * float(np.sum(viol * viol))


def calibrated_edges(c: np.ndarray | list[float], channel_edges: np.ndarray) -> np.ndarray:
    """Energy bin edges obtained by applying the calibration to channel edges."""
    return poly3(c, channel_edges)
