"""Energy calibration: E(x) = c3 x^3 + c2 x^2 + c1 x + c0  (x = channel).

The calibrated experimental histogram is obtained by mapping every channel
bin edge through E(x).  Because the map is 1:1 and monotonic, the calibrated
data histogram keeps the *same* per-bin counts and errors — only the axis is
relabelled:

    calibrated bin j covers [E(x_j), E(x_{j+1})]   (x_j = channel edges)

This is the correct, count-conserving transform (equivalent to an exact
rebin of the channel histogram onto the energy grid E(channel edges)).
"""

from __future__ import annotations

import numpy as np

# Calibration parameters c0..c3: default initial values and fit bounds.
# (units: c0 keV, c1 keV/channel, c2 keV/channel^2, c3 keV/channel^3)
DEFAULT_C = np.array([-181.0, 1.49, 3.88e-4, 1.95e-8])
BOUNDS_C = [
    (-300.0, -100.0),  # c0
    (1.3, 1.7),     # c1
    (-1e-3, 1e-3),  # c2
    (-1e-6, 1e-6),  # c3
]


def poly3(c: np.ndarray | list[float], x: np.ndarray | float) -> np.ndarray:
    """Evaluate the calibration polynomial E(x) = c3 x^3 + c2 x^2 + c1 x + c0."""
    c0, c1, c2, c3 = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    return c3 * x**3 + c2 * x**2 + c1 * x + c0


def is_monotonic(c: np.ndarray | list[float], channel_edges: np.ndarray) -> bool:
    """True if E(x) is strictly increasing over the given channel edges."""
    e = poly3(c, channel_edges)
    return bool(np.all(np.diff(e) > 0))


def calibrated_edges(c: np.ndarray | list[float], channel_edges: np.ndarray) -> np.ndarray:
    """Energy bin edges obtained by applying the calibration to channel edges."""
    return poly3(c, channel_edges)
