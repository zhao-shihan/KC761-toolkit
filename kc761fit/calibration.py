"""Energy calibration E(x) = c3 x^3 + c2 x^2 + c1 x + c0 (x = channel)."""

from __future__ import annotations

import numpy as np

CALIB_ENERGIES = np.array([60.0, 609.0, 1461.0, 2614.0])
INIT_X = np.array([160.0, 500.0, 900.0, 1350.0])

MONOTONICITY_PENALTY = 10.0
_MAX_COND = 1e14


def poly3(c: np.ndarray | list[float], x: np.ndarray | float) -> np.ndarray:
    c0, c1, c2, c3 = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    return c3 * x**3 + c2 * x**2 + c1 * x + c0


def _nan_c(jacobian: bool):
    c = np.full(4, np.nan)
    if jacobian:
        return c, np.full((4, 4), np.nan)
    return c


def channels_to_c(x: np.ndarray | list[float],
                  energies: np.ndarray | list[float] = CALIB_ENERGIES,
                  jacobian: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    e = np.asarray(energies, dtype=float)
    v = np.stack([np.ones_like(x), x, x**2, x**3], axis=1)
    if not np.isfinite(x).all():
        return _nan_c(jacobian)
    try:
        if np.linalg.cond(v) > _MAX_COND:
            return _nan_c(jacobian)
        c = np.linalg.solve(v, e)
    except np.linalg.LinAlgError:
        return _nan_c(jacobian)
    if not jacobian:
        return c
    try:
        v_inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return _nan_c(jacobian)
    jac = np.empty((4, 4))
    dv = np.zeros_like(v)
    for j in range(4):
        dv[:] = 0.0
        dv[j] = [0.0, 1.0, 2.0 * x[j], 3.0 * x[j] ** 2]
        jac[:, j] = -v_inv @ (dv @ c)
    return c, jac


def monotonicity_penalty(x: np.ndarray | list[float]) -> float:
    gaps = np.diff(np.asarray(x, dtype=float))
    viol = np.maximum(-gaps, 0.0)
    return MONOTONICITY_PENALTY * float(np.sum(viol * viol))
