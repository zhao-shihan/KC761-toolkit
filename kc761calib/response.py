"""Detector response: deposition-energy calibration E(channel) and resolution
sigma^2(E).

The calibration is cubic in the channel number, parameterized by the
intercept plus three slopes ``(c0, k1, k2, k3)``:

   E(channel) = c0 + k1 channel + (4 k2 - 3 k1 - k3)/(2 channel_max) channel^2
                 + 2 (k1 - 2 k2 + k3)/(3 channel_max^2) channel^3,
   E'(0) = k1,   E'(channel_max/2) = k2,   E'(channel_max) = k3.

Monotonicity on [0, channel_max] is guaranteed by the fit bounds; the
reported plain cubic ``(c0, c1, c2, c3)`` is recovered from the slopes
through the constant map ``c0k1k2k3_to_c0c1c2c3``.

``sigma^2`` is a quadratic Bernstein polynomial in ``t = E / RESOL_E_REF``
(control values ``(b0^2, b1^2, b2^2)``):

   sigma^2(t) = (1-t)^2 b0^2 + 2(1-t)t b1^2 + t^2 b2^2

``t`` is clamped at 0, so ``sigma`` saturates at ``b0`` for ``E <= 0`` (the
polynomial is not usable there) and the variance floor ``MIN_SIGMA`` is
purely numerical safety; above ``RESOL_E_REF`` the polynomial runs
unclamped.  The response matrix is assembled in :mod:`kc761calib.folding`.
"""

from __future__ import annotations

import numba
import numpy as np

from .util import _bernstein_basis

# --- initial values and fit bounds ---

N_CALIB = 4  # (c0, k1, k2, k3)
INIT_CALIB = np.array([-180.0, 1.5, 2.5, 3.5])
BOUNDS_CALIB = [(-300.0, -100.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

N_RESOL = 3  # (b0, b1, b2)
RESOL_E_REF = 2000.0  # keV, reference energy
MIN_SIGMA = 0.001  # keV, sigma floor
INIT_RESOL = np.array([2.0, 20.0, 40.0])
BOUNDS_RESOL = [(0.0, 10.0), (0.0, 80.0), (0.0, 100.0)]

PARAM_NAMES_CORE = ["c0", "k1", "k2", "k3", "b0", "b1", "b2"]
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_K = ["k1", "k2", "k3"]
PARAM_NAMES_B = ["b0", "b1", "b2"]


# --- canonical model formulas ---
#
# The single source of truth for the formula texts: stored verbatim in the
# ROOT export (kc761calib.export) and printed by the console report
# (kc761calib.report).  They must stay consistent with calib_model and
# resol_sigma_model.

# Plain-cubic parameterization reported by c0k1k2k3_to_c0c1c2c3.
CALIB_FORMULA = ("E(ch) = c0 + c1*ch + c2*ch^2 + c3*ch^3"
                 "   (E in keV, ch = channel)")

# Bernstein sigma^2 in t = max(E, 0)/RESOL_E_REF (see module docstring).
RESOL_FORMULA = (f"sigma^2(E) = (1-t)^2*b0^2 + 2*(1-t)*t*b1^2 + t^2*b2^2,"
                 f"   t = max(E, 0)/{RESOL_E_REF:g} keV"
                 f"   (sigma in keV, saturated at b0 for E <= 0)")


# --- energy calibration E(ch) ---

def poly_basis(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Monomial basis vector [1, x, ..., x^degree] on the last axis."""
    x = np.asarray(x, dtype=float)
    return np.stack([x**k for k in range(degree + 1)], axis=-1)


@numba.njit(inline="always", cache=True)
def calib_model(calib_params, channel, channel_max):
    """Cubic E(channel) from ``(c0, k1, k2, k3)`` (module docstring).

    ``channel_max`` is the maximum channel number of the acquisition, a
    fixed constant of the data; ``calib_params`` is a float64 array of
    length 4 and ``channel`` a float64 scalar or array.
    """
    c0 = calib_params[0]
    k1 = calib_params[1]
    k2 = calib_params[2]
    k3 = calib_params[3]
    return (c0 + k1 * channel
            + (4 * k2 - 3 * k1 - k3) / (2.0 * channel_max) * channel**2
            + 2.0 * (k1 - 2.0 * k2 + k3) / (3.0 * channel_max**2) * channel**3)


def c0k1k2k3_to_c0c1c2c3(calib_params: np.ndarray | list[float], channel_max: float) -> np.ndarray:
    """Plain cubic ``[c0, c1, c2, c3]`` from the slope parameterization (module
    docstring: ``c1 = k1`` and the slope-image map for ``c2``, ``c3``)."""
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


def _basis_transform_cov(cov: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Similarity-transform a covariance into a new basis, NaN-aware.

    ``out = T @ cov @ T^T``.  A NaN covariance entry (an undetermined
    parameter) is kept NaN-exactly where it truly propagates: any product
    term with a NaN covariance and a nonzero transform weight makes the
    target entry NaN, while exact-zero weights -- which would otherwise
    produce spurious ``NaN * 0`` artifacts of the dense transform -- do
    not.  ``cov`` and ``transform`` must be square of equal size.
    """
    cov = np.asarray(cov, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if cov.shape[0] != cov.shape[1] or transform.shape != cov.shape:
        raise ValueError(
            f"transform shape {transform.shape} does not match the square "
            f"covariance shape {cov.shape}")
    nan = np.isnan(cov)
    work = np.where(nan, 0.0, cov)
    out = transform @ work @ transform.T
    undefined = (np.abs(transform) @ nan.astype(float)
                 @ np.abs(transform).T) > 0.0
    out[undefined] = np.nan
    return out


def reported_core_cov(core_cov: np.ndarray, channel_max: float) -> np.ndarray:
    """7x7 core covariance in the reported basis ``(c0..c3, b0..b2)``.

    The internal basis ``(c0, k1, k2, k3, b0, b1, b2)`` maps to the
    reported one through the constant similarity transform
    ``diag(jac_c0k1k2k3, I_3)``, with the same NaN-aware semantics as
    :func:`reported_calib`.
    """
    core_cov = np.asarray(core_cov, dtype=float)
    n = N_CALIB + N_RESOL
    if core_cov.shape != (n, n):
        raise ValueError(
            f"core covariance must have shape ({n}, {n}), got {core_cov.shape}")
    transform = np.zeros((n, n), dtype=float)
    transform[:N_CALIB, :N_CALIB] = jac_c0k1k2k3(channel_max)
    transform[N_CALIB:, N_CALIB:] = np.eye(N_RESOL)
    return _basis_transform_cov(core_cov, transform)


def reported_calib(calib_params: np.ndarray | list[float], calib_cov: np.ndarray,
                   channel_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reported (c0, c1, c2, c3) values, errors and covariance.

    Display-only: the coefficients come from ``c0k1k2k3_to_c0c1c2c3`` and the
    covariance is similarity-transformed by the same map, NaN-aware (a
    reported entry is NaN exactly when it depends on an undetermined
    parameter).
    """
    c = c0k1k2k3_to_c0c1c2c3(calib_params, channel_max)
    cov = _basis_transform_cov(calib_cov, jac_c0k1k2k3(channel_max))
    err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return c, err, cov

# --- resolution: Gaussian sigma(E) ---


@numba.njit(inline="always", cache=True)
def gaussian_pdf(d, sigma):
    """Normal density at offset ``d`` for ``sigma > 0``.

    ``exp(-d^2 / (2 sigma^2)) / (sqrt(2 pi) sigma)``, elementwise over
    float64 scalars or broadcastable arrays.  ``np.exp``/``np.sqrt`` are
    numba intrinsics that constant-fold inside the fused folding kernel.
    """
    return np.exp(-0.5 * (d / sigma)**2) / (np.sqrt(2.0 * np.pi) * sigma)


@numba.njit(cache=True)
def _sigma_intermediates(resol_params, energy):
    """Clipped ``t``, the Bernstein basis, ``var`` and ``sigma`` of the model.

    Shared by :func:`resol_sigma_model` and :func:`resol_sigma_model_grad`,
    so value and derivatives can never drift apart: ``var = sum_k B_k(t)
    b_k^2`` with ``sigma = sqrt(max(var, MIN_SIGMA**2))``.  ``resol_params``
    is a float64 array of length 3 and ``energy`` a float64 scalar or array.
    """
    e = np.asarray(energy, dtype=np.float64)
    b_sq = np.asarray(resol_params, dtype=np.float64) ** 2
    t = np.clip(e, 0.0, np.inf) / RESOL_E_REF
    basis = _bernstein_basis(t, 2)  # (..., 3)
    var = np.dot(basis, b_sq)
    sigma = np.sqrt(np.maximum(var, MIN_SIGMA**2))
    return t, basis, var, sigma


@numba.njit(cache=True)
def resol_sigma_model_grad(resol_params, energy):
    """Elementwise derivatives of ``sigma(E)``.

    Returns ``(ds_dE, ds_db)`` with ``ds_db`` having a last axis over
    ``k = 0..2``.  Both the basis and the clamps are shared with
    :func:`resol_sigma_model` through :func:`_sigma_intermediates`, so the
    derivative is exact: it vanishes where ``t`` is clamped (``E <= 0``) or
    where the ``MIN_SIGMA`` floor is active (one-sided derivative).
    """
    e = np.asarray(energy, dtype=np.float64)
    b_sq = np.asarray(resol_params, dtype=np.float64) ** 2
    t, basis, var, sigma = _sigma_intermediates(resol_params, e)
    ds_dvar = np.where(var > MIN_SIGMA**2, 0.5 / sigma, 0.0)
    # dB/dt of the degree-2 Bernstein basis: [-2(1-t), 2(1-2t), 2t],
    # assembled column-wise so 0-D (scalar) arrays stay supported.
    db_dt = np.empty(e.shape + (3,), dtype=np.float64)
    db_dt[..., 0] = 2.0 * (t - 1.0)
    db_dt[..., 1] = 2.0 * (1.0 - 2.0 * t)
    db_dt[..., 2] = 2.0 * t
    dvar_dt = np.sum(db_dt * b_sq, axis=-1)
    ds_dE = ds_dvar * dvar_dt * (e > 0.0) / RESOL_E_REF
    ds_db = np.empty(e.shape + (3,), dtype=np.float64)
    for k in range(3):
        ds_db[..., k] = ds_dvar * basis[..., k] * (2.0 * resol_params[k])
    return ds_dE, ds_db


@numba.njit(cache=True)
def resol_sigma_model(resol_params, energy):
    """sigma(E) from ``resol_params = [b0, b1, b2]`` (in keV).

    The module's Bernstein model: ``t`` clamped at 0, saturating at ``b0``
    for ``E <= 0`` and running unclamped above ``RESOL_E_REF`` (the
    variance stays non-negative because the Bernstein basis is a
    non-negative partition of unity on ``t`` in [0, 1]).  ``resol_params``
    is a float64 array of length 3 and ``energy`` a float64 scalar or
    array.
    """
    return _sigma_intermediates(resol_params, energy)[3]
