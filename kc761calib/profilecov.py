"""Profile-likelihood covariance of the fitted core parameters.

The chi-square surface of a peak-alignment model is only locally
quadratic (see the module history): with the scale parameters fixed, the
differential peak shifts produce a genuinely non-quadratic profile, so a
quadratic fit over any finite sampling region is ill-conditioned.  The
robust quantity of such a surface is the one-dimensional profile: for
each of the 7 core parameters a line search along the +- axis directions
locates the ``dchi2 = 1`` crossings, giving the (possibly asymmetric)
1-sigma widths ``sigma_lo``/``sigma_hi``.  The crossings live far above
the surface's fine kink noise and do not require the reported optimum to
be an exact minimum -- the asymmetry itself absorbs a small offset.

The symmetric covariance used for downstream propagation combines these
widths with the linearized correlation structure:

    sigma_sym = (sigma_lo + sigma_hi) / 2,      cov = D rho_lin D,

with ``rho_lin`` the correlation matrix of the finite-difference
linearized covariance (:func:`kc761calib.covariance.parameter_covariance`
machinery on the core block): correlations are far less sensitive to the
surface's nonlinearity than the amplitudes.  The asymmetric widths are
kept as reporting metadata (``err_lo``/``err_hi``); the covariance
matrix itself stays symmetric, exactly as every downstream consumer
expects.  A parameter pinned at a fit bound (one side infeasible)
reports that side as 0 and is flagged bound-limited.
"""

from __future__ import annotations

import warnings

import numpy as np

from .covariance import _inverse_fisher, numerical_jacobian
from .fitparamspace import CORE, N_CORE

_WIDTH_EXCESS = 20.0  # warn when a fitted width exceeds this x its seed


def _linear_seed_covariance(model, q, q_core, bounds_core) -> np.ndarray:
    """Linearized 7x7 covariance of the core parameters (scale seed).

    Finite-difference Jacobian of the residual vector w.r.t. the core
    parameters only (scales stay fixed at the optimum, exactly as in the
    profile scans), inverted on the identifiable subspace.
    """

    def fun(qc):
        qq = np.array(q, copy=True)
        qq[CORE] = qc
        return model.residuals(qq)

    jac = numerical_jacobian(fun, q_core, bounds_core)
    finite_cols = np.all(np.isfinite(jac), axis=0)
    cov0 = np.full((N_CORE, N_CORE), np.nan)
    if finite_cols.any():
        finite_rows = np.all(np.isfinite(jac[:, finite_cols]), axis=1)
        jac_f = jac[finite_rows][:, finite_cols]
        if jac_f.shape[0] > int(finite_cols.sum()):
            cov0[np.ix_(finite_cols, finite_cols)] = _inverse_fisher(jac_f)
    return cov0


def _dchi2(model, q, q_core, delta, chi2_min):
    """dchi2 at the probe ``q_core + delta``; None if infeasible."""
    qq = np.array(q, copy=True)
    qq[CORE] = q_core + delta
    try:
        d = model.evaluate(qq) - chi2_min
    except ValueError:
        return None  # infeasible probe (e.g. non-monotone calibration)
    return d if np.isfinite(d) else None


def _axis_crossing(model, q, q_core, k, sign, scale, chi2_min, bounds):
    """The dchi2 = 1 crossing of the axis profile of parameter ``k``.

    ``sign`` selects the direction (+-1); ``scale`` is the search unit
    (the linearized width).  Probes outside the fit bounds are
    infeasible (the model's resolution enters through ``b_p^2`` and the
    scale model through squares, so probing past a bound would find
    mirrored crossings that do not exist).  Returns the crossing as a
    multiple of ``scale``, or NaN when the side is infeasible
    (parameter at a fit bound -- pure scheme A).
    """
    lo_b, hi_b = bounds[k]

    def at(t):
        value = q_core[k] + sign * t * scale
        if value < lo_b or value > hi_b:
            return None
        delta = np.zeros(N_CORE)
        delta[k] = sign * t * scale
        return _dchi2(model, q, q_core, delta, chi2_min)

    f1 = at(1.0)
    if f1 is None:
        return np.nan
    if f1 <= 1.0:
        # Expand until the profile crosses 1 (the profile may dip below
        # zero first when the reported optimum is offset from the true
        # minimum; the first crossing from above is the 1-sigma limit).
        t = 1.0
        while f1 is not None and f1 <= 1.0 and t < 1024.0:
            t *= 2.0
            f1 = at(t)
        if f1 is None or t >= 1024.0:
            return np.nan
        lo, hi = t / 2.0, t
    else:
        lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f = at(mid)
        if f is None or f > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def profile_covariance(model, q,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """Profile-likelihood covariance of the 7 core parameters.

    ``model`` is the fitted :class:`kc761calib.globalfit.GlobalFitModel`
    and ``q`` the fitted parameter vector (scale parameters stay fixed at
    the optimum throughout).  Returns ``(cov7, sigma7, bound_limited,
    err_lo, err_hi)``: the symmetric 7x7 covariance in the fit (internal
    slope) basis, its diagonal sqrt, the bound-limited flags, and the
    asymmetric 1-sigma widths (metadata only; bound-side infeasibility
    reports that side as 0).
    """
    q = np.asarray(q, dtype=float)
    q_core = np.asarray(q[CORE], dtype=float)
    bounds = [tuple(b) for b in model.bounds[CORE]]
    chi2_min = float(model.evaluate(q))

    seed = _linear_seed_covariance(model, q, q_core, bounds)
    # Search units: the linearized widths (fallback: a tenth of the bound
    # width); the crossings themselves are unit-independent.
    scale = np.sqrt(np.maximum(np.diag(seed), 0.0))
    for k in range(N_CORE):
        if not np.isfinite(scale[k]) or scale[k] <= 0.0:
            scale[k] = 0.1 * (bounds[k][1] - bounds[k][0])
    scale0 = scale.copy()

    lo = np.full(N_CORE, np.nan)
    hi = np.full(N_CORE, np.nan)
    bound_limited = np.zeros(N_CORE, dtype=bool)
    for k in range(N_CORE):
        t_neg = _axis_crossing(model, q, q_core, k, -1.0, scale[k],
                               chi2_min, bounds)
        t_pos = _axis_crossing(model, q, q_core, k, 1.0, scale[k],
                               chi2_min, bounds)
        if np.isfinite(t_neg):
            lo[k] = t_neg * scale[k]
        else:
            lo[k] = 0.0
            bound_limited[k] = True
        if np.isfinite(t_pos):
            hi[k] = t_pos * scale[k]
        else:
            hi[k] = 0.0
            bound_limited[k] = True

    # Symmetric width: the mean of the two crossings for interior
    # parameters, but the feasible side's width for bound-limited ones
    # (averaging against the zeroed infeasible side would halve it).
    sigma_sym = 0.5 * (lo + hi)
    sigma_sym[bound_limited] = np.maximum(lo[bound_limited],
                                          hi[bound_limited])
    excess = sigma_sym > _WIDTH_EXCESS * scale0
    if excess.any():
        warnings.warn(
            f"profile covariance: {int(excess.sum())} width(s) exceed "
            f"{_WIDTH_EXCESS:g}x their linear seed; the parameters are "
            f"flagged bound-limited", RuntimeWarning)
        bound_limited |= excess

    # Symmetric covariance: profile widths x linearized correlations.
    cov_lin = seed
    var_lin = np.diag(cov_lin)
    rho = np.zeros((N_CORE, N_CORE))
    for j in range(N_CORE):
        for k in range(N_CORE):
            denom = np.sqrt(max(var_lin[j], 0.0) * max(var_lin[k], 0.0))
            rho[j, k] = cov_lin[j, k] / denom if denom > 0.0 else np.nan
    np.fill_diagonal(rho, 1.0)
    # An undetermined linear covariance (any NaN) makes the whole
    # parameter row/column undefined, including its diagonal -- a finite
    # diagonal would defeat the downstream NaN checks and let NaN
    # propagate silently through the quadratic forms.
    bad = np.any(np.isnan(rho), axis=1)
    rho[bad, :] = np.nan
    rho[:, bad] = np.nan
    cov7 = sigma_sym[:, None] * rho * sigma_sym[None, :]

    return cov7, sigma_sym, bound_limited, lo, hi
