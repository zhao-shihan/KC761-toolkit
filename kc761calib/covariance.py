"""Parameter covariance from numerically differentiated residuals.

Near the least-squares minimum of the weighted residuals of
:class:`kc761calib.globalfit.GlobalFitModel` the covariance is estimated by
the Gauss-Newton formula ``cov = s^2 (J^T J)^-1`` with ``J = dr/dq`` and
``s^2 = chi^2 / ndof`` the reduced chi-square -- the same residual-variance
rescaling as ``scipy.optimize.curve_fit``: ``chi^2/ndof ~ 1`` leaves the
covariance unchanged, a poor fit inflates the reported uncertainties.

``J`` is always obtained by finite differences (the model has no analytic
derivatives).  The step balances the ``O(h^2)`` truncation error against the
``O(eps/h)`` round-off error, giving the relative step ``eps^(1/3) ~ 6e-6``
for central differences (``eps^(1/2)`` for the one-sided differences used at
bounds), based on float64 eps (see :func:`numerical_jacobian`).  The
covariance is only the sampling seed of the reported profile covariance
(:mod:`kc761calib.profilecov`) and of ``tmp/chi2_shape.py``; it is
assembled by inverting ``J^T J`` on the identifiable subspace
(SVD rank test); undetermined parameters report NaN uncertainties instead
of a misleading finite or zero value.
"""

from __future__ import annotations

import numpy as np

_CENTRAL_STEP = np.finfo(np.float64).eps ** (1.0 / 3.0)
_ONESIDED_STEP = np.finfo(np.float64).eps ** 0.5


def numerical_jacobian(fun, x: np.ndarray,
                       bounds: list[tuple[float, float]] | None = None,
                       *,
                       rel_step: float = _CENTRAL_STEP,
                       x_scale: np.ndarray | list[float] | None = None,
                       ) -> np.ndarray:
    """Finite-difference Jacobian of a vector-valued ``fun`` at ``x``.

    Column ``k`` differentiates w.r.t. parameter ``x[k]``.  The step is
    ``rel_step * max(|x[k]|, scale[k])`` with ``scale[k]`` the per-parameter
    typical magnitude (default: 1.0).  Central differences are used when both
    probes stay inside ``bounds`` and evaluate finitely; near a bound the
    column degrades to a one-sided difference with the smaller optimal step,
    and to NaN only when neither side can be probed.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("x must have at least one parameter")
    if x_scale is None:
        x_scale = np.ones(n)
    x_scale = np.asarray(x_scale, dtype=float)
    if x_scale.shape != (n,) or not np.all(np.isfinite(x_scale)):
        raise ValueError("x_scale must be a finite per-parameter vector")
    if bounds is not None and len(bounds) != n:
        raise ValueError("bounds must provide one (lo, hi) pair per parameter")

    f0 = np.asarray(fun(x), dtype=float)
    jac = np.empty((len(f0), n))

    for k in range(n):
        scale = float(x_scale[k]) * max(1.0, abs(float(x[k])))
        lo = -np.inf if bounds is None else float(bounds[k][0])
        hi = np.inf if bounds is None else float(bounds[k][1])
        hc = float(rel_step) * scale

        if x[k] - hc >= lo and x[k] + hc <= hi:
            q_p, q_m = np.array(x, copy=True), np.array(x, copy=True)
            q_p[k] += hc
            q_m[k] -= hc
            f_p = np.asarray(fun(q_p), dtype=float)
            f_m = np.asarray(fun(q_m), dtype=float)
            if (f_p.shape == f0.shape and f_m.shape == f0.shape
                    and np.all(np.isfinite(f_p)) and np.all(np.isfinite(f_m))):
                jac[:, k] = (f_p - f_m) / (2.0 * hc)
                continue

        # One-sided difference: probe inward with the one-sided optimal step,
        # preferring the direction with room to spare (handles optima sitting
        # exactly on a bound without collapsing the step to ~0).
        h1 = float(_ONESIDED_STEP) * scale
        candidates = sorted(
            ((sign, min(h1, float(room))) for sign, room in
             ((1.0, hi - x[k]), (-1.0, x[k] - lo)) if room > 0.0),
            key=lambda pair: pair[1], reverse=True)
        for sign, h in candidates:
            q_p = np.array(x, copy=True)
            q_p[k] += sign * h
            if q_p[k] == x[k]:  # step too small to be representable
                continue
            f_p = np.asarray(fun(q_p), dtype=float)
            if f_p.shape == f0.shape and np.all(np.isfinite(f_p)):
                jac[:, k] = sign * (f_p - f0) / h
                break
        else:
            jac[:, k] = np.nan
    return jac


def _inverse_fisher(jac: np.ndarray) -> np.ndarray:
    """Inverse of ``J^T J`` on the identifiable subspace; NaN elsewhere.

    A singular-value rank test (relative tolerance ``max(shape) * eps``)
    separates the identifiable directions from the undetermined ones.  A
    parameter whose coordinate direction has any component in the dropped
    (null) subspace cannot be estimated individually, so its variance -- and
    its covariances with every other parameter -- is reported as NaN.
    """
    jtj = jac.T @ jac
    jtj = 0.5 * (jtj + jtj.T)  # remove numerical asymmetry before SVD
    u, s, vt = np.linalg.svd(jtj)
    tol = float(s[0]) * max(jtj.shape) * np.finfo(float).eps
    keep = s > tol
    inv_s = np.zeros_like(s)
    np.divide(1.0, s, out=inv_s, where=keep)
    cov = (u * inv_s) @ vt
    if not keep.all():
        null = u[:, ~keep]
        undetermined = np.any(np.abs(null) > 1e-8, axis=1)
        cov[undetermined, :] = np.nan
        cov[:, undetermined] = np.nan
    return cov


def parameter_covariance(model, q: np.ndarray, *,
                         reduced_chi2: float | None = None,
                         rel_step: float = _CENTRAL_STEP) -> np.ndarray:
    """Covariance of the fitted parameters from numerically differentiated residuals.

    ``reduced_chi2``, when finite and positive, rescales the covariance by
    ``s^2 = chi^2/ndof``; pass None for the unscaled ``(J^T J)^-1``.
    Parameters that cannot be probed or are undetermined (rank-deficient
    Jacobian) report NaN variances.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    cov = np.full((n, n), np.nan)

    def fun(qq): return model.residuals(qq)

    jac = numerical_jacobian(fun, q, model.bounds, rel_step=rel_step)
    finite_cols = np.all(np.isfinite(jac), axis=0)
    if finite_cols.any():
        # Drop any residual row that is not finite on the retained columns
        # (defensive: a single bad row would poison the whole J^T J).
        finite_rows = np.all(np.isfinite(jac[:, finite_cols]), axis=1)
        jac_f = jac[finite_rows][:, finite_cols]
        if jac_f.shape[0] > int(finite_cols.sum()):
            cov[np.ix_(finite_cols, finite_cols)] = _inverse_fisher(jac_f)
    if (reduced_chi2 is not None and np.isfinite(reduced_chi2)
            and reduced_chi2 > 0.0):
        cov = cov * reduced_chi2
    return cov
