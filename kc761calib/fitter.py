"""Parameter fit with the bounded Nelder-Mead derivative-free optimizer."""

from __future__ import annotations

import sys

import numpy as np
from scipy import optimize

from .params import CHANNELS, RESOLUTIONS
from .response import channels_to_c, resol_to_b
from .types import FitResult


def _fit_once(model, x0, bounds, maxiter):
    x0 = np.asarray(x0, dtype=float)
    return optimize.minimize(
        model.evaluate, x0, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-3, adaptive=True))


def finite_difference_jacobian(fun, x0: np.ndarray,
                               bounds: list[tuple[float, float]],
                               rel_step: float = 1e-4) -> np.ndarray:
    """Central-difference Jacobian of a vector-valued fun.

    Steps are clipped to stay inside bounds.  When one probe falls outside
    the bounds or into a rejected (non-finite) region -- e.g. at an optimum
    sitting on the feasibility boundary -- the column degrades gracefully
    to a one-sided difference; only if neither side evaluates is it NaN.
    """
    x0 = np.asarray(x0, dtype=float)
    f0 = np.asarray(fun(x0), dtype=float)
    jac = np.empty((len(f0), len(x0)))
    for k in range(len(x0)):
        h = rel_step * max(1.0, abs(x0[k]))
        lo, hi = bounds[k]
        h = min(h, hi - x0[k], x0[k] - lo)
        if not np.isfinite(h) or h <= 0.0:
            jac[:, k] = np.nan
            continue
        q_p, q_m = np.array(x0), np.array(x0)
        q_p[k] += h
        q_m[k] -= h
        f_p = np.asarray(fun(q_p))
        f_m = np.asarray(fun(q_m))
        ok_p, ok_m = bool(np.all(np.isfinite(f_p))), bool(
            np.all(np.isfinite(f_m)))
        if ok_p and ok_m:
            jac[:, k] = (f_p - f_m) / (2.0 * h)
        elif ok_p:
            jac[:, k] = (f_p - f0) / h
        elif ok_m:
            jac[:, k] = (f0 - f_m) / h
        else:
            jac[:, k] = np.nan
    return jac


def _covariance(model, q) -> np.ndarray:
    """Inverse Fisher matrix from frozen-mask residuals; NaN when singular."""
    mask_list = model.masks(q)
    def fun(qq): return model.residuals(qq, mask_list)
    jac = finite_difference_jacobian(fun, q, model.bounds)
    n_q = len(q)
    if (np.all(np.isfinite(jac)) and jac.shape[0] > n_q):
        try:
            return np.linalg.inv(jac.T @ jac)
        except np.linalg.LinAlgError:
            pass
    return np.full((n_q, n_q), np.nan)


def _fit_statistics(model, q):
    det = model.detail(q)
    cov = _covariance(model, q)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    c, jac_c = channels_to_c(q[CHANNELS], jacobian=True)
    b, jac_b = resol_to_b(q[RESOLUTIONS], jacobian=True)
    cov_c = jac_c @ cov[CHANNELS, CHANNELS] @ jac_c.T
    cov_b = jac_b @ cov[RESOLUTIONS, RESOLUTIONS] @ jac_b.T
    perr_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    perr_b = np.sqrt(np.clip(np.diag(cov_b), 0, None))
    return det, cov, perr, cov_c, perr_c, cov_b, perr_b


def _reconcile_success(det, success: bool, message: str):
    if not det.valid or not np.isfinite(det.chi2):
        return False, ("degenerate fit (insufficient data coverage or "
                       "infeasible parameters)")
    if success:
        return True, message
    return True, f"converged (Nelder-Mead stopped early: {message})"


def _finalize(model, q, success: bool = True, message: str = "",
              nfev: int = 0) -> FitResult:
    q = np.asarray(q, dtype=float)
    det, cov, perr, cov_c, perr_c, cov_b, perr_b = _fit_statistics(model, q)
    success, message = _reconcile_success(det, success, message)
    chi2 = float(det.chi2)
    ndof = int(det.ndof)

    return FitResult(
        success=success,
        message=message,
        nfev=int(nfev),
        params=q,
        errors=perr,
        names=model.space.names,
        chi2=chi2,
        ndof=ndof,
        reduced_chi2=chi2 / ndof if ndof > 0 else np.nan,
        cov=cov,
        params_c=np.asarray(det.c), errors_c=perr_c, cov_c=cov_c,
        params_b=np.asarray(det.b), errors_b=perr_b, cov_b=cov_b,
        model=model,
        detail=det,
    )


def _fit_passes(model, x0, bounds, maxiter, n_passes, verbose):
    """Fit repeatedly, re-freezing the energy grid from the fitted anchors."""
    x0 = np.asarray(x0, dtype=float)
    if not np.isfinite(x0).all():
        raise ValueError("fit starting point x0 contains NaN/inf")
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    if not model.is_valid(x0):
        print("[calib] warning: the starting point is degenerate (insufficient "
              "data coverage); the fit may not be meaningful", file=sys.stderr)
    n_passes = max(1, int(n_passes))

    m = None
    best = None
    nfev_total = 0
    for k in range(1, n_passes + 1):
        st = x0 if best is None else best.x
        m_new = model.rebuilt(st[CHANNELS])
        if not (m_new.grid_ok() and m_new.is_valid(st)):
            break  # report the previous, still-valid pass
        m = m_new
        best = _fit_once(m, st, bounds, maxiter)
        nfev_total += int(best.nfev)
        tag = ("grid from initial calibration" if k == 1
               else "grid from fitted calibration")
        n_bins = sum(len(mm.grid_centers) for mm in m.models)
        if verbose:
            print(f"[calib] pass {k}: {n_bins} bins ({tag}), "
                  f"best chi2 = {best.fun:.2f}, nfev = {best.nfev}")
    return m, best, nfev_total


def run_fit(model, x0=None, maxiter: int = 10000, n_passes: int = 5,
            verbose: bool = True) -> FitResult:
    if x0 is None:
        x0 = model.x0
    m, best, nfev_total = _fit_passes(model, x0, model.bounds, maxiter,
                                      n_passes, verbose)
    if best is None:
        # No admissible grid even at the start point; report honestly.
        return _finalize(model, x0, success=False,
                         message="degenerate fit: inadmissible starting "
                                 "grid for all passes")
    return _finalize(m, best.x, success=bool(best.success),
                     message=str(best.message), nfev=nfev_total)


def make_x0(model, scale_values: list[float] | None = None) -> np.ndarray:
    """Model default start vector with optional per-dataset scale overrides."""
    x0 = np.array(model.x0, dtype=float, copy=True)
    if scale_values:
        tail = model.space.scales
        for k, v in enumerate(scale_values):
            if v is not None:
                x0[tail.start + k] = v
    return x0
