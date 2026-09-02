"""Parameter fit with bounded derivative-free optimizers.

The fit runs on a normalized ``[0,1]^d`` parameter space (each coordinate scaled
by its own bounds), so its step sizes and convergence tolerances are uniform
across parameters with very different physical ranges.  A single fit runs a
stage-1 optimizer to locate the basin, then a stage-2 optimizer from its result
to convergence.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy import optimize

from .fitparamspace import CALIB, RESOL
from .types import FitResult

# Progress-line print cadence for each stage (modulo, when verbose).
_STAGE1_PROGRESS_LOG_MODULO = 3
_STAGE2_PROGRESS_LOG_MODULO = 100

# The two fit stages, both on the unit cube: the stage-1 optimizer locates the
# basin, then the stage-2 optimizer polishes from its result to convergence.
# Each stage's iteration budget is supplied per call in `_fit_once`.
_STAGE1_OPTIMIZER = dict(
    method="L-BFGS-B",
    options=dict(ftol=1e-8, maxls=100),
)
_STAGE2_OPTIMIZER = dict(
    method="Nelder-Mead",
    options=dict(fatol=1e-6, adaptive=True),
)


def _bounds_arrays(bounds):
    """Split bounds into ``(lo, hi)`` float vectors.

    Every parameter must have a finite, non-degenerate interval.
    """
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    if np.any(hi - lo <= 0.0):
        raise ValueError("every parameter bound must have lo < hi")
    return lo, hi


def _normalize(x, lo, hi) -> np.ndarray:
    """Physical params ``x`` -> unit cube ``[0,1]^d`` via bounds ``(lo, hi)``."""
    return (np.asarray(x, dtype=float) - lo) / (hi - lo)


def _denormalize(u, lo, hi) -> np.ndarray:
    """Unit-cube params ``u`` -> physical ``x`` via bounds ``(lo, hi)``."""
    return lo + np.asarray(u, dtype=float) * (hi - lo)


def _progress_callback(model, tag: str, fit_progress_modulo: int,
                       to_physical=None):
    """Build a minimize callback that prints chi^2 and chi^2/ndof periodically.

    ``to_physical`` maps the minimizer's (normalized) point back to the physical
    parameter space before evaluating diagnostics; defaults to the identity.
    """
    if to_physical is None:
        def to_physical(x): return x
    state = {"iter": 0, "start": time.perf_counter()}

    def callback(xk, convergence=None):
        state["iter"] += 1
        if state["iter"] % fit_progress_modulo != 0:
            return
        avg_ms = (time.perf_counter() - state["start"]) / state["iter"] * 1e3
        det = model.detail(to_physical(np.asarray(xk, dtype=float)))
        chi2 = float(det.chi2)
        line_text = f"[calib] {tag} iter {state['iter']:<6d}:"
        if det.valid and det.ndof > 0 and np.isfinite(chi2):
            line_text += f" chi2/ndof = {chi2:>14.4f} / {det.ndof:<6d} = {chi2 / det.ndof:<14.6f}"
        else:
            line_text += f" chi2      = {chi2:<14.6f}"
        print(f"{line_text} ({avg_ms:.2f} ms/iter)", flush=True)
    return callback


def _fit_once(model, x0, bounds, stage1_maxiter, stage2_maxiter,
              tag: str = "fit", fit_progress_modulo: int = 0):
    """Two optimizer stages on the unit-cube space, returning physical params.

    ``x0`` and ``bounds`` are in physical units; the minimizer only ever sees the
    affine-normalized ``[0,1]^d`` parameters.  The stage-1 optimizer locates the
    basin, then the stage-2 optimizer polishes from its result to convergence.
    The returned ``OptimizeResult`` carries ``x`` mapped back to physical space
    and ``nfev`` summed over both stages.
    """
    x0 = np.asarray(x0, dtype=float)
    lo, hi = _bounds_arrays(bounds)

    def to_physical(u):
        """Unit-cube -> physical parameter vector."""
        return _denormalize(u, lo, hi)

    def objective(u):
        """Chi-square in physical space, evaluated at the minimizer's ``u``."""
        return model.evaluate(to_physical(u))

    u0 = _normalize(x0, lo, hi)
    unit_bounds = [(0.0, 1.0)] * len(u0)

    def make_callback(stage_tag, modulo):
        return (_progress_callback(model, stage_tag, modulo, to_physical)
                if modulo > 0 else None)

    # Stage 1: the stage-1 optimizer locates the basin.
    res = optimize.minimize(
        objective, u0, method=_STAGE1_OPTIMIZER["method"],
        bounds=unit_bounds,
        options=dict(maxiter=stage1_maxiter, **_STAGE1_OPTIMIZER["options"]),
        callback=make_callback(f"{tag} stage 1", fit_progress_modulo))
    nfev = int(res.nfev)

    # Stage 2: the stage-2 optimizer polishes from the stage-1 result, minimized
    # to convergence (reuses the same progress callback with a coarser modulo).
    stage2_modulo = (_STAGE2_PROGRESS_LOG_MODULO
                     if fit_progress_modulo > 0 else 0)
    res = optimize.minimize(
        objective, res.x, method=_STAGE2_OPTIMIZER["method"],
        bounds=unit_bounds,
        options=dict(maxiter=stage2_maxiter, **_STAGE2_OPTIMIZER["options"]),
        callback=make_callback(f"{tag} stage 2", stage2_modulo))
    res.nfev = int(res.nfev) + nfev
    res.x = to_physical(np.asarray(res.x, dtype=float))
    return res


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
    perr = np.sqrt(np.maximum(np.diag(cov), 0.0))

    calib_cov = np.asarray(cov[CALIB, CALIB], dtype=float)
    resol_cov = np.asarray(cov[RESOL, RESOL], dtype=float)
    calib_err = np.sqrt(np.maximum(np.diag(calib_cov), 0.0))
    resol_err = np.sqrt(np.maximum(np.diag(resol_cov), 0.0))
    return det, cov, perr, calib_cov, calib_err, resol_cov, resol_err


def _reconcile_success(det, success: bool, message: str):
    if not det.valid or not np.isfinite(det.chi2):
        return False, ("degenerate fit (insufficient data coverage or "
                       "infeasible parameters)")
    if success:
        return True, message
    return True, f"converged (optimizer stopped early: {message})"


def _finalize(model, q, success: bool = True, message: str = "",
              nfev: int = 0) -> FitResult:
    q = np.asarray(q, dtype=float)
    det, cov, perr, calib_cov, calib_err, resol_cov, resol_err = (
        _fit_statistics(model, q))
    success, message = _reconcile_success(det, success, message)
    chi2 = float(det.chi2)
    ndof = int(det.ndof)

    return FitResult(
        success=success,
        message=message,
        nfev=int(nfev),
        params=q,
        errors=perr,
        names=model.param_space.names,
        chi2=chi2,
        ndof=ndof,
        reduced_chi2=chi2 / ndof if ndof > 0 else np.nan,
        cov=cov,
        calib_params=np.asarray(q[CALIB], dtype=float),
        calib_errors=calib_err, calib_cov=calib_cov,
        resol_params=np.asarray(q[RESOL], dtype=float),
        resol_errors=resol_err, resol_cov=resol_cov,
        detail=det,
    )


def run_fit(model, x0=None, stage1_maxiter: int = 300,
            stage2_maxiter: int = 100000, verbose: bool = True) -> FitResult:
    if x0 is None:
        x0 = model.x0
    x0 = np.asarray(x0, dtype=float)
    if not np.isfinite(x0).all():
        raise ValueError("fit starting point x0 contains NaN/inf")
    x0 = np.clip(x0, [b[0] for b in model.bounds], [b[1]
                 for b in model.bounds])
    if not model.is_valid(x0):
        print("[calib] warning: the starting point is degenerate (insufficient "
              "data coverage); the fit may not be meaningful", file=sys.stderr)
    best = _fit_once(model, x0, model.bounds, stage1_maxiter, stage2_maxiter,
                     tag="fit",
                     fit_progress_modulo=(_STAGE1_PROGRESS_LOG_MODULO
                                          if verbose else 0))
    return _finalize(model, best.x, success=bool(best.success),
                     message=str(best.message), nfev=int(best.nfev))
