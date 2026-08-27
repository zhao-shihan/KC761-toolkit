"""Parameter fit with the bounded Nelder-Mead derivative-free optimizer."""

from __future__ import annotations

import sys

import numpy as np
from scipy import optimize

from .calibration import channels_to_c
from .resolution import resol_to_a
from .types import FitResult


def _fit_once(model, x0, bounds, maxiter):
    x0 = np.asarray(x0, dtype=float)
    return optimize.minimize(
        model.evaluate, x0, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-3, adaptive=True))


def _jacobian(model, q0, rel_step=1e-4):
    det0 = model.detail(q0)
    if det0.mask is None:
        return None
    mask = det0.mask
    r0 = model.residuals(q0, mask)
    if np.any(~np.isfinite(r0)):
        return None
    n_q = len(q0)
    jac = np.empty((len(r0), n_q))
    for k in range(n_q):
        h = rel_step * max(1.0, abs(q0[k]))
        lo, hi = model.bounds[k]
        if q0[k] + h >= hi:
            q_m = np.array(q0)
            q_m[k] -= h
            r_m = model.residuals(q_m, mask)
            jac[:, k] = (r0 - r_m) / h if np.all(np.isfinite(r_m)) else np.nan
        elif q0[k] - h <= lo:
            q_p = np.array(q0)
            q_p[k] += h
            r_p = model.residuals(q_p, mask)
            jac[:, k] = (r_p - r0) / h if np.all(np.isfinite(r_p)) else np.nan
        else:
            q_hi = np.array(q0)
            q_lo = np.array(q0)
            q_hi[k] += h
            q_lo[k] -= h
            r_hi = model.residuals(q_hi, mask)
            r_lo = model.residuals(q_lo, mask)
            if (np.any(~np.isfinite(r_hi)) or np.any(~np.isfinite(r_lo))) \
                    and h > 1e-12:
                h_small = 1e-6 * max(1.0, abs(q0[k]))
                q_s = np.array(q0)
                q_s[k] += h_small
                q_m = np.array(q0)
                q_m[k] -= h_small
                r_ps = model.residuals(q_s, mask)
                r_pm = model.residuals(q_m, mask)
                if np.all(np.isfinite(r_ps)) and np.all(np.isfinite(r_pm)):
                    jac[:, k] = (r_ps - r_pm) / (2.0 * h_small)
                elif np.all(np.isfinite(r_ps)):
                    jac[:, k] = (r_ps - r0) / h_small
                elif np.all(np.isfinite(r_pm)):
                    jac[:, k] = (r0 - r_pm) / h_small
                else:
                    jac[:, k] = np.nan
            elif np.all(np.isfinite(r_hi)) and np.all(np.isfinite(r_lo)):
                jac[:, k] = (r_hi - r_lo) / (2.0 * h)
            else:
                jac[:, k] = np.nan
    return jac


def _covariance(model, q) -> np.ndarray:
    jac = _jacobian(model, q)
    if jac is not None and np.all(np.isfinite(jac)) and jac.shape[0] > len(q):
        try:
            return np.linalg.inv(jac.T @ jac)
        except np.linalg.LinAlgError:
            pass
    return np.full((len(q), len(q)), np.nan)


def _fit_statistics(model, q):
    det = model.detail(q)
    cov = _covariance(model, q)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    c, jac_c = channels_to_c(q[model.space.channels], jacobian=True)
    a, jac_a = resol_to_a(q[model.space.resolutions], jacobian=True)
    cov_c = jac_c @ cov[model.space.channels, model.space.channels] @ jac_c.T
    cov_a = jac_a @ cov[model.space.resolutions,
                        model.space.resolutions] @ jac_a.T
    perr_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    perr_a = np.sqrt(np.clip(np.diag(cov_a), 0, None))
    return det, cov, perr, c, cov_c, perr_c, a, cov_a, perr_a


def _reconcile_success(model, q, det, success: bool, message: str):
    if not model.is_valid(q) or not np.isfinite(det.chi2):
        return False, "degenerate fit (insufficient data coverage)"
    if success:
        return True, message
    return True, f"converged (Nelder-Mead stopped early: {message})"


def _finalize(model, q, success: bool = True, message: str = "",
              nfev: int = 0) -> FitResult:
    det, cov, perr, c, cov_c, perr_c, a, cov_a, perr_a = _fit_statistics(
        model, q)
    success, message = _reconcile_success(model, q, det, success, message)
    chi2 = det.chi2
    ndof = det.ndof

    return FitResult(
        success=success,
        message=message,
        nfev=nfev,
        params=np.asarray(q, dtype=float),
        errors=perr,
        names=model.space.names,
        chi2=float(chi2),
        ndof=int(ndof),
        reduced_chi2=float(chi2 / ndof) if ndof > 0 else np.nan,
        cov=cov,
        params_c=c, errors_c=perr_c, cov_c=cov_c,
        params_a=a, errors_a=perr_a, cov_a=cov_a,
        model=model,
        detail=det,
        scales=np.asarray(q[model.space.scales], dtype=float),
        scale_errors=np.where(np.isfinite(perr[model.space.scales]),
                              perr[model.space.scales], np.nan),
        chi2_per_dataset=np.asarray(det.chi2_per_dataset, dtype=float),
        bins_per_dataset=np.asarray(det.bins_per_dataset, dtype=int),
    )


def _fit_model(model, x0, bounds, maxiter, n_passes, verbose) -> FitResult:
    if x0 is None:
        x0 = model.x0
    if bounds is None:
        bounds = model.bounds
    m, q, nfev_total, best = _fit_passes(model, x0, bounds, maxiter, n_passes,
                                         verbose)
    return _finalize(m, q, success=bool(best.success),
                     message=str(best.message), nfev=int(nfev_total))


def _fit_passes(model, x0, bounds, maxiter, n_passes, verbose):
    x0 = np.asarray(x0, dtype=float)
    if not np.isfinite(x0).all():
        raise ValueError("fit starting point x0 contains NaN/inf")
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    if not model.is_valid(x0):
        print("[fit] warning: the starting point is degenerate (insufficient "
              "data coverage); the fit may not be meaningful", file=sys.stderr)
    n_passes = max(1, int(n_passes))

    m = None
    best = None
    nfev_total = 0
    for k in range(1, n_passes + 1):
        if k == 1:
            m_new = model.rebuilt(x0[model.space.channels])
            st = x0
        else:
            st = best.x
            m_new = model.rebuilt(st[model.space.channels])
        if not m_new.grid_ok() or not m_new.is_valid(st):
            if best is not None:
                if verbose:
                    print(f"[fit] pass {k} skipped (degenerate grid); "
                          f"reporting pass {k - 1}")
                break
            m_new = model
            st = x0
        m = m_new
        best = _fit_once(m, st, bounds, maxiter)
        nfev_total += int(best.nfev)
        tag = "grid from initial calibration" if k == 1 \
            else "grid from fitted calibration"
        if verbose:
            print(f"[fit] pass {k}: {len(m.grid_centers)} bins ({tag}), "
                  f"best chi2 = {best.fun:.2f}, nfev = {best.nfev}")
    model_final, q = m, best.x

    return model_final, q, nfev_total, best


def run_fit(model, x0=None, bounds=None, maxiter: int = 10000,
            n_passes: int = 5, verbose: bool = True) -> FitResult:
    return _fit_model(model, x0, bounds, maxiter, n_passes, verbose)


def make_x0(model, core_overrides: dict | None = None,
            scale_values: list[float] | None = None) -> list[float]:
    x0 = list(model.x0)
    space = model.space
    if core_overrides:
        for i, name in enumerate(space.names[:space.scale_start]):
            v = core_overrides.get(name)
            if v is not None:
                x0[i] = v
    if scale_values:
        for k, v in enumerate(scale_values):
            if v is not None:
                x0[space.scale_start + k] = v
    return x0
