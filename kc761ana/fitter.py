"""Parameter fit with the Nelder-Mead derivative-free optimiser.

The chi^2 of :class:`kc761ana.fitmodel.FitModel` is a piecewise-smooth
(jagged) function of the calibration parameters (the exact data rebinning
has kinks where channel boundaries cross the energy grid), so the fit uses
the derivative-free Nelder-Mead method (bounds enforced by clamping; the
monotonicity and coverage conditions are enforced by returning chi^2 = inf
for infeasible points).  Neither mechanism contributes to a valid fit, so
the result is unbiased.

The fit runs in the *internal* (reparameterised) parameter space
q = [b0..b3, g0..g2, s] (s is the simulation normalisation, an explicit fit
parameter); the reported ``FitResult`` contains the original parameters
p = [c0, c1, c2, c3, a0, a1, a2, s] together with their errors and
covariance, mapped back through the linear reparameterisation.

Between passes the energy grid is rebuilt from the fitted calibration and
narrowed (3x coarse -> native), so the final grid matches the actual
channel-to-energy density and the final chi^2/ndof is meaningful.

Parameter uncertainties are estimated from the weighted-residual Jacobian
at the best fit, evaluated on the fixed grid:

    r(q) = (d(q) - s m(q)) / sigma(q),   cov_int = (J^T J)^-1,

mapped to the original parameters with cov_orig[i, j] = cov_int[i, j] *
T[i] * T[j], where T is the diagonal scale vector of the reparameterisation
(T = 1 for the scale s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .fitmodel import PARAM_NAMES


@dataclass
class FitResult:
    success: bool
    message: str
    nfev: int
    params: np.ndarray
    errors: np.ndarray
    names: list[str]
    chi2: float
    ndof: int
    reduced_chi2: float
    scale: float
    scale_err: float
    cov: np.ndarray
    params_internal: np.ndarray = field(default_factory=lambda: np.array([]))
    model: object = None
    detail: dict = field(default_factory=dict)


def _clamp(q, bounds):
    return np.array([min(max(q[i], bounds[i][0]), bounds[i][1]) for i in range(len(q))])


def _fit_once(model, x0, bounds, maxiter):
    # """Nelder-Mead minimisation of chi^2 from the starting point ``x0``.

    # Nelder-Mead has no native bounds: the parameters are clamped to the
    # bounds in the objective, and infeasible points are rejected by
    # ``evaluate`` returning inf.
    # """
    return minimize(lambda q: model.evaluate(_clamp(q, bounds)), x0,
                    method="Nelder-Mead", 
                    options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-8))


def _residual(q, model, fixed_mask):
    """Weighted residuals (d - s*m)/sigma on the fixed grid bins."""
    d, err, m_raw = model.arrays(q)
    d = d[fixed_mask]
    err = err[fixed_mask]
    m_raw = m_raw[fixed_mask]
    # Variance floor consistent with FitModel.detail.
    err = np.sqrt(np.maximum(err**2, model.min_variance))
    s = float(q[7])
    return (d - s * m_raw) / err


def _jacobian(model, q0, rel_step=1e-4):
    """Central-difference Jacobian of the residuals on the fixed grid.

    Returns None if the point is infeasible (degenerate detail) or the
    residuals are not finite, in which case parameter errors are not
    defined.
    """
    det0 = model.detail(q0)
    if det0 is None or det0["mask"] is None:
        return None
    mask = det0["mask"]
    r0 = _residual(q0, model, mask)
    if np.any(~np.isfinite(r0)):
        return None
    n_q = len(q0)
    jac = np.empty((len(r0), n_q))
    for k in range(n_q):
        h = rel_step * max(1.0, abs(q0[k]))
        q_hi = np.array(q0)
        q_lo = np.array(q0)
        q_hi[k] += h
        q_lo[k] -= h
        r_hi = _residual(q_hi, model, mask)
        r_lo = _residual(q_lo, model, mask)
        if (np.any(~np.isfinite(r_hi)) or np.any(~np.isfinite(r_lo))) \
                and h > 1e-12:
            # Central difference crossed an invalid region: retry with a
            # smaller step, then fall back to a one-sided difference.
            h_small = 1e-6 * max(1.0, abs(q0[k]))
            q_s = np.array(q0); q_s[k] += h_small
            q_m = np.array(q0); q_m[k] -= h_small
            r_ps = _residual(q_s, model, mask)
            r_pm = _residual(q_m, model, mask)
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


def _finalize(model, q):
    """Build the FitResult (original parameters) for a converged point q."""
    det = model.detail(q)
    chi2 = det["chi2"] if det is not None else np.nan
    ndof = det["ndof"] if det is not None else 0

    # Parameter uncertainties from the residual Jacobian on the fixed grid.
    jac = _jacobian(model, q)
    if jac is not None and np.all(np.isfinite(jac)) and jac.shape[0] > len(q):
        try:
            cov_int = np.linalg.inv(jac.T @ jac)
        except np.linalg.LinAlgError:
            cov_int = np.full((len(q), len(q)), np.nan)
    else:
        cov_int = np.full((len(q), len(q)), np.nan)

    # Map internal parameters and covariance back to the original space
    # (the scale s is unchanged by the reparameterisation: T = 1).
    t_scale = np.concatenate([model.calib_t.scale, model.res_t.scale, [1.0]])
    p_orig = np.concatenate([model.calib_t.from_internal(q[:4]),
                             model.res_t.from_internal(q[4:7]), [q[7]]])
    cov = cov_int * np.outer(t_scale, t_scale)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    return FitResult(
        success=True,
        message="",
        nfev=0,
        params=p_orig,
        errors=perr,
        names=PARAM_NAMES,
        chi2=float(chi2),
        ndof=int(ndof),
        reduced_chi2=float(chi2 / ndof) if ndof > 0 else np.nan,
        scale=float(q[7]),
        scale_err=float(perr[7]) if np.isfinite(perr[7]) else np.nan,
        cov=cov,
        params_internal=q,
        model=model,
        detail=det,
    )


def run_fit(model, x0=None, bounds=None, maxiter: int = 600,
            n_passes: int = 3, verbose: bool = True) -> FitResult:
    """Minimise chi^2 on the model's energy grid; return the fit result.

    ``x0`` / ``bounds`` are in the *internal* parameter space; by default
    the model's own ``x0`` / ``bounds`` are used.  The returned parameters,
    errors and covariance are in the original (reported) space.

    Multi-pass scheme (``n_passes``, default 3): each pass fits from a single
    starting point on a *fixed* energy grid — pass 1 starts from the initial
    values, later passes warm-start from the previous pass's solution.
    Between passes the grid is rebuilt from the fitted calibration, with the
    grid width factor decreasing linearly from 3x (coarse, smoother chi^2,
    faster) to 1x (native: one bin per data channel, so the final
    chi^2/ndof is statistically meaningful).  The last executed pass is
    reported.
    """
    if x0 is None:
        x0 = model.x0
    if bounds is None:
        bounds = model.bounds
    x0 = np.asarray(x0, dtype=float)
    n_passes = max(1, int(n_passes))

    if n_passes == 1:
        # Single pass on the original model's grid.
        best = _fit_once(model, x0, bounds, maxiter)
        model_final, q = model, best.x
        nfev_total = int(best.nfev)
        if verbose:
            print(f"[fit] pass 1: {len(model.grid_centers)} bins (native), "
                  f"best chi2 = {best.fun:.2f}, nfev = {best.nfev}")
    else:
        m = None
        best = None
        nfev_total = 0
        for k in range(1, n_passes + 1):
            # Grid width factor: 3.0 for pass 1, down to 1.0 for the last pass.
            wf = max(1.0, 3.0 * (n_passes - k) / (n_passes - 1))
            if k == 1:
                m_new = model.rebuilt(model.calib_t.from_internal(x0[:4]),
                                      width_factor=wf)
                st = x0
            else:
                c = m.calib_t.from_internal(best.x[:4])
                m_new = model.rebuilt(c, width_factor=wf)
                st = _clamp(best.x, m_new.bounds)  # warm start from previous pass
            # Keep the previous (good) model if the rebuilt grid is
            # degenerate: a degenerate grid cannot be fit meaningfully.
            if len(m_new.grid_centers) < 20:
                if best is not None:
                    if verbose:
                        print(f"[fit] pass {k} skipped (degenerate grid); "
                              f"reporting pass {k - 1}")
                    break
                m_new = model  # even pass 1 degenerate: fall back to the input model
                st = x0
            m = m_new
            best = _fit_once(m, st, m.bounds, maxiter)
            nfev_total += int(best.nfev)
            tag = "3x-coarsened, grid from initial calibration" if k == 1 \
                else "grid from fitted calibration"
            if verbose:
                print(f"[fit] pass {k}: {len(m.grid_centers)} bins ({tag}), "
                      f"best chi2 = {best.fun:.2f}, nfev = {best.nfev}")
        model_final, q = m, best.x

    # Nelder-Mead evaluates the *clamped* parameters; clamp the reported
    # solution to the bounds so the reported parameters and chi^2 are
    # consistent with what was actually optimised.
    q = _clamp(q, model_final.bounds)
    result = _finalize(model_final, q)
    result.nfev = int(nfev_total)
    result.success = bool(best.success)
    result.message = str(best.message)
    return result
