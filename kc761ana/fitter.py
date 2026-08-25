"""Parameter fit with the bounded Nelder-Mead derivative-free optimiser.

The chi^2 of :class:`kc761ana.fitmodel.FitModel` is a piecewise-smooth
(jagged) function of the calibration parameters (the exact data rebinning
has kinks where channel boundaries cross the energy grid), so the fit uses
the derivative-free Nelder-Mead method.  The box bounds (channels within
(0, n_bins), relative resolutions within (0, 1), scale) are passed to scipy,
whose bounded Nelder-Mead clips every simplex vertex to the box.  The
monotonicity conditions (channels strictly increasing, resolutions strictly
decreasing) cannot be expressed as box bounds; they are enforced *softly* by
penalties added to chi^2 (see ``calibrate.monotonicity_penalty`` /
``resolution.monotonicity_penalty``), so the objective stays finite and
smooth everywhere and the fit can converge even when the optimum lies on the
ordering boundary.  The penalty is zero for any physically ordered point, so
the minimum is unbiased; the only hard-degeneracy condition is insufficient
data coverage.

The fit runs directly in the physically meaningful parameter space
q = [x60..x2614, r60..r2614, s] (channel positions of the calibration lines,
relative resolutions, and the simulation normalisation s).  The reported
``FitResult`` contains these 8 parameters together with the derived
calibration coefficients c0..c3 and resolution coefficients a0..a2 (with
errors propagated through the linear maps).

Between passes the energy grid is rebuilt from the fitted calibration and
narrowed (3x coarse -> native), so the final grid matches the actual
channel-to-energy density and the final chi^2/ndof is meaningful.

Parameter uncertainties are estimated from the weighted-residual Jacobian
at the best fit, evaluated on the fixed grid:

    r(q) = (d(q) - s m(q)) / sigma(q),   cov = (J^T J)^-1,

and the reported coefficient errors are propagated as
cov_c = J_c cov[0:4,0:4] J_c^T, cov_a = J_a cov[4:7,4:7] J_a^T.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .calibrate import channels_to_c
from .fitmodel import PARAM_NAMES
from .resolution import res_to_a


@dataclass
class FitResult:
    success: bool
    message: str
    nfev: int
    params: np.ndarray           # [x60..x2614, r60..r2614, s]
    errors: np.ndarray
    names: list[str]
    chi2: float
    ndof: int
    reduced_chi2: float
    scale: float
    scale_err: float
    cov: np.ndarray
    params_c: np.ndarray = field(
        default_factory=lambda: np.array([]))   # c0..c3
    errors_c: np.ndarray = field(default_factory=lambda: np.array([]))
    cov_c: np.ndarray = field(default_factory=lambda: np.array([]))
    params_a: np.ndarray = field(
        default_factory=lambda: np.array([]))   # a0..a2
    errors_a: np.ndarray = field(default_factory=lambda: np.array([]))
    cov_a: np.ndarray = field(default_factory=lambda: np.array([]))
    model: object = None
    detail: dict = field(default_factory=dict)


def _fit_once(model, x0, bounds, maxiter):
    """Bounded Nelder-Mead minimisation of chi^2 from the starting point ``x0``.

    The box ``bounds`` are passed to scipy's bounded Nelder-Mead, which clips
    the initial point and every simplex vertex to the box before evaluating
    them, so the search always stays inside the bounds.  Ordering violations
    are handled softly by penalties inside ``evaluate`` (finite, smoothly
    rising), so the objective is well defined everywhere.
    """
    return optimize.minimize(
        model.evaluate, x0, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-6, adaptive=True))


def _residual(q, model, fixed_mask):
    """Weighted residuals (d - s*m)/sigma on the fixed grid bins."""
    d, err, m_raw = model.arrays(q)
    d = d[fixed_mask]
    err = err[fixed_mask]
    m_raw = m_raw[fixed_mask]
    # Statistical + fractional-systematic errors, consistent with
    # FitModel.detail.
    err = model.total_errors(d, err)
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
            q_s = np.array(q0)
            q_s[k] += h_small
            q_m = np.array(q0)
            q_m[k] -= h_small
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
    """Build the FitResult for a converged point q (fitted parameters)."""
    det = model.detail(q)
    chi2 = det["chi2"] if det is not None else np.nan
    ndof = det["ndof"] if det is not None else 0

    # Parameter uncertainties from the residual Jacobian on the fixed grid.
    jac = _jacobian(model, q)
    if jac is not None and np.all(np.isfinite(jac)) and jac.shape[0] > len(q):
        try:
            cov = np.linalg.inv(jac.T @ jac)
        except np.linalg.LinAlgError:
            cov = np.full((len(q), len(q)), np.nan)
    else:
        cov = np.full((len(q), len(q)), np.nan)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    # Derived coefficients with propagated uncertainties (linear maps).
    c, jac_c = channels_to_c(q[:4], jacobian=True)
    a, jac_a = res_to_a(q[4:7], jacobian=True)
    cov_c = jac_c @ cov[:4, :4] @ jac_c.T
    cov_a = jac_a @ cov[4:7, 4:7] @ jac_a.T
    perr_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    perr_a = np.sqrt(np.clip(np.diag(cov_a), 0, None))

    return FitResult(
        success=True,
        message="",
        nfev=0,
        params=np.asarray(q, dtype=float),
        errors=perr,
        names=PARAM_NAMES,
        chi2=float(chi2),
        ndof=int(ndof),
        reduced_chi2=float(chi2 / ndof) if ndof > 0 else np.nan,
        scale=float(q[7]),
        scale_err=float(perr[7]) if np.isfinite(perr[7]) else np.nan,
        cov=cov,
        params_c=c, errors_c=perr_c, cov_c=cov_c,
        params_a=a, errors_a=perr_a, cov_a=cov_a,
        model=model,
        detail=det,
    )


def run_fit(model, x0=None, bounds=None, maxiter: int = None,
            n_passes: int = 5, verbose: bool = True) -> FitResult:
    """Minimise chi^2 on the model's energy grid; return the fit result.

    ``x0`` / ``bounds`` are in the fit parameter space
    [x60..x2614, r60..r2614, s]; by default the model's own ``x0`` / ``bounds``
    are used.  The returned parameters, errors and covariance are in the same
    space, and the derived coefficients (c0..c3, a0..a2) are reported too.

    Multi-pass scheme (``n_passes``, default 5): each pass fits from a single
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
    # Clip the starting point into the box: scipy would do it anyway, but a
    # clipped x0 avoids a warning and keeps the pass-1 grid estimate sane.
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    if not model.is_valid(x0):
        print("[fit] warning: the starting point is degenerate (insufficient "
              "data coverage); the fit may not be meaningful", file=sys.stderr)
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
                m_new = model.rebuilt(x0[:4], width_factor=wf)
                st = x0
            else:
                # Warm-start from the previous pass.  scipy's bounded
                # Nelder-Mead returns an in-bounds vertex, so the rebuilt
                # grid never follows an out-of-bounds (degenerate)
                # calibration.
                st = best.x
                m_new = model.rebuilt(st[:4], width_factor=wf)
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

    # scipy's bounded Nelder-Mead keeps every simplex vertex inside the box,
    # so the reported solution is in-bounds by construction.
    result = _finalize(model_final, q)
    result.nfev = int(nfev_total)
    result.success = bool(best.success)
    result.message = str(best.message)
    return result
