"""Parameter fit with the bounded Nelder-Mead derivative-free optimizer.

The chi^2 of :class:`kc761fit.fitmodel.FitModel` is a piecewise-smooth
(jagged) function of the calibration parameters (the exact data rebinning
has kinks where channel boundaries cross the energy grid), so the fit uses
the derivative-free Nelder-Mead method.  The box bounds (channels within
(0, n_bins), relative resolutions within (0, 1), scale) are passed to scipy,
whose bounded Nelder-Mead clips every simplex vertex to the box.  The
monotonicity conditions (channels strictly increasing, resolutions strictly
decreasing) cannot be expressed as box bounds; they are enforced *softly* by
penalties added to chi^2 (see ``calibration.monotonicity_penalty`` /
``resolution.monotonicity_penalty``), so the objective stays finite and
smooth everywhere and the fit can converge even when the optimum lies on the
ordering boundary.  The penalty is zero for any physically ordered point, so
the minimum is unbiased; the only hard-degeneracy condition is insufficient
data coverage.

The fit runs directly in the physically meaningful parameter space
q = [x60..x2614, r60..r2614, s] (channel positions of the calibration lines,
relative resolutions, and the simulation normalization s).  The reported
``FitResult`` contains these 8 parameters together with the derived
calibration coefficients c0..c3 and resolution coefficients a0..a2 (with
errors propagated through the linear maps).

Global (multi-dataset) fits share the calibration and resolution across
several (data, simulation, range) pairs and give each dataset its own scale:
see :class:`kc761fit.globalfit.GlobalFitModel` and :func:`run_global_fit`,
which returns a :class:`GlobalFitResult` with the per-dataset scales and
chi^2 contributions.

Multi-pass scheme: each pass fits on a *fixed* energy grid at the native
resolution (one bin per data channel) and the grid is rebuilt from the
previous pass's fitted calibration, so it always matches the actual
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

from .calibration import channels_to_c
from .fitmodel import PARAM_NAMES
from .resolution import resol_to_a


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


@dataclass
class GlobalFitResult:
    """Result of a global (multi-dataset) fit.

    Parameter vector ``params`` = [x60..x2614, r60..r2614, s0..s_{N-1}]: the
    energy calibration and resolution are *global-fit* parameters (7, common
    to all datasets), each dataset gets its own normalization scale.
    ``scales`` / ``scale_errors`` are the per-dataset normalizations;
    ``chi2_per_dataset`` and ``bins_per_dataset`` give the per-dataset
    contribution to the total chi^2 and the number of masked grid bins.
    """

    success: bool
    message: str
    nfev: int
    params: np.ndarray           # [x60..x2614, r60..r2614, s0..s_{N-1}]
    errors: np.ndarray
    names: list[str]
    chi2: float
    ndof: int
    reduced_chi2: float
    scales: np.ndarray           # per-dataset normalization scales
    scale_errors: np.ndarray
    chi2_per_dataset: np.ndarray
    bins_per_dataset: np.ndarray
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
    """Bounded Nelder-Mead minimization of chi^2 from the starting point ``x0``.

    The box ``bounds`` are passed to scipy's bounded Nelder-Mead, which clips
    the initial point and every simplex vertex to the box before evaluating
    them, so the search always stays inside the bounds.  Ordering violations
    are handled softly by penalties inside ``evaluate`` (finite, smoothly
    rising), so the objective is well defined everywhere.
    """
    return optimize.minimize(
        model.evaluate, x0, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-6, adaptive=True))


def _jacobian(model, q0, rel_step=1e-4):
    """Central-difference Jacobian of the weighted residuals.

    The residuals are ``model.residuals(q, mask)`` on the fixed grid bins
    (concatenated over all datasets for a :class:`GlobalFitModel`).  Returns
    None if the point is degenerate (its ``detail`` has no mask) or the
    residuals are not finite, in which case parameter errors are not defined.
    """
    det0 = model.detail(q0)
    if det0["mask"] is None:
        return None
    mask = det0["mask"]
    r0 = model.residuals(q0, mask)
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
        r_hi = model.residuals(q_hi, mask)
        r_lo = model.residuals(q_lo, mask)
        if (np.any(~np.isfinite(r_hi)) or np.any(~np.isfinite(r_lo))) \
                and h > 1e-12:
            # Central difference crossed an invalid region: retry with a
            # smaller step, then fall back to a one-sided difference.
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
    """Parameter covariance (len(q) x len(q)) from the residual Jacobian on
    the fixed grid(s); NaN-filled if the Jacobian is unavailable (degenerate
    point), non-finite, under-determined, or singular."""
    jac = _jacobian(model, q)
    if jac is not None and np.all(np.isfinite(jac)) and jac.shape[0] > len(q):
        try:
            return np.linalg.inv(jac.T @ jac)
        except np.linalg.LinAlgError:
            pass
    return np.full((len(q), len(q)), np.nan)


def _finalize(model, q, success: bool = True, message: str = "", nfev: int = 0):
    """Build the FitResult for a converged point q (fitted parameters)."""
    det = model.detail(q)
    chi2 = det["chi2"]
    ndof = det["ndof"]

    # Parameter uncertainties from the residual Jacobian on the fixed grid.
    cov = _covariance(model, q)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    # Derived coefficients with propagated uncertainties (linear maps).
    c, jac_c = channels_to_c(q[:4], jacobian=True)
    a, jac_a = resol_to_a(q[4:7], jacobian=True)
    cov_c = jac_c @ cov[:4, :4] @ jac_c.T
    cov_a = jac_a @ cov[4:7, 4:7] @ jac_a.T
    perr_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    perr_a = np.sqrt(np.clip(np.diag(cov_a), 0, None))

    return FitResult(
        success=success,
        message=message,
        nfev=nfev,
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


def _finalize_global(model, q, success: bool = True, message: str = "",
                     nfev: int = 0) -> GlobalFitResult:
    """Build the GlobalFitResult for a converged global point q."""
    det = model.detail(q)
    chi2 = det["chi2"]
    ndof = det["ndof"]

    # Parameter uncertainties from the residual Jacobian on the fixed grids.
    cov = _covariance(model, q)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))

    # Derived coefficients with propagated uncertainties (linear maps).
    c, jac_c = channels_to_c(q[:4], jacobian=True)
    a, jac_a = resol_to_a(q[4:7], jacobian=True)
    cov_c = jac_c @ cov[:4, :4] @ jac_c.T
    cov_a = jac_a @ cov[4:7, 4:7] @ jac_a.T
    perr_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    perr_a = np.sqrt(np.clip(np.diag(cov_a), 0, None))

    n = model.n_datasets
    scales = np.asarray(q[7:7 + n], dtype=float)
    scale_errors = np.where(np.isfinite(perr[7:7 + n]), perr[7:7 + n], np.nan)

    return GlobalFitResult(
        success=success,
        message=message,
        nfev=nfev,
        params=np.asarray(q, dtype=float),
        errors=perr,
        names=model.param_names,
        chi2=float(chi2),
        ndof=int(ndof),
        reduced_chi2=float(chi2 / ndof) if ndof > 0 else np.nan,
        scales=scales,
        scale_errors=scale_errors,
        chi2_per_dataset=np.asarray(det["chi2_per_dataset"], dtype=float),
        bins_per_dataset=np.asarray(det["bins_per_dataset"], dtype=int),
        cov=cov,
        params_c=c, errors_c=perr_c, cov_c=cov_c,
        params_a=a, errors_a=perr_a, cov_a=cov_a,
        model=model,
        detail=det,
    )


def run_global_fit(model, x0=None, bounds=None, maxiter: int = None,
                   n_passes: int = 5, verbose: bool = True) -> GlobalFitResult:
    """Minimize the summed chi^2 of a global (multi-dataset) fit.

    ``model`` is a :class:`kc761fit.globalfit.GlobalFitModel`; its parameter
    space is [x60..x2614, r60..r2614, s0..s_{N-1}] (global-fit calibration and
    resolution, one normalization scale per dataset).  ``x0`` / ``bounds``
    default to the model's own.  The multi-pass scheme is the same as
    :func:`run_fit`: every pass fits on fixed native-resolution grids, rebuilt
    from the global-fit fitted calibration between passes.  The returned
    :class:`GlobalFitResult` carries the per-dataset scales and chi^2
    contributions alongside the global-fit parameters and the derived
    coefficients (c0..c3, a0..a2).
    """
    if x0 is None:
        x0 = model.x0
    if bounds is None:
        bounds = model.bounds
    m, q, nfev_total, best = _fit_passes(model, x0, bounds, maxiter, n_passes,
                                         verbose)
    # scipy's bounded Nelder-Mead keeps every simplex vertex inside the box,
    # so the reported solution is in-bounds by construction.
    return _finalize_global(m, q, success=bool(best.success),
                            message=str(best.message), nfev=int(nfev_total))


def _fit_passes(model, x0, bounds, maxiter, n_passes, verbose):
    """Multi-pass bounded Nelder-Mead; return ``(model_final, q, nfev, best)``.

    Each pass fits on a *fixed* energy grid at the native resolution (one bin
    per data channel); between passes the grid is rebuilt from the fitted
    calibration so it follows the actual channel-to-energy density (for a
    :class:`GlobalFitModel` every dataset's grid is rebuilt from the global-fit
    fitted calibration).  ``best`` is the scipy result of the last executed
    pass, whose ``.x`` is the reported solution.
    """
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
            # Native-resolution grid every pass; it is rebuilt from the
            # fitted calibration so it follows the channel-to-energy density.
            if k == 1:
                m_new = model.rebuilt(x0[:4])
                st = x0
            else:
                # Warm-start from the previous pass.  scipy's bounded
                # Nelder-Mead returns an in-bounds vertex, so the rebuilt
                # grid never follows an out-of-bounds (degenerate)
                # calibration.
                st = best.x
                m_new = model.rebuilt(st[:4])
            # Keep the previous (good) model if the rebuilt grid is
            # degenerate: a degenerate grid cannot be fit meaningfully.
            if len(m_new.grid_centers) < 20 or not m_new.is_valid(st):
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
            tag = "grid from initial calibration" if k == 1 \
                else "grid from fitted calibration"
            if verbose:
                print(f"[fit] pass {k}: {len(m.grid_centers)} bins ({tag}), "
                      f"best chi2 = {best.fun:.2f}, nfev = {best.nfev}")
        model_final, q = m, best.x

    return model_final, q, nfev_total, best


def run_fit(model, x0=None, bounds=None, maxiter: int = None,
            n_passes: int = 5, verbose: bool = True) -> FitResult:
    """Minimize chi^2 on the model's energy grid; return the fit result.

    ``x0`` / ``bounds`` are in the fit parameter space
    [x60..x2614, r60..r2614, s]; by default the model's own ``x0`` / ``bounds``
    are used.  The returned parameters, errors and covariance are in the same
    space, and the derived coefficients (c0..c3, a0..a2) are reported too.

    Multi-pass scheme (``n_passes``, default 5): each pass fits from a single
    starting point on a *fixed* energy grid at the native resolution (one bin
    per data channel) — pass 1 starts from the initial values, later passes
    warm-start from the previous pass's solution.  Between passes the grid is
    rebuilt from the fitted calibration, so it follows the actual
    channel-to-energy density.  The last executed pass is reported.

    For a global (multi-dataset) fit use :func:`run_global_fit`, which shares
    the same multi-pass scheme and returns a :class:`GlobalFitResult`.
    """
    if x0 is None:
        x0 = model.x0
    if bounds is None:
        bounds = model.bounds
    m, q, nfev_total, best = _fit_passes(model, x0, bounds, maxiter, n_passes,
                                         verbose)
    # scipy's bounded Nelder-Mead keeps every simplex vertex inside the box,
    # so the reported solution is in-bounds by construction.
    return _finalize(m, q, success=bool(best.success),
                     message=str(best.message), nfev=int(nfev_total))
