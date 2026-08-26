"""Parameter fit with the bounded Nelder-Mead derivative-free optimizer.

The chi^2 of :class:`kc761fit.fitmodel.FitModel` is a jagged (piecewise-smooth
but not differentiable) function of the calibration parameters: the exact data
rebinning interpolates between the channel edges, and its interpolation
breakpoints move as the calibration changes, so the objective has kinks where
a grid edge's channel coordinate crosses a channel edge (verified numerically:
the slope changes by ~10^4 chi^2 units per keV of a line position at the
kinks).  The fit therefore uses the derivative-free Nelder-Mead method.  The
box bounds (channels within (0, n_bins), relative resolutions within BOUNDS_R,
scale) are passed to scipy, whose bounded Nelder-Mead clips every simplex
vertex to the box.  The monotonicity conditions (channels strictly
increasing, resolutions strictly decreasing) cannot be expressed as box
bounds; they are enforced *softly* by penalties added to chi^2 (see
``calibration.monotonicity_penalty`` / ``resolution.monotonicity_penalty``),
so the objective stays finite and smooth everywhere and the fit can converge
even when the optimum lies on the ordering boundary.  The penalty is zero for
any physically ordered point, so the minimum is unbiased; the only
hard-degeneracy condition is insufficient data coverage.

The fit runs directly in the physically meaningful parameter space
q = [x60..x2614, r60..r2614, s] (channel positions of the calibration lines,
relative resolutions, and the simulation normalization s).  The reported
``FitResult`` contains these 8 parameters together with the derived
calibration coefficients c0..c3 and resolution coefficients a0..a2 (with
errors propagated through the linear maps).

Global (multi-dataset) fits share the calibration and resolution across
several (data, simulation, range) pairs and give each dataset its own scale:
see :class:`kc761fit.globalfit.GlobalFitModel` and :func:`run_global_fit`,
which returns a :class:`FitResult` with the per-dataset scales and
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

import numpy as np
from scipy import optimize

from .calibration import channels_to_c
from .resolution import resol_to_a
from .types import FitResult

# Relative tolerance of the Nelder-Mead f-spread termination, relative to the
# objective's magnitude at the start point (a chi^2 of arbitrary scale, e.g.
# ~10^4 for the Am-241 fit and up to ~10^8 for high-statistics simulations).
# An absolute ``fatol`` is meaningless there: it would either be trivially met
# or (with a large chi^2) block the "converged" break entirely.  The value is
# small enough (1e-11 relative, with the old 1e-6 absolute floor) that it
# reproduces the previous behaviour for the current datasets (chi^2 <= ~10^5)
# while scaling up gracefully for much larger objectives.
FATOL_REL = 1e-11
FATOL_ABS = 1e-6


def _fit_once(model, x0, bounds, maxiter):
    """Bounded Nelder-Mead minimization of chi^2 from the starting point ``x0``.

    The box ``bounds`` are passed to scipy's bounded Nelder-Mead, which clips
    the initial point and every simplex vertex to the box before evaluating
    them, so the search always stays inside the bounds.  Ordering violations
    are handled softly by penalties inside ``evaluate`` (finite, smoothly
    rising), so the objective is well defined everywhere.

    ``fatol`` is scaled relative to the objective's magnitude at the start
    point (``FATOL_REL``, floored at the previous ``FATOL_ABS``), so
    convergence is judged at the same relative precision regardless of the
    absolute chi^2 scale; ``maxfev`` bounds the number of function
    evaluations per pass explicitly.
    """
    x0 = np.asarray(x0, dtype=float)
    f0 = model.evaluate(np.clip(x0, [b[0] for b in bounds],
                                [b[1] for b in bounds]))
    if not np.isfinite(f0):
        f0 = 1.0
    fatol = max(FATOL_ABS, FATOL_REL * abs(f0))
    return optimize.minimize(
        model.evaluate, x0, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=maxiter, maxfev=2 * maxiter + 200,
                     xatol=1e-3, fatol=fatol, adaptive=True))


def _jacobian(model, q0, rel_step=1e-4):
    """Central-difference Jacobian of the weighted residuals.

    The residuals are ``model.residuals(q, mask)`` on the fixed grid bins
    (concatenated over all datasets for a :class:`GlobalFitModel`).  Returns
    None if the point is degenerate (its ``detail`` has no mask) or the
    residuals are not finite, in which case parameter errors are not defined.
    """
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
            # At the upper bound: one-sided difference downward (the positive
            # perturbation would leave the box).
            q_m = np.array(q0)
            q_m[k] -= h
            r_m = model.residuals(q_m, mask)
            jac[:, k] = (r0 - r_m) / h if np.all(np.isfinite(r_m)) else np.nan
        elif q0[k] - h <= lo:
            # At the lower bound: one-sided difference upward.
            q_p = np.array(q0)
            q_p[k] += h
            r_p = model.residuals(q_p, mask)
            jac[:, k] = (r_p - r0) / h if np.all(np.isfinite(r_p)) else np.nan
        else:
            # Central difference, with a small-step / one-sided fallback when
            # a perturbation crosses an invalid region.
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


def _fit_statistics(model, q):
    """Covariance, parameter errors and derived coefficients at the point q.

    Returns ``(det, cov, perr, c, cov_c, perr_c, a, cov_a, perr_a)``: the
    model ``detail`` dict, the parameter covariance / errors (from the
    weighted-residual Jacobian on the fixed grid(s)), and the derived
    calibration / resolution coefficients with uncertainties propagated
    through their linear maps.
    """
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
    """Cross-check the optimizer's success flag against the final point.

    scipy's bounded Nelder-Mead can report ``success=False`` (maxiter
    reached, or its absolute ``fatol`` never met on a jagged objective) for a
    fit that has actually converged, and ``success=True`` for a degenerate
    point (insufficient data coverage) whose chi^2 is inf.  A point is
    reported as successful only if the model is valid there and the chi^2 is
    finite; otherwise ``success=False`` with an explicit message.
    """
    if not model.is_valid(q) or not np.isfinite(det.chi2):
        return False, "degenerate fit (insufficient data coverage)"
    if success:
        return True, message
    return True, f"converged (Nelder-Mead stopped early: {message})"


def _fill_unsmeared(det) -> None:
    """Fill the plotting-only un-convolved simulation curves on the final
    detail.  Computed once per fit (not per evaluation), so the hot detail
    path stays free of this work."""
    for ds in det.datasets:
        if ds.model is not None:
            ds.m_raw_unsmeared = ds.model.raw_model_counts()


def _finalize(model, q, success: bool = True, message: str = "",
              nfev: int = 0) -> FitResult:
    """Build the FitResult for a converged point q (fitted parameters).

    Both single- and multi-dataset fits share one result shape: the
    per-dataset ``scales`` / ``scale_errors`` / ``chi2_per_dataset`` /
    ``bins_per_dataset`` arrays (length 1 for a single-dataset fit).
    """
    det, cov, perr, c, cov_c, perr_c, a, cov_a, perr_a = _fit_statistics(
        model, q)
    _fill_unsmeared(det)
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
    """Shared single/global multi-pass fit; see ``run_fit`` / ``run_global_fit``."""
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


def run_global_fit(model, x0=None, bounds=None, maxiter: int = 10000,
                   n_passes: int = 5, verbose: bool = True) -> FitResult:
    """Minimize the summed chi^2 of a global (multi-dataset) fit.

    ``model`` is a :class:`kc761fit.globalfit.GlobalFitModel`; its parameter
    space is [x60..x2614, r60..r2614, s0..s_{N-1}] (global-fit calibration and
    resolution, one normalization scale per dataset).  ``x0`` / ``bounds``
    default to the model's own.  The multi-pass scheme is the same as
    :func:`run_fit`: every pass fits on fixed native-resolution grids, rebuilt
    from the global-fit fitted calibration between passes.  The returned
    :class:`FitResult` carries the per-dataset scales and chi^2 contributions
    alongside the global-fit parameters and the derived coefficients
    (c0..c3, a0..a2).
    """
    return _fit_model(model, x0, bounds, maxiter, n_passes, verbose)


def _fit_passes(model, x0, bounds, maxiter, n_passes, verbose):
    """Multi-pass bounded Nelder-Mead; return ``(model_final, q, nfev, best)``.

    Every pass (``n_passes == 1`` included) fits on a *fixed* energy grid at
    the native resolution (one bin per data channel); pass 1 rebuilds the grid
    from the starting channel positions (numerically identical to the input
    model's initial-calibration grid, verified), later passes from the
    previous pass's fitted calibration, so the grid follows the actual
    channel-to-energy density.  ``best`` is the scipy result of the last
    executed pass, whose ``.x`` is the reported solution.
    """
    x0 = np.asarray(x0, dtype=float)
    if not np.isfinite(x0).all():
        raise ValueError("fit starting point x0 contains NaN/inf")
    # Clip the starting point into the box: scipy would do it anyway, but a
    # clipped x0 avoids a warning and keeps the pass-1 grid estimate sane.
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
            # Pass 1: native-resolution grid from the (possibly overridden)
            # starting channels.  For the default starting point this grid is
            # identical to the input model's (verified).
            m_new = model.rebuilt(x0[model.space.channels])
            st = x0
        else:
            # Warm-start from the previous pass.  scipy's bounded
            # Nelder-Mead returns an in-bounds vertex, so the rebuilt
            # grid never follows an out-of-bounds (degenerate)
            # calibration.
            st = best.x
            m_new = model.rebuilt(st[model.space.channels])
        # Keep the previous (good) model if a rebuilt grid is degenerate:
        # a degenerate grid cannot be fit meaningfully.
        if not m_new.grid_ok() or not m_new.is_valid(st):
            if best is not None:
                if verbose:
                    print(f"[fit] pass {k} skipped (degenerate grid); "
                          f"reporting pass {k - 1}")
                break
            m_new = model  # even pass 1 degenerate: fall back to the input model
            st = x0
        m = m_new
        # The caller's ``bounds`` are used on every pass (they are always set:
        # run_fit / run_global_fit default them to the model's own).
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
    """Minimize chi^2 on the model's energy grid; return the fit result.

    ``x0`` / ``bounds`` are in the fit parameter space
    [x60..x2614, r60..r2614, s]; by default the model's own ``x0`` / ``bounds``
    are used.  The returned parameters, errors and covariance are in the same
    space, and the derived coefficients (c0..c3, a0..a2) are reported too.

    Multi-pass scheme (``n_passes``, default 5): every pass fits from a single
    starting point on a *fixed* energy grid at the native resolution (one bin
    per data channel).  Pass 1 starts from the initial values on the grid of
    the starting calibration; later passes warm-start from the previous
    pass's solution with the grid rebuilt from its fitted calibration, so the
    grid follows the actual channel-to-energy density.  The last executed
    pass is reported (``n_passes=1`` is the single-pass case, handled by the
    same loop).

    The same function fits a multi-dataset :class:`GlobalFitModel`; the
    dedicated :func:`run_global_fit` name exists for readability.
    """
    return _fit_model(model, x0, bounds, maxiter, n_passes, verbose)


def make_x0(model, core_overrides: dict | None = None,
            scale_values: list[float] | None = None) -> list[float]:
    """Fit starting point from the model's own defaults with overrides.

    ``core_overrides`` maps core parameter names (``x60..x2614``,
    ``r60..r2614``) to values; ``scale_values`` (one entry per dataset, or a
    single value in single-dataset mode) overrides the normalization scale(s).
    ``None`` entries are ignored, so the defaults (the data-driven initial
    scale for single fits, the per-dataset auto estimates for global fits)
    are kept where no override is given.
    """
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
