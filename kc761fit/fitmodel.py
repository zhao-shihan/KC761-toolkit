"""Chi-square forward model linking calibration + resolution to the data.

The model evaluates chi^2 on a *fixed* uniform energy grid over [elow, ehigh]:

1. The experimental (background-subtracted) channel spectrum is calibrated
   with the cubic E(x) fixed by the channel positions of the 60/609/1461/2614
   keV lines (see :mod:`kc761fit.calibration`) and *exactly rebinned* onto the
   grid.  The rebin uses the cumulative distribution of the channel histogram
   (linear interpolation of the cumulative counts and cumulative
   sum-of-weights): for grid bin [g_lo, g_hi] with inverse-calibration
   channels x_lo = E^-1(g_lo), x_hi = E^-1(g_hi),

       d_grid   = C(x_hi) - C(x_lo)
       var_grid = W(x_hi) - W(x_lo)

   which is the count- and error-conserving transform.
2. The intrinsic simulation spectrum is convolved with the Gaussian
   resolution, whose *relative* widths sigma(E)/E at 60/609/2614 keV are the
   fit parameters (see :mod:`kc761fit.resolution`), directly onto the same
   grid, matching the simulation to the data binning.
3. chi^2 = sum over grid bins with error > 0 of (d - s m)^2 / sigma^2, where
   sigma is the statistical error plus a fractional systematic proportional
   to the bin counts (``sys_frac``, default ``DEFAULT_SYS_FRAC`` = 5%) plus
   the x-direction (finite energy-bin-width) error projected onto y through
   the model slope (``FitModel.model_error``).

Parameters
----------
The optimizer works directly with the physically meaningful parameters

    q = [x60, x609, x1461, x2614, r60, r609, r2614, s]

where x are the channel positions of the calibration lines, r = sigma/E the
relative resolutions at the resolution lines, and s the normalization scale.
Bounds are simple box constraints (channels in (0, n_channels), r in BOUNDS_R,
s in BOUNDS_S).  The ordering conditions (x60 < x609 < x1461 < x2614 and
r60 > r609 > r2614) are enforced *softly*: violations add a quadratically
rising penalty to chi^2 (see ``calibration.monotonicity_penalty`` /
``resolution.monotonicity_penalty``) instead of returning inf, so the
objective stays finite everywhere and the derivative-free optimizer can
converge even when the optimum sits on the ordering boundary.  The penalty is
zero for any physically ordered point, so it does not bias the minimum; the
only hard-degeneracy condition is insufficient data coverage.

The polynomial coefficients c0..c3 and a0..a2 are derived from q
(``channels_to_c`` / ``resol_to_a``) only where the forward model needs them
and reported alongside the fitted channels/resolutions on output.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .calibration import (
    INIT_X, channels_to_c,
    monotonicity_penalty as calib_monotonicity_penalty, poly3,
)
from .params import (
    DEFAULT_SYS_FRAC, ParameterSpace,
)
from .resolution import (
    INIT_R,
    monotonicity_penalty as resol_monotonicity_penalty, resol_to_a, smear,
)
from .types import DatasetDetail, FitDetail

# The order of the fitted (reported) parameters, the derived coefficients and
# the parameter-space layout are owned by kc761fit.params (ParameterSpace).


def finish_dataset(model, d, err, m_raw, s, mask=None) -> DatasetDetail:
    """Mask, weight and assemble a DatasetDetail from full-grid arrays.

    Shared by :class:`FitModel` and :class:`GlobalFitModel`: applies the bin
    mask (``err > 0`` when ``mask`` is None), computes the model slope over
    the *full* grid first (the x-direction error term needs the unmasked
    gradient), applies ``model_error`` and sums the masked chi^2.  The
    operation order matches the previous per-model implementations exactly.
    """
    if mask is None:
        mask = err > 0
    m_prime = np.gradient(m_raw, model.grid_centers)
    d, err, m_raw, m_prime = (d[mask], err[mask], m_raw[mask], m_prime[mask])
    err = model.model_error(d, err, float(s), m_prime)
    mu = model.grid_centers[mask]
    m = float(s) * m_raw
    chi2 = float(np.sum((d - m) ** 2 / err**2))
    return DatasetDetail(
        d=d, err=err, m_raw=m_raw, m=m, s=float(s), mu=mu,
        chi2=chi2, n_bins=int(len(d)), mask=mask,
        grid_edges=model.grid_edges,
        model=model,
    )


def degenerate_detail(space, p, c, a) -> FitDetail:
    """Placeholder FitDetail for an infeasible parameter point.

    chi^2 = inf rejects the point for any optimizer; a valid fit is never
    affected, so this cannot bias the result.  ``p`` is the full parameter
    vector in the ``space`` layout; ``s`` is the per-dataset scales array.
    """
    return FitDetail(
        datasets=[], s=np.asarray(p[space.scales], dtype=float),
        chi2=np.inf, ndof=0, pen=0.0,
        c=c, a=a, x=p[space.channels], r=p[space.resolutions],
        valid=False,
    )


class FitModel:
    """Bundles data + simulation and evaluates chi^2 on a fixed energy grid.

    ``evaluate`` / ``detail`` / ``arrays`` expect the fit parameter vector
    q = [x60..x2614, r60..r2614, s] (channels, relative resolutions, scale);
    the reported ``detail`` (:class:`~kc761fit.types.FitDetail`) also carries
    the derived coefficients (c, a) and the fitted channels / resolutions
    (x, r).  ``self.space`` is the :class:`~kc761fit.params.ParameterSpace`
    that owns the vector layout (``space.channels`` / ``space.resolutions`` /
    ``space.scales`` slices and ``space.scale_start``).
    """

    def __init__(self, data, sim, elow: float, ehigh: float,
                 width: float | None = None, sys_frac: float = DEFAULT_SYS_FRAC,
                 *, _grid_channels=None, _resolutions=None, _scale_init=None):
        """Bundles data + simulation and evaluates chi^2 on a fixed energy grid.

        The public arguments set up the comparison grid over ``[elow, ehigh]``
        (``width`` = optional fixed bin width; default: about one data channel
        width).  The keyword-only ``_grid_channels`` / ``_resolutions`` /
        ``_scale_init`` arguments are internal: ``rebuilt`` uses them to
        construct a grid-following model from fitted parameters without
        re-running the data-driven initial-scale estimate.
        """
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        if not (self.elow < self.ehigh):
            raise ValueError(
                f"elow ({self.elow}) must be < ehigh ({self.ehigh})")
        if width is not None and width <= 0:
            raise ValueError(f"grid width must be > 0, got {width}")
        # Per-bin fractional systematic error (dimensionless, e.g. 0.05 = 5%,
        # DEFAULT_SYS_FRAC by default), added in quadrature to the statistical
        # errors proportional to the bin counts:
        # err = sqrt(err_stat^2 + (sys_frac * d)^2).
        self.sys_frac = float(sys_frac)

        # Variance floor (counts^2) applied to the data errors in the chi^2
        # weights.  Real background-subtracted errors are sqrt(S + r^2 B);
        # bins with S = B = 0 would otherwise get infinite weight and drive
        # the profiled scale to zero.  A one-count floor is the standard
        # "a zero-count bin has ~1 count of uncertainty" approximation.
        # (Bins whose statistical error is exactly 0 are *excluded* by the
        # ``err > 0`` mask before this floor is applied, so they impose no
        # constraint.)
        self.min_variance = 1.0

        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")

        # Cumulative channel histogram and sum-of-weights, used for the exact
        # rebin of the calibrated data onto the energy grid.
        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        # Parameter space: channels within (0, n_bins), relative resolutions
        # in BOUNDS_R, scale in BOUNDS_S.  The scale's initial value (1.0) is
        # replaced below by a data-driven weighted least-squares estimate.
        self.space = ParameterSpace.from_anchors(data.n_bins).with_scales(
            1, names=["s"])
        self.bounds = self.space.bounds
        self.n_datasets = 1

        grid_channels = (np.asarray(_grid_channels, dtype=float)
                         if _grid_channels is not None else INIT_X)
        resolutions = (np.asarray(_resolutions, dtype=float)
                       if _resolutions is not None else INIT_R)
        self.grid_edges = self._make_grid(
            width, c_orig=channels_to_c(grid_channels))
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])
        # Uniform grid: energy width of one grid bin (keV).  Half of it is the
        # x-direction (energy) uncertainty of each bin, projected onto y
        # through the model slope in ``model_error``.
        self.bin_width = float(np.diff(self.grid_edges)[0])

        # Initial parameters: the grid-reference channels (the fitted ones for
        # a rebuilt model), the resolutions, and the scale (auto-estimated at
        # first construction; carried forward by ``rebuilt``).
        self.x0 = np.concatenate([grid_channels, resolutions])
        if _scale_init is None:
            # Reasonable initial scale: the weighted least-squares estimate at
            # the initial calibration / resolution, weighted consistently with
            # the chi^2 objective (see ``_initial_scale``).
            scale = self._initial_scale(self.x0)
        else:
            scale = float(_scale_init)
        self.x0 = np.append(self.x0, scale)

    def model_error(self, d, err_stat, s, m_prime) -> np.ndarray:
        """Total per-bin errors with the x-direction (energy) term included.

        As in ROOT's TGraphErrors treatment, the error along x (here half the
        energy-grid bin width, ``0.5 * bin_width``) is projected onto the y
        direction through the slope of the model f = s * m_raw:

            err = sqrt(err_stat^2 + (sys_frac * d)^2
                       + (0.5 * bin_width * s * m_prime)^2)

        with ``m_prime = d(m_raw)/dE`` the central-difference derivative of
        the resolution-smeared model over the grid centers (the derivative of
        the full model is ``s * m_prime``).  This adds a covariance-like term
        to the weights that accounts for the finite energy width of each bin
        (the bins of a real histogram are not independent along x).  Because
        it contains the fitted scale ``s`` and the model slope, the weights
        are model-dependent: this is intentional, but it means the parameter
        covariance reported from ``J^T J`` (which assumes fixed weights) is
        an approximation.
        """
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * s * m_prime) ** 2)
        return np.sqrt(np.maximum(var, self.min_variance))

    def _initial_scale(self, q_core) -> float:
        """Weighted least-squares scale estimate at the 7 parameters ``q_core``
        (channels and resolutions, without the scale).

        The weights are the *same* as in the chi^2 objective (statistical +
        fractional-systematic + x-direction terms, see ``model_error``), so
        the estimate matches the profiled scale the fit would converge to at
        fixed (channels, resolutions).  Because the x-direction term contains
        the scale itself, the scale is found by a bounded 1-D minimization of
        the profiled chi^2 rather than a closed-form ratio.
        """
        d, err, m_raw = self.arrays(q_core)
        mask = err > 0
        if not np.any(mask):
            return 1.0
        # Model slope over the full grid (as in ``detail``), then mask.
        m_prime = np.gradient(m_raw, self.grid_centers)
        d, err, m_raw, m_prime = (d[mask], err[mask], m_raw[mask],
                                  m_prime[mask])
        # Statistical + fractional-systematic variance (floored at
        # min_variance), the s = 0 limit of ``model_error``.
        var = self.model_error(d, err, 0.0, 0.0) ** 2
        smm = float(np.sum(m_raw * m_raw / var))
        if smm <= 0:
            return 1.0
        # Closed-form WLS ignoring the x-term: a cheap bracketing start.
        s0 = float(np.sum(d * m_raw / var) / smm)

        def chi2_at(s):
            # Same per-bin weights as ``model_error`` (statistical + systematic
            # + x-direction terms together, floored).
            v = self.model_error(d, err, s, m_prime) ** 2
            return float(np.sum((d - s * m_raw) ** 2 / v))

        lo, hi = self.bounds[self.space.scale_start]
        res = minimize_scalar(chi2_at, bounds=(lo, hi), method="bounded")
        if res.success and np.isfinite(res.fun):
            return float(res.x)
        return float(np.clip(s0, lo, hi))

    # -- fixed comparison grid -------------------------------------------
    def _make_grid(self, width: float | None, c_orig=None) -> np.ndarray:
        """Uniform energy grid over [elow, ehigh].

        Default width: mean energy width of one data channel in the range,
        estimated with the calibration ``c_orig`` (default: the model's
        initial calibration), so roughly one grid bin per data channel ->
        weakly correlated bins.
        """
        if c_orig is None:
            c_orig = channels_to_c(self.x0[self.space.channels])
        if width is None:
            e = poly3(c_orig, self._ch_edges)
            # Channel index of elow / ehigh under the reference calibration.
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def rebuilt(self, channels, width: float | None = None):
        """New FitModel whose comparison grid follows the calibration fixed
        by the fitted channel positions ``channels``.

        Used to re-bin the fit after a first pass, so that the grid always
        matches the actual (fitted) channel-to-energy density at the native
        resolution (one grid bin per data channel).  The resolutions and the
        initial scale are carried forward from this model (the fit warm-starts
        from the previous pass's solution anyway).
        """
        return FitModel(
            self.data, self.sim, self.elow, self.ehigh, width=width,
            sys_frac=self.sys_frac,
            _grid_channels=channels,
            _resolutions=self.x0[self.space.resolutions],
            _scale_init=self.x0[self.space.scale_start],
        )

    def grid_ok(self) -> bool:
        """True if this model's comparison grid is large enough to fit."""
        return len(self.grid_centers) >= self.min_usable_bins

    @property
    def channel_max(self) -> float:
        """Last channel edge (plotting range of the calibration curve)."""
        return float(self.data.edges[-1])

    @property
    def n_channel_bins(self) -> int:
        """MCA channel count (upper bound of the calibration channels)."""
        return int(self.data.n_bins)

    # -- data rebinning ---------------------------------------------------
    def rebin_data(self, c) -> tuple[np.ndarray, np.ndarray]:
        """Calibrate the channel data and rebin it exactly onto the grid.

        Returns (d_grid, err_grid) over all grid bins.  Bins with no channel
        coverage get zero counts and zero error.
        """
        e_edges = poly3(c, self._ch_edges)
        # Inverse calibration: channel coordinate of each grid edge.  For a
        # (nearly) non-monotonic calibration the pairs are sorted so that the
        # interpolation stays finite (used only as a smooth fallback region
        # for the optimizer).
        if np.all(np.diff(e_edges) > 0):
            e_sorted, x_sorted = e_edges, self._ch_edges
        else:
            order = np.argsort(e_edges, kind="stable")
            e_sorted, x_sorted = e_edges[order], self._ch_edges[order]
        x_edges = np.interp(self.grid_edges, e_sorted, x_sorted)
        c_int = np.interp(x_edges, self._ch_edges, self._cum_counts)
        w_int = np.interp(x_edges, self._ch_edges, self._cum_sumw2)
        d = np.diff(c_int)
        var = np.clip(np.diff(w_int), 0.0, None)
        return d, np.sqrt(var)

    # -- validity -----------------------------------------------------------
    @property
    def min_usable_bins(self) -> int:
        """Minimum number of data-covered grid bins for a meaningful fit.

        Below this the calibration has essentially no overlap with the
        measured channels (a degenerate "fit" of a handful of empty bins),
        which is treated as invalid.
        """
        return max(10, int(0.1 * len(self.grid_centers)))

    def _check(self, q):
        """Single-pass degeneracy check + model arrays.

        Returns ``(ok, d, err, m_raw, c, a)``.  The only hard-degeneracy
        condition is data coverage (too few usable grid bins); ordering of
        the channels / resolutions is enforced *softly* by penalties added
        to chi^2 (see ``detail``), so the objective stays finite everywhere
        for the derivative-free optimizer.
        """
        c = channels_to_c(q[self.space.channels])
        a = resol_to_a(q[self.space.resolutions])
        d, err, m_raw = self.arrays(q, c, a)
        if int(np.sum(err > 0)) < self.min_usable_bins:
            return False, None, None, None, c, a
        return True, d, err, m_raw, c, a

    def is_valid(self, q) -> bool:
        """True unless the point is degenerate (insufficient data coverage).

        Monotonicity of the calibration channels / resolutions is not a hard
        validity condition: ordering violations are handled by soft penalties
        in the objective, keeping the search finite everywhere.
        """
        return self._check(q)[0]

    # -- model ------------------------------------------------------------
    def model_counts(self, a) -> np.ndarray:
        """Simulation convolved with resolution onto the fixed grid."""
        return smear(self.sim.counts, self.sim.edges, self.grid_edges, a)

    def raw_model_counts(self) -> np.ndarray:
        """Intrinsic (un-smeared) simulation rebinned onto the fixed grid.

        Count-conserving histogram rebin of the simulation onto the grid
        edges; used for plotting the "no resolution" comparison curve.
        """
        sim_cum = np.concatenate(([0.0], np.cumsum(self.sim.counts)))
        c_lo = np.interp(self.grid_edges[:-1], self.sim.edges, sim_cum)
        c_hi = np.interp(self.grid_edges[1:], self.sim.edges, sim_cum)
        return c_hi - c_lo

    # -- evaluation -------------------------------------------------------
    def arrays(self, q, c=None, a=None):
        """Full-grid (d, err, m_raw) arrays at parameters q.

        ``q`` is the fit parameter vector.  The calibration / resolution
        coefficients are derived from it unless already computed and passed
        in as ``c`` / ``a`` (used by the hot evaluation path to avoid
        re-solving them).
        """
        if c is None:
            c = channels_to_c(q[self.space.channels])
        if a is None:
            a = resol_to_a(q[self.space.resolutions])
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(a)
        return d, err, m_raw

    def detail(self, q) -> FitDetail:
        """Masked evaluation at fit parameters q (always a FitDetail).

        ``q`` = [x60..x2614, r60..r2614, s]; the returned detail carries the
        fitted channels (x) / resolutions (r), the derived coefficients
        (c, a), the per-dataset scale (``s``), the single dataset entry
        (``datasets``), the *data* chi^2 (``chi2``) and the monotonicity
        penalty (``pen``).  Ordering violations of the channels / resolutions
        do not make the point infeasible: they just add a quadratically rising
        penalty (zero for any physically ordered point).  Only a degenerate
        point (insufficient data coverage) returns chi^2 = inf (with
        ``valid=False``).
        """
        ok, d, err, m_raw, c, a = self._check(q)
        s = float(q[self.space.scale_start])
        x = np.asarray(q[self.space.channels], dtype=float)
        r = np.asarray(q[self.space.resolutions], dtype=float)
        p = np.concatenate([x, r, [s]])
        if not ok:
            return degenerate_detail(self.space, p, c, a)
        ds = finish_dataset(self, d, err, m_raw, s)
        ds.elow = self.elow
        ds.ehigh = self.ehigh
        pen = calib_monotonicity_penalty(x) + resol_monotonicity_penalty(r)
        ndof = ds.n_bins - self.space.size
        return FitDetail(
            datasets=[ds], s=np.array([ds.s]),
            chi2=ds.chi2, chi2_per_dataset=np.array([ds.chi2]),
            bins_per_dataset=np.array([ds.n_bins], dtype=int),
            ndof=ndof, pen=pen,
            c=c, a=a, x=x, r=r,
        )

    def evaluate(self, q) -> float:
        det = self.detail(q)
        return det.chi2 + det.pen

    def residuals(self, q, mask=None) -> np.ndarray:
        """Weighted residuals (d - s*m)/sigma over the fixed grid bins.

        ``mask`` selects the grid bins (e.g. ``detail().mask``); with
        ``mask=None`` all bins are used.  Statistical + fractional-systematic
        + x-direction errors enter via ``model_error``, consistent with
        ``detail``.  This is the vector whose central-difference Jacobian
        gives the parameter uncertainties
        (see :func:`kc761fit.fitter._jacobian`).
        """
        d, err, m_raw = self.arrays(q)
        s = float(q[self.space.scale_start])
        ds = finish_dataset(self, d, err, m_raw, s, mask=mask)
        return (ds.d - ds.s * ds.m_raw) / ds.err
