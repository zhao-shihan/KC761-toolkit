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
   to the bin counts (``sys_frac``, default 0) plus the x-direction
   (finite energy-bin-width) error projected onto y through the model slope
   (``FitModel.model_error``).

Parameters
----------
The optimizer works directly with the physically meaningful parameters

    q = [x60, x609, x1461, x2614, r60, r609, r2614, s]

where x are the channel positions of the calibration lines, r = sigma/E the
relative resolutions at the resolution lines, and s the normalization scale.
Bounds are simple box constraints (channels in (0, n_channels), r in (0, 1),
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

from .calibration import (
    INIT_X, channels_to_c, monotonicity_penalty as calib_monotonicity_penalty,
    poly3,
)
from .resolution import (
    BOUNDS_R, INIT_R, monotonicity_penalty as resol_monotonicity_penalty,
    resol_to_a, smear,
)

# Order of the fitted (reported) parameters.
PARAM_NAMES = ["x60", "x609", "x1461", "x2614", "r60", "r609", "r2614", "s"]
# Derived coefficients reported on output (from channels / resolutions).
PARAM_NAMES_C = ["c0", "c1", "c2", "c3"]
PARAM_NAMES_A = ["a0", "a1", "a2"]

# Nominal initial scale; replaced in __init__ by a data-driven estimate.
SCALE_INIT = 1.0
# Scale fit bounds (dimensionless normalization).
BOUNDS_S = [(1e-3, 1e3)]

# Initial values: channel positions of the calibration lines, relative
# resolutions, scale.  The per-channel upper bound (0, n_bins) depends on
# the data and is built in FitModel.__init__.
INIT_PARAMS = np.concatenate([INIT_X, INIT_R, [SCALE_INIT]])


class FitModel:
    """Bundles data + simulation and evaluates chi^2 on a fixed energy grid.

    ``evaluate`` / ``detail`` / ``arrays`` expect the fit parameter vector
    q = [x60..x2614, r60..r2614, s] (channels, relative resolutions, scale);
    the reported ``detail`` dict also carries the derived coefficients
    (c, a) and the fitted channels / resolutions (x, r).
    """

    def __init__(self, data, sim, elow: float, ehigh: float, width: float | None = None,
                 sys_frac: float = 0.0):
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        # Per-bin fractional systematic error (dimensionless, e.g. 0.05 = 5%,
        # 0 by default), added in quadrature to the statistical errors
        # proportional to the bin counts:
        # err = sqrt(err_stat^2 + (sys_frac * d)^2).
        self.sys_frac = float(sys_frac)

        # Variance floor (counts^2) applied to the data errors in the chi^2
        # weights.  Real background-subtracted errors are sqrt(S + r^2 B);
        # bins with S = B = 0 would otherwise get infinite weight and drive
        # the profiled scale to zero.  A one-count floor is the standard
        # "a zero-count bin has ~1 count of uncertainty" approximation.
        self.min_variance = 1.0

        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")

        # Cumulative channel histogram and sum-of-weights, used for the exact
        # rebin of the calibrated data onto the energy grid.
        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        # Fit bounds: channels in (0, n_bins), relative resolutions in
        # (0, 1), scale in BOUNDS_S.
        n = float(data.n_bins)
        self.bounds = [(0.0, n)] * 4 + BOUNDS_R + BOUNDS_S

        self.x0 = np.array(INIT_PARAMS, dtype=float)
        self.grid_edges = self._make_grid(width)
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])
        # Uniform grid: energy width of one grid bin (keV).  Half of it is the
        # x-direction (energy) uncertainty of each bin, projected onto y
        # through the model slope in ``model_error``.
        self.bin_width = float(np.diff(self.grid_edges)[0])

        # Reasonable initial scale: the weighted least-squares estimate at
        # the initial calibration / resolution.
        self.x0 = np.append(self.x0[:-1], self._initial_scale(self.x0[:-1]))

    def total_errors(self, d, err_stat) -> np.ndarray:
        """Total per-bin errors used in the chi^2 weights.

        Statistical error plus a fractional systematic proportional to the
        bin counts (``sys_frac``), added in quadrature, with the
        ``min_variance`` floor:

            err = sqrt(err_stat^2 + (sys_frac * d)^2),  floored at 1 count.

        The x-direction (finite energy-bin-width) term is *not* included here;
        use :meth:`model_error` for the full chi^2 weighting.
        """
        var = err_stat**2 + (self.sys_frac * d) ** 2
        return np.sqrt(np.maximum(var, self.min_variance))

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
        to the weights that accounts for the finite energy width of each bin.
        """
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * s * m_prime) ** 2)
        return np.sqrt(np.maximum(var, self.min_variance))

    def _initial_scale(self, q7) -> float:
        """Weighted least-squares scale estimate at the 7 parameters q7
        (channels and resolutions, without the scale)."""
        d, err, m_raw = self.arrays(q7)
        mask = err > 0
        if not np.any(mask):
            return 1.0
        d, err, m_raw = d[mask], err[mask], m_raw[mask]
        err = self.total_errors(d, err)
        smm = float(np.sum(m_raw * m_raw / err**2))
        if smm <= 0:
            return 1.0
        return float(np.sum(d * m_raw / err**2) / smm)

    # -- fixed comparison grid -------------------------------------------
    def _make_grid(self, width: float | None, c_orig=None) -> np.ndarray:
        """Uniform energy grid over [elow, ehigh].

        Default width: mean energy width of one data channel in the range,
        estimated with the calibration ``c_orig`` (default: the model's
        initial calibration), so roughly one grid bin per data channel ->
        weakly correlated bins.
        """
        if c_orig is None:
            c_orig = channels_to_c(self.x0[:4])
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
        resolution (one grid bin per data channel).  The initial values of
        the resolution parameters are kept; the initial scale is re-estimated
        for the new grid.
        """
        m = FitModel(self.data, self.sim, self.elow, self.ehigh, width=width,
                     sys_frac=self.sys_frac)
        channels = np.asarray(channels, dtype=float)
        # Keep the resolutions, drop the scale: it is re-estimated below for
        # the new grid.
        m.x0 = np.concatenate([channels, m.x0[4:7]])
        m.grid_edges = m._make_grid(width, c_orig=channels_to_c(channels))
        m.grid_centers = 0.5 * (m.grid_edges[:-1] + m.grid_edges[1:])
        m.x0 = np.append(m.x0, m._initial_scale(m.x0))
        return m

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
        c = channels_to_c(q[:4])
        a = resol_to_a(q[4:7])
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
            c = channels_to_c(q[:4])
        if a is None:
            a = resol_to_a(q[4:7])
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(a)
        return d, err, m_raw

    @staticmethod
    def _degenerate_detail(p) -> dict:
        """Placeholder detail for an infeasible parameter point.

        The returned chi^2 is ``np.inf`` so that any optimizer (including
        derivative-free ones) rejects the point; a valid fit is never
        affected, so this cannot bias the result.
        """
        c = channels_to_c(p[:4])
        a = resol_to_a(p[4:7])
        s = float(p[7])
        return dict(
            d=np.array([]), err=np.array([]), m_raw=np.array([]),
            m=np.array([]), s=s, mu=np.array([]),
            chi2=np.inf, ndof=0, pen=0.0,
            c=c, a=a, x=p[:4], r=p[4:7], mask=None, grid_centers=None,
        )

    def detail(self, q):
        """Masked evaluation at fit parameters q (always a dict).

        ``q`` = [x60..x2614, r60..r2614, s]; the returned dict carries the
        fitted channels (x) / resolutions (r), the derived coefficients
        (c, a), the scale s, the *data* chi^2 (``chi2``) and the
        monotonicity penalty (``pen``).  Ordering violations of the channels
        / resolutions do not make the point infeasible: they just add a
        quadratically rising penalty (zero for any physically ordered point).
        Only a degenerate point (insufficient data coverage) returns
        chi^2 = inf.
        """
        ok, d, err, m_raw, c, a = self._check(q)
        s = float(q[7])
        x = np.asarray(q[:4], dtype=float)
        r = np.asarray(q[4:7], dtype=float)
        p = np.concatenate([x, r, [s]])
        if not ok:
            return self._degenerate_detail(p)
        mask = err > 0
        # Model slope over the full grid (for the x-direction error term),
        # then mask everything consistently.
        m_prime = np.gradient(m_raw, self.grid_centers)
        d, err, m_raw, m_prime = (
            d[mask], err[mask], m_raw[mask], m_prime[mask])
        # Statistical + fractional-systematic errors + the x-direction
        # (finite bin-width) term in the chi^2 weights (see model_error).
        err = self.model_error(d, err, s, m_prime)
        mu = self.grid_centers[mask]
        m = s * m_raw
        chi2 = float(np.sum((d - m) ** 2 / err**2))
        pen = calib_monotonicity_penalty(x) + resol_monotonicity_penalty(r)
        ndof = len(d) - len(PARAM_NAMES)
        return dict(
            d=d, err=err, m_raw=m_raw, m=m, s=s,
            mu=mu, chi2=chi2, ndof=ndof, pen=pen,
            c=c, a=a, x=x, r=r, mask=mask,
            grid_centers=self.grid_centers,
        )

    def evaluate(self, q) -> float:
        det = self.detail(q)
        return det["chi2"] + det["pen"]

    def residuals(self, q, mask=None) -> np.ndarray:
        """Weighted residuals (d - s*m)/sigma over the fixed grid bins.

        ``mask`` selects the grid bins (e.g. ``detail()["mask"]``); with
        ``mask=None`` all bins are used.  Statistical + fractional-systematic
        + x-direction errors enter via ``model_error``, consistent with
        ``detail``.  This is the vector whose central-difference Jacobian
        gives the parameter uncertainties
        (see :func:`kc761fit.fitter._jacobian`).
        """
        d, err, m_raw = self.arrays(q)
        m_prime = np.gradient(m_raw, self.grid_centers)
        if mask is not None:
            d, err, m_raw, m_prime = (d[mask], err[mask], m_raw[mask],
                                      m_prime[mask])
        s = float(q[7])
        err = self.model_error(d, err, s, m_prime)
        return (d - s * m_raw) / err
