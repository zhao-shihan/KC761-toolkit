"""Chi-square forward model linking calibration + resolution to the data.

Forward model (all on a *fixed* uniform energy grid over [elow, ehigh])
-----------------------------------------------------------------------
1. The experimental (background-subtracted) channel spectrum is calibrated
   with E(x) = c3 x^3 + c2 x^2 + c1 x + c0 and *exactly rebinned* onto the
   fixed energy grid.  The rebin uses the cumulative distribution of the
   channel histogram (linear interpolation of the cumulative counts and
   cumulative sum-of-weights): for grid bin [g_lo, g_hi] with inverse
   calibration channels x_lo = E^-1(g_lo), x_hi = E^-1(g_hi),

        d_grid   = C(x_hi) - C(x_lo)
        var_grid = W(x_hi) - W(x_lo)

   which is the count- and error-conserving transform.
2. The intrinsic simulation spectrum is convolved with the Gaussian
   resolution sigma(E) = a2 E + a1 sqrt(E) + a0 (see
   :mod:`kc761ana.resolution`) directly onto the same fixed grid, matching
   the simulation to the data binning.
3. chi^2 = sum over grid bins with error > 0 of (d - s m)^2 / sigma^2.

Normalisation (scale)
---------------------
The simulation counts are per simulated event; the measurement has its own
live time and source strength, so a scale factor s is an explicit fit
parameter (the 8th).  Its initial value is the weighted least-squares
estimate at the default calibration.

Parameters
----------
The optimiser works with the *internal* (reparameterised) parameters
q = [b0, b1, b2, b3, g0, g1, g2, s] (see :mod:`kc761ana.reparam`); the
model converts them back to the original physics parameters
p = [c0, c1, c2, c3, a0, a1, a2, s] (order matches PARAM_NAMES), which are
the values reported by the fitter.
"""

from __future__ import annotations

import numpy as np

from .calibrate import BOUNDS_C, DEFAULT_C, poly3
from .reparam import CalibTransform, RES_E_REF, ResTransform
from .resolution import BOUNDS_A, DEFAULT_A, sigma_model, smear

PARAM_NAMES = ["c0", "c1", "c2", "c3", "a0", "a1", "a2", "s"]

# Nominal initial scale; replaced in __init__ by a data-driven estimate.
SCALE_INIT = 1.0
# Scale fit bounds (dimensionless normalisation).
BOUNDS_S = [(1e-3, 1e3)]

# Default initial values and fit bounds in the ORIGINAL (reported) parameter
# space.  The single source of truth for the calibration / resolution
# defaults and bounds is calibrate.py (DEFAULT_C, BOUNDS_C) and
# resolution.py (DEFAULT_A, BOUNDS_A); they are composed here.  FitModel
# maps the shape parameters to the internal space.
DEFAULT_INIT_ORIG = np.concatenate([DEFAULT_C, DEFAULT_A, [SCALE_INIT]])
DEFAULT_BOUNDS_ORIG = BOUNDS_C + BOUNDS_A + BOUNDS_S

# Feasibility of parameter points is enforced by the two NonlinearConstraint
# objects returned by FitModel.constraints() (for constraint-capable
# optimisers) and, for the other optimisers, by returning chi^2 = inf for
# infeasible points (see FitModel.detail).  Neither mechanism contributes to
# valid fits, so the fit is unbiased.


class FitModel:
    """Bundles data + simulation and evaluates chi^2 on a fixed energy grid.

    ``evaluate`` / ``detail`` / ``arrays`` expect the *internal* parameter
    vector q = [b0..b3, g0..g2] (well-scaled); the reported ``detail`` dict
    carries the original physics parameters (c, a).
    """

    def __init__(self, data, sim, elow: float, ehigh: float, width: float | None = None,
                 width_factor: float = 1.0):
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        self.width_factor = float(width_factor)

        # Variance floor (counts^2) applied to the data errors in the chi^2
        # weights.  Real background-subtracted errors are sqrt(S + r^2 B);
        # bins with S = B = 0 would otherwise get infinite weight and drive
        # the profiled scale to zero.  A one-count floor is the standard
        # "zero-count bin has ~1 count of uncertainty" approximation.
        self.min_variance = 1.0

        # Reparameterisation: normalised channel axis (u = x/N) and a fixed
        # reference energy (662 keV) for the resolution model.  self.x0 starts
        # as the 7 shape parameters (needed by _make_grid); the initial scale
        # is appended afterwards.
        self.calib_t = CalibTransform(data.n_bins)
        self.res_t = ResTransform(RES_E_REF)
        self.x0 = np.concatenate([
            self.calib_t.to_internal(DEFAULT_INIT_ORIG[:4]),
            self.res_t.to_internal(DEFAULT_INIT_ORIG[4:7]),
        ])
        self.bounds = self._map_bounds(DEFAULT_BOUNDS_ORIG)

        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        self.grid_edges = self._make_grid(width)
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])

        # Reasonable initial value for the normalisation: the weighted
        # least-squares estimate at the default calibration.
        self.x0 = np.append(self.x0, self._initial_scale(self.x0))

    def _initial_scale(self, q7) -> float:
        """Weighted least-squares scale estimate at parameters q7."""
        d, err, m_raw = self.arrays(q7)
        mask = err > 0
        if not np.any(mask):
            return 1.0
        d, err, m_raw = d[mask], err[mask], m_raw[mask]
        smm = float(np.sum(m_raw * m_raw / err**2))
        if smm <= 0:
            return 1.0
        return float(np.sum(d * m_raw / err**2) / smm)

    def _map_bounds(self, bounds_orig) -> list[tuple[float, float]]:
        """Map the original-space bounds into the internal parameter space."""
        lo = np.array([b[0] for b in bounds_orig], dtype=float)
        hi = np.array([b[1] for b in bounds_orig], dtype=float)
        lo_int = np.concatenate([self.calib_t.to_internal(lo[:4]),
                                 self.res_t.to_internal(lo[4:7]), [lo[7]]])
        hi_int = np.concatenate([self.calib_t.to_internal(hi[:4]),
                                 self.res_t.to_internal(hi[4:7]), [hi[7]]])
        return list(zip(lo_int, hi_int))

    # -- fixed comparison grid -------------------------------------------
    def _make_grid(self, width: float | None, c_orig=None) -> np.ndarray:
        """Uniform energy grid over [elow, ehigh].

        Default width: mean energy width of one data channel in the range,
        estimated with the calibration ``c_orig`` (default: the model's
        initial calibration), so roughly one grid bin per data channel ->
        weakly correlated bins.
        """
        if c_orig is None:
            c_orig = self.calib_t.from_internal(self.x0[:4])
        if width is None:
            e = poly3(c_orig, self._ch_edges)
            # channel index of elow / ehigh under the reference calibration
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        width *= self.width_factor
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def rebuilt(self, c_orig, width: float | None = None,
                width_factor: float | None = None):
        """New FitModel whose comparison grid follows the calibration c_orig.

        Used to re-bin the fit after a first pass, so that the final grid
        matches the actual (fitted) channel-to-energy density.  The initial
        values of the resolution parameters are kept; the initial scale is
        re-estimated for the new grid.
        """
        m = FitModel(self.data, self.sim, self.elow, self.ehigh, width=width,
                     width_factor=self.width_factor if width_factor is None
                     else width_factor)
        c_orig = np.asarray(c_orig, dtype=float)
        m.x0 = np.concatenate([m.calib_t.to_internal(c_orig), m.x0[4:]])
        m.grid_edges = m._make_grid(width, c_orig=c_orig)
        m.grid_centers = 0.5 * (m.grid_edges[:-1] + m.grid_edges[1:])
        m.x0 = np.append(m.x0[:-1], m._initial_scale(m.x0[:-1]))
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
        # for the optimiser; the monotonicity penalty keeps the solution
        # monotonic).
        if np.all(np.diff(e_edges) > 0):
            e_sorted, x_sorted = e_edges, self._ch_edges
        else:
            order = np.argsort(e_edges, kind="stable")
            e_sorted, x_sorted = e_edges[order], self._ch_edges[order]
        x_edges = np.interp(self.grid_edges, e_sorted, x_sorted)
        c_int = np.interp(x_edges, self._ch_edges, self._cum_counts)
        w_int = np.interp(x_edges, self._ch_edges, self._cum_sumw2)
        d = np.diff(c_int)
        var = np.diff(w_int)
        var = np.clip(var, 0.0, None)
        return d, np.sqrt(var)

    # -- validity / constraints --------------------------------------------
    @property
    def min_usable_bins(self) -> int:
        """Minimum number of data-covered grid bins for a meaningful fit.

        Below this the calibration has essentially no overlap with the
        measured channels (a degenerate "fit" of a handful of empty bins),
        which is treated as invalid.
        """
        return max(10, int(0.1 * len(self.grid_centers)))

    def is_monotonic(self, c) -> bool:
        """True if the calibration E(x) is strictly increasing over all
        channel edges (a correct calibration is always monotonic)."""
        return bool(np.all(np.diff(poly3(c, self._ch_edges)) > 0))

    def monotonicity_slack(self, q) -> float:
        """Constraint slack (>= 0 for a monotonic calibration): the minimum
        slope of E(x) over the channel edges, in keV per channel."""
        c = self.calib_t.from_internal(q[:4])
        return float(np.min(np.diff(poly3(c, self._ch_edges))))

    def coverage_slack(self, q) -> int:
        """Constraint slack (>= 0 for sufficient coverage): the number of
        grid bins with data, minus the minimum usable count."""
        _d, err, _m = self.arrays(q)
        return int(np.sum(err > 0)) - self.min_usable_bins

    def constraints(self):
        """The two feasibility constraints of the fit as scipy
        ``NonlinearConstraint`` objects (for constraint-capable optimisers):

        * monotonicity: ``min(E') > 0`` (smooth, differentiable);
        * coverage:     usable grid bins >= ``min_usable_bins`` (piecewise
          constant, non-differentiable -- use a derivative-free method such
          as COBYLA).
        """
        from scipy.optimize import NonlinearConstraint
        return [
            NonlinearConstraint(self.monotonicity_slack, 0.0, np.inf,
                                keep_feasible=True),
            NonlinearConstraint(self.coverage_slack, 0.0, np.inf),
        ]

    def is_valid(self, q) -> bool:
        """Feasibility of a parameter point: monotonic calibration,
        non-negative resolution, and non-degenerate data coverage."""
        c = self.calib_t.from_internal(q[:4])
        a = self.res_t.from_internal(q[4:7])
        if not self.is_monotonic(c):
            return False
        d, err, _m = self.arrays(q)
        if int(np.sum(err > 0)) < self.min_usable_bins:
            return False
        mu = self.grid_centers[err > 0]
        if np.any(sigma_model(a, mu) <= 0):
            return False
        return True

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
    def arrays(self, q):
        """Full-grid (d, err, m_raw) arrays.

        ``q`` is the internal parameter vector; it is converted to the
        original physics parameters before evaluating the models.
        """
        c = self.calib_t.from_internal(q[:4])
        a = self.res_t.from_internal(q[4:7])
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(a)
        return d, err, m_raw

    @staticmethod
    def _degenerate_detail(p) -> dict:
        """Placeholder detail for an infeasible parameter point.  The
        returned chi^2 is ``np.inf`` so that any optimiser (including
        derivative-free ones) rejects the point; a valid fit is never
        affected, so this cannot bias the result."""
        c = np.asarray(p[:4], dtype=float)
        a = np.asarray(p[4:7], dtype=float)
        s = float(p[7])
        return dict(
            d=np.array([]), err=np.array([]), m_raw=np.array([]),
            m=np.array([]), s=s, mu=np.array([]),
            chi2=np.inf, ndof=0, pen=0.0,
            c=c, a=a, mask=None, grid_centers=None,
        )

    def detail(self, q):
        """Masked evaluation at internal parameters q (always a dict).

        ``q`` = [b0..b3, g0..g2, s]; the returned dict carries the original
        physics parameters (c, a) and the scale s.  Infeasible points
        (non-monotonic calibration, no coverage, non-positive resolution)
        return a degenerate dict with chi^2 = inf.
        """
        c = self.calib_t.from_internal(q[:4])
        a = self.res_t.from_internal(q[4:7])
        s = float(q[7])
        p = np.concatenate([c, a, [s]])
        if not self.is_valid(q):
            return self._degenerate_detail(p)
        d, err, m_raw = self.arrays(q)
        mask = err > 0
        d, err, m_raw = d[mask], err[mask], m_raw[mask]
        # Variance floor in the weights (see __init__).
        err = np.sqrt(np.maximum(err**2, self.min_variance))
        mu = self.grid_centers[mask]
        m = s * m_raw
        chi2 = float(np.sum((d - m) ** 2 / err**2))
        ndof = len(d) - len(PARAM_NAMES)
        return dict(
            d=d, err=err, m_raw=m_raw, m=m, s=s,
            mu=mu, chi2=chi2, ndof=ndof, pen=0.0,
            c=c, a=a, mask=mask,
            grid_centers=self.grid_centers,
        )

    def evaluate(self, q) -> float:
        det = self.detail(q)
        return det["chi2"]
