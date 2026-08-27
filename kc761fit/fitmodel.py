"""Per-dataset chi-square forward model.

The fit model used everywhere is :class:`kc761fit.globalfit.GlobalFitModel`,
which bundles N of these units and shares the energy calibration and the
detector resolution across all of them (N = 1 is a single-dataset fit).  Each
:class:`FitModel` owns one dataset: the comparison grid over ``[elow, ehigh]``,
the exact rebin of its channel data onto the grid, and the convolution of its
simulation onto the grid.  It only ever sees the *core* parameter vector

    q_core = [x60, x609, x1461, x2614, r60, r609, r2614]

(the 4 calibration-line channel positions and the 3 relative resolutions); the
normalization scale(s) and the full parameter-vector layout live in the owning
``GlobalFitModel.space``.

Model details
-------------
1. The experimental (background-subtracted) channel spectrum is calibrated
   with the cubic E(x) fixed by the channel positions of the 60/609/1461/2614
   keV lines and *exactly rebinned* onto the fixed uniform energy grid (the
   count- and error-conserving cumulative rebin, see ``rebin_data``).
2. The intrinsic simulation is convolved with the Gaussian resolution, whose
   *relative* widths sigma(E)/E at 60/609/2614 keV are the fit parameters,
   directly onto the same grid (see :mod:`kc761fit.resolution`).
3. chi^2 = sum over grid bins with positive statistical error of
   (d - s m)^2 / err^2, with err the statistical error plus a fractional
   systematic proportional to the bin counts (``sys_frac``) plus the
   x-direction (finite bin-width) error projected through the model slope.
   The x-direction term is evaluated at a *fixed* reference scale (the
   data-driven WLS estimate ``s_ref``), so the chi^2 weights are independent
   of the fitted parameters and the reported J^T J covariance is meaningful
   (see ``error_model``).
"""

from __future__ import annotations

import numpy as np

from .calibration import INIT_X, channels_to_c, poly3
from .params import BOUNDS_S, DEFAULT_SYS_FRAC, ParameterSpace
from .resolution import INIT_R, resol_to_a, smear
from .types import DatasetDetail

# Variance floor (counts^2) applied to the data errors in the chi^2 weights.
# Real background-subtracted errors are sqrt(S + r^2 B); bins with S = B = 0
# would otherwise get infinite weight and drive the profiled scale to zero.
# A one-count floor is the standard "a zero-count bin has ~1 count of
# uncertainty" approximation.  (Bins whose statistical error is exactly 0 are
# *excluded* by the ``err > 0`` mask before this floor is applied.)
MIN_VARIANCE = 1.0


class FitModel:
    """One (data, simulation, energy-range) pair of a (possibly global) fit.

    Builds the fixed comparison grid over ``[elow, ehigh]`` and evaluates the
    forward model arrays (``arrays``) and the masked per-dataset chi^2
    (``dataset_detail``) for the shared core parameters ``q_core``.  ``space``
    is the core-only :class:`~kc761fit.params.ParameterSpace` (channels +
    resolutions); the owning :class:`GlobalFitModel` extends it with the
    per-dataset scales.
    """

    def __init__(self, data, sim, elow: float, ehigh: float,
                 width: float | None = None, sys_frac: float = DEFAULT_SYS_FRAC,
                 *, _channels=None):
        """Bundles one dataset and sets up the comparison grid.

        ``width`` is an optional fixed grid bin width (default: about one data
        channel width, estimated with the starting calibration).  The internal
        ``_channels`` argument is used by ``rebuilt`` to construct a model
        whose grid follows the fitted calibration.
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
        # errors proportional to the bin counts.
        self.sys_frac = float(sys_frac)
        self.min_variance = MIN_VARIANCE

        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")

        # Cumulative channel histogram and sum-of-weights, used for the exact
        # rebin of the calibrated data onto the energy grid.
        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        # Core parameter space (channels within (0, n_bins), resolutions in
        # BOUNDS_R); the scale block belongs to the owning GlobalFitModel.
        self.space = ParameterSpace.from_anchors(data.n_bins)

        # Grid follows the starting channel positions (INIT_X, or the fitted
        # channels when rebuilt between fit passes).
        grid_channels = (np.asarray(_channels, dtype=float)
                         if _channels is not None else INIT_X)
        self.grid_edges = self._make_grid(width, c_orig=channels_to_c(
            grid_channels))
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])
        # Uniform grid: energy width of one grid bin (keV).  Half of it is the
        # x-direction (energy) uncertainty of each bin, projected onto y
        # through the model slope in ``error_model``.
        self.bin_width = float(np.diff(self.grid_edges)[0])

        # Starting core parameters (channels + resolutions).  The x-direction
        # error term in ``error_model`` is evaluated at a *fixed* reference
        # scale (``s_ref``), so the chi^2 weights stay independent of the
        # fitted parameters; the reference is the data-driven WLS estimate.
        self.x0_core = np.concatenate([grid_channels, INIT_R])
        self.s_ref = 0.0  # x-term off while the reference scale is estimated
        self.initial_scale = self._initial_scale(self.x0_core)
        self.s_ref = float(self.initial_scale)

    def error_model(self, d, err_stat, m_prime) -> np.ndarray:
        """Total per-bin errors with the x-direction (energy) term included.

        As in ROOT's TGraphErrors treatment, the error along x (here half the
        energy-grid bin width, ``0.5 * bin_width``) is projected onto the y
        direction through the slope of the model f = s_ref * m_raw:

            err = sqrt(err_stat^2 + (sys_frac * d)^2
                       + (0.5 * bin_width * s_ref * m_prime)^2)

        with ``m_prime = d(m_raw)/dE`` the central-difference derivative of the
        resolution-smeared model over the grid centers.  The x-term is
        evaluated at the *fixed* reference scale ``s_ref`` (not the fitted
        scale), so the chi^2 weights are parameter-independent: the objective
        is a proper weighted least squares and the reported J^T J covariance
        is meaningful.
        """
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * self.s_ref * m_prime) ** 2)
        return np.sqrt(np.maximum(var, self.min_variance))

    def _initial_scale(self, q_core) -> float:
        """Data-driven initial normalization scale at the core parameters
        ``q_core`` (closed-form weighted least squares).

        The weights are the same as in the chi^2 objective (statistical +
        fractional-systematic; the x-direction term is zero at ``s_ref = 0``),
        so the estimate matches the profiled scale the fit would converge to
        at fixed (channels, resolutions) with parameter-independent weights.
        """
        d, err, m_raw = self.arrays(q_core)
        mask = err > 0
        if not np.any(mask):
            return 1.0
        # Model slope over the full grid (as in ``dataset_detail``), then mask.
        m_prime = np.gradient(m_raw, self.grid_centers)
        d, err, m_raw, m_prime = (d[mask], err[mask], m_raw[mask],
                                  m_prime[mask])
        var = self.error_model(d, err, m_prime) ** 2
        smm = float(np.sum(m_raw * m_raw / var))
        if smm <= 0:
            return 1.0
        s0 = float(np.sum(d * m_raw / var) / smm)
        lo, hi = BOUNDS_S[0]
        return float(np.clip(s0, lo, hi))

    # -- fixed comparison grid -------------------------------------------
    def _make_grid(self, width: float | None, c_orig) -> np.ndarray:
        """Uniform energy grid over [elow, ehigh].

        Default width: mean energy width of one data channel in the range,
        estimated with the calibration ``c_orig``, so roughly one grid bin per
        data channel -> weakly correlated bins.
        """
        if width is None:
            e = poly3(c_orig, self._ch_edges)
            # Channel index of elow / ehigh under the reference calibration.
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def rebuilt(self, channels):
        """New FitModel whose comparison grid follows the calibration fixed by
        the fitted channel positions ``channels``.

        Used to re-bin the fit after a first pass, so the grid always matches
        the actual (fitted) channel-to-energy density at the native resolution.
        The fitter warm-starts every pass from the previous solution, so this
        model's own ``initial_scale`` is only a seed.
        """
        return FitModel(self.data, self.sim, self.elow, self.ehigh,
                        sys_frac=self.sys_frac, _channels=channels)

    def grid_ok(self) -> bool:
        """True if this model's comparison grid is large enough to fit."""
        return len(self.grid_centers) >= self.min_usable_bins

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

    def is_valid(self, q_core) -> bool:
        """True unless the point is degenerate (insufficient data coverage).

        Monotonicity of the calibration channels / resolutions is not a hard
        validity condition: ordering violations are handled by soft penalties
        in the objective, keeping the search finite everywhere.
        """
        d, err, m_raw = self.arrays(q_core)
        return int(np.sum(err > 0)) >= self.min_usable_bins

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
    def arrays(self, q_core, c=None, a=None):
        """Full-grid (d, err, m_raw) arrays at the core parameters q_core.

        The calibration / resolution coefficients are derived from ``q_core``
        unless already computed and passed in as ``c`` / ``a`` (used by the hot
        evaluation path to avoid re-solving them).
        """
        if c is None:
            c = channels_to_c(q_core[self.space.channels])
        if a is None:
            a = resol_to_a(q_core[self.space.resolutions])
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(a)
        return d, err, m_raw

    def dataset_detail(self, d, err, m_raw, s, mask=None,
                       label: str = "") -> DatasetDetail:
        """Mask, weight and assemble a DatasetDetail from full-grid arrays.

        Applies the bin mask (``err > 0`` when ``mask`` is None), computes the
        model slope over the *full* grid first (the x-direction error term
        needs the unmasked gradient), applies ``error_model`` and sums the
        masked chi^2.
        """
        if mask is None:
            mask = err > 0
        m_prime = np.gradient(m_raw, self.grid_centers)
        d, err, m_raw, m_prime = d[mask], err[mask], m_raw[mask], m_prime[mask]
        err = self.error_model(d, err, m_prime)
        mu = self.grid_centers[mask]
        m = float(s) * m_raw
        chi2 = float(np.sum((d - m) ** 2 / err**2))
        return DatasetDetail(
            d=d, err=err, m_raw=m_raw, m=m, s=float(s), mu=mu,
            chi2=chi2, n_bins=int(len(d)), mask=mask,
            grid_edges=self.grid_edges,
            label=label, elow=self.elow, ehigh=self.ehigh,
        )
