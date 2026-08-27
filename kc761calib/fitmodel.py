"""Per-dataset forward model: calibrated rebinning, smearing, weighting."""

from __future__ import annotations

import numpy as np

from .params import BOUNDS_S, DEFAULT_SYS_FRAC
from .response import INIT_R, INIT_X, channels_to_c, poly3, resol_to_b, smear
from .types import DatasetDetail

MIN_VARIANCE = 1.0


class FitModel:
    """One dataset on a fixed energy grid; shared parameters enter via c and b."""

    def __init__(self, data, sim, elow: float, ehigh: float,
                 width: float | None = None, sys_frac: float = DEFAULT_SYS_FRAC,
                 *, init_channels=None):
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        if not (self.elow < self.ehigh):
            raise ValueError(
                f"elow ({self.elow}) must be < ehigh ({self.ehigh})")
        if width is not None and width <= 0:
            raise ValueError(f"grid width must be > 0, got {width}")
        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")
        self.sys_frac = float(sys_frac)
        self.min_variance = MIN_VARIANCE

        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        self.init_channels = (INIT_X if init_channels is None
                              else np.asarray(init_channels, dtype=float))
        self.grid_edges = self._make_grid(
            width, c_orig=channels_to_c(self.init_channels))
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])
        self.bin_width = float(np.diff(self.grid_edges)[0])

        # s_ref starts at zero so that the initial scale estimate ignores the
        # bin-width systematic (its model gradient is not meaningful yet); it
        # then holds the estimate for the error weights used during fitting.
        self.s_ref = 0.0
        self.initial_scale = self._initial_scale()
        self.s_ref = float(self.initial_scale)

    # ----- data / model assembly -----------------------------------------

    def error_model(self, d, err_stat, m_prime) -> np.ndarray:
        """Total sigma per bin: stat + fractional sys + calibration-edge term.

        The last term models the count shift from a half-bin-wide calibration
        uncertainty; ``s_ref`` is frozen at the start of each grid pass, which
        keeps the weights constant while the scale itself is fitted.
        """
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * self.s_ref * m_prime) ** 2)
        return np.sqrt(np.maximum(var, self.min_variance))

    def rebin_data(self, c) -> tuple[np.ndarray, np.ndarray]:
        """Integrate data counts/sumw2 onto the grid bins via E(x).

        Requires an increasing calibration, which the fit gate guarantees.
        """
        e_edges = poly3(c, self._ch_edges)
        x_edges = np.interp(self.grid_edges, e_edges, self._ch_edges)
        d_int = np.interp(x_edges, self._ch_edges, self._cum_counts)
        w_int = np.interp(x_edges, self._ch_edges, self._cum_sumw2)
        d = np.diff(d_int)
        var = np.clip(np.diff(w_int), 0.0, None)
        return d, np.sqrt(var)

    def model_counts(self, b) -> np.ndarray:
        return smear(self.sim.counts, self.sim.edges, self.grid_edges, b)

    def sim_on_grid(self) -> np.ndarray:
        """Unsmeared simulation integrated onto the grid (plot reference)."""
        sim_cum = np.concatenate(([0.0], np.cumsum(self.sim.counts)))
        c_lo = np.interp(self.grid_edges[:-1], self.sim.edges, sim_cum)
        c_hi = np.interp(self.grid_edges[1:], self.sim.edges, sim_cum)
        return c_hi - c_lo

    def arrays(self, c, b):
        """Full-grid rebinned data and smeared model (unmasked)."""
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(b)
        return d, err, m_raw

    def pulled(self, c, b, s: float, mask=None):
        """Masked data/model/weight triple shared by all evaluation paths.

        Returns ``(d, w, m_raw, mask)`` restricted to usable bins (positive
        statistical error).  An explicitly passed mask freezes bin selection,
        as required when differencing residuals numerically.
        """
        d, err, m_raw = self.arrays(c, b)
        if mask is None:
            mask = err > 0
        m_prime = np.gradient(m_raw, self.grid_centers)[mask]
        dm, em, mm = d[mask], err[mask], m_raw[mask]
        w = self.error_model(dm, em, m_prime)
        return dm, w, mm, mask

    def dataset_detail(self, label: str, c, b, s: float) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics."""
        dm, w, mm, mask = self.pulled(c, b, s)
        res = (dm - s * mm) / w
        return DatasetDetail(
            label=label, elow=self.elow, ehigh=self.ehigh,
            mu=self.grid_centers[mask],
            d=dm, err=w, m=s * mm, m_raw=mm,
            sim_raw=self.sim_on_grid(), s=float(s),
            chi2=float(res @ res), n_bins=int(len(dm)),
            grid_edges=self.grid_edges,
        )

    # ----- validity --------------------------------------------------------

    @property
    def min_usable_bins(self) -> int:
        return max(10, int(0.1 * len(self.grid_centers)))

    def usable(self, c) -> int:
        return int((self.rebin_data(c)[1] > 0).sum())

    def is_valid(self, c) -> bool:
        return self.usable(c) >= self.min_usable_bins

    def grid_ok(self) -> bool:
        return len(self.grid_centers) >= self.min_usable_bins

    # ----- initialization helpers ------------------------------------------

    def _make_grid(self, width: float | None, c_orig) -> np.ndarray:
        if width is None:
            # About one bin per native data channel within the fit window.
            e = poly3(c_orig, self._ch_edges)
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def _initial_scale(self) -> float:
        c = channels_to_c(self.init_channels)
        b = resol_to_b(INIT_R)
        d, err, m_raw = self.arrays(c, b)
        mask = err > 0
        if not np.any(mask):
            return 1.0
        m_prime = np.gradient(m_raw, self.grid_centers)[mask]
        dm, em, mm = d[mask], err[mask], m_raw[mask]
        var = self.error_model(dm, em, m_prime) ** 2
        smm = float(np.sum(mm * mm / var))
        if smm <= 0:
            return 1.0
        s0 = float(np.sum(dm * mm / var) / smm)
        lo, hi = BOUNDS_S[0]
        return float(np.clip(s0, lo, hi))
