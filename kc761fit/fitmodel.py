"""Per-dataset chi-square forward model."""

from __future__ import annotations

import numpy as np

from .calibration import INIT_X, channels_to_c, poly3
from .params import BOUNDS_S, DEFAULT_SYS_FRAC, ParameterSpace
from .resolution import INIT_R, resol_to_a, smear
from .types import DatasetDetail

MIN_VARIANCE = 1.0


class FitModel:
    def __init__(self, data, sim, elow: float, ehigh: float,
                 width: float | None = None, sys_frac: float = DEFAULT_SYS_FRAC,
                 *, _channels=None):
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        if not (self.elow < self.ehigh):
            raise ValueError(
                f"elow ({self.elow}) must be < ehigh ({self.ehigh})")
        if width is not None and width <= 0:
            raise ValueError(f"grid width must be > 0, got {width}")
        self.sys_frac = float(sys_frac)
        self.min_variance = MIN_VARIANCE

        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")

        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges

        self.space = ParameterSpace.from_anchors(data.n_bins)

        grid_channels = (np.asarray(_channels, dtype=float)
                         if _channels is not None else INIT_X)
        self.grid_edges = self._make_grid(width, c_orig=channels_to_c(
            grid_channels))
        self.grid_centers = 0.5 * (self.grid_edges[:-1] + self.grid_edges[1:])
        self.bin_width = float(np.diff(self.grid_edges)[0])

        self.x0_core = np.concatenate([grid_channels, INIT_R])
        self.s_ref = 0.0
        self.initial_scale = self._initial_scale(self.x0_core)
        self.s_ref = float(self.initial_scale)

    def error_model(self, d, err_stat, m_prime) -> np.ndarray:
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * self.s_ref * m_prime) ** 2)
        return np.sqrt(np.maximum(var, self.min_variance))

    def _initial_scale(self, q_core) -> float:
        d, err, m_raw = self.arrays(q_core)
        mask = err > 0
        if not np.any(mask):
            return 1.0
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

    def _make_grid(self, width: float | None, c_orig) -> np.ndarray:
        if width is None:
            e = poly3(c_orig, self._ch_edges)
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def rebuilt(self, channels):
        return FitModel(self.data, self.sim, self.elow, self.ehigh,
                        sys_frac=self.sys_frac, _channels=channels)

    def grid_ok(self) -> bool:
        return len(self.grid_centers) >= self.min_usable_bins

    def rebin_data(self, c) -> tuple[np.ndarray, np.ndarray]:
        e_edges = poly3(c, self._ch_edges)
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

    @property
    def min_usable_bins(self) -> int:
        return max(10, int(0.1 * len(self.grid_centers)))

    def is_valid(self, q_core) -> bool:
        d, err, m_raw = self.arrays(q_core)
        return int(np.sum(err > 0)) >= self.min_usable_bins

    def model_counts(self, a) -> np.ndarray:
        return smear(self.sim.counts, self.sim.edges, self.grid_edges, a)

    def raw_model_counts(self) -> np.ndarray:
        sim_cum = np.concatenate(([0.0], np.cumsum(self.sim.counts)))
        c_lo = np.interp(self.grid_edges[:-1], self.sim.edges, sim_cum)
        c_hi = np.interp(self.grid_edges[1:], self.sim.edges, sim_cum)
        return c_hi - c_lo

    def arrays(self, q_core, c=None, a=None):
        if c is None:
            c = channels_to_c(q_core[self.space.channels])
        if a is None:
            a = resol_to_a(q_core[self.space.resolutions])
        d, err = self.rebin_data(c)
        m_raw = self.model_counts(a)
        return d, err, m_raw

    def dataset_detail(self, d, err, m_raw, s, mask=None,
                       label: str = "") -> DatasetDetail:
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
