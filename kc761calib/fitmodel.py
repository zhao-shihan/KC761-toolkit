"""Per-dataset forward model: calibrated rebinning, smearing, weighting."""

from __future__ import annotations

import numpy as np

from .response import INIT_CALIB, INIT_RESOL, calib_model, smear
from .scaling import scale_model
from .types import DatasetDetail

DEFAULT_SYS_FRAC = 0.05


class FitModel:
    """One dataset on a fixed energy binning; shared parameters enter via c and b."""

    def __init__(self, data, sim, elow: float, ehigh: float,
                 width: float | None = None, sys_frac: float = DEFAULT_SYS_FRAC,
                 *, init_calib=None):
        self.data = data
        self.sim = sim
        self.elow = float(elow)
        self.ehigh = float(ehigh)
        if not (self.elow < self.ehigh):
            raise ValueError(
                f"elow ({self.elow}) must be < ehigh ({self.ehigh})")
        if width is not None and width <= 0:
            raise ValueError(f"bin width must be > 0, got {width}")
        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")
        self.sys_frac = float(sys_frac)

        self._cum_counts = np.concatenate(([0.0], np.cumsum(data.counts)))
        self._cum_sumw2 = np.concatenate(([0.0], np.cumsum(data.errors**2)))
        self._ch_edges = data.edges
        self.x_max = float(data.edges[-1])

        init_calib = INIT_CALIB if init_calib is None else np.asarray(
            init_calib, dtype=float)
        self.init_calib = np.asarray(init_calib, dtype=float)
        self.init_resol = INIT_RESOL
        self.bin_edges = self._make_binning(width, calib_orig=self.init_calib)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = float(np.diff(self.bin_edges)[0])

        self._sim_centers = 0.5 * (self.sim.edges[:-1] + self.sim.edges[1:])

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
        uncertainty; ``s_ref`` is frozen at the start of each binning pass, which
        keeps the weights constant while the scale itself is fitted.
        """
        var = (err_stat**2 + (self.sys_frac * d) ** 2
               + (0.5 * self.bin_width * self.s_ref * m_prime) ** 2)
        return np.sqrt(np.maximum(var, 1.0))  # bound min error to 1

    def rebin_data(self, calib) -> tuple[np.ndarray, np.ndarray]:
        """Integrate data counts/sumw2 onto the bins via E(x).

        Requires an increasing calibration, which the fit gate guarantees.
        """
        e_edges = calib_model(calib, self._ch_edges, self.x_max)
        x_edges = np.interp(self.bin_edges, e_edges, self._ch_edges)
        d_int = np.interp(x_edges, self._ch_edges, self._cum_counts)
        w_int = np.interp(x_edges, self._ch_edges, self._cum_sumw2)
        d = np.diff(d_int)
        var = np.clip(np.diff(w_int), 0.0, None)
        return d, np.sqrt(var)

    def model_counts(self, resol) -> np.ndarray:
        return smear(self.sim.counts, self.sim.edges, self.bin_edges, resol,
                     sim_centers=self._sim_centers)

    def sim_on_bins(self) -> np.ndarray:
        """Unsmeared simulation integrated onto the bins (plot reference)."""
        sim_cum = np.concatenate(([0.0], np.cumsum(self.sim.counts)))
        c_lo = np.interp(self.bin_edges[:-1], self.sim.edges, sim_cum)
        c_hi = np.interp(self.bin_edges[1:], self.sim.edges, sim_cum)
        return c_hi - c_lo

    def arrays(self, calib, resol):
        """Full rebinned data and smeared model (unmasked)."""
        d, err = self.rebin_data(calib)
        m_raw = self.model_counts(resol)
        return d, err, m_raw

    def pulled(self, calib, resol, mask=None):
        """Masked data/model/weight triple shared by all evaluation paths.

        Returns ``(d, w, m_raw, mask)`` restricted to usable bins (positive
        statistical error).  ``m_raw`` is the unscaled model; callers apply the
        per-dataset scale themselves.  An explicitly passed mask freezes bin
        selection, as required when differencing residuals numerically.
        """
        d, err, m_raw = self.arrays(calib, resol)
        if mask is None:
            mask = err > 0
        m_prime = np.gradient(m_raw, self.bin_centers)[mask]
        dm, em, mm = d[mask], err[mask], m_raw[mask]
        w = self.error_model(dm, em, m_prime)
        return dm, w, mm, mask

    def dataset_detail(self, label: str, calib, resol,
                       scale_params) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics.

        The model prediction is ``s(E) * m(E)`` with the per-bin scale curve
        ``s(E) = scale_model(scale_params, E, elow, ehigh)`` evaluated at each
        bin center; the ``smeared_sim``/``raw_sim`` fields remain unscaled.
        """
        dm, w, mm, mask = self.pulled(calib, resol)
        scale_full = scale_model(scale_params, self.bin_centers, self.elow,
                                 self.ehigh)
        scale_masked = scale_full[mask]
        res = (dm - scale_masked * mm) / w
        return DatasetDetail(
            label=label, elow=self.elow, ehigh=self.ehigh,
            bin_centers=self.bin_centers[mask],
            bin_counts=dm, sigma=w, smeared_model=scale_masked * mm,
            smeared_sim=mm, raw_sim=self.sim_on_bins(),
            scale_params=np.asarray(scale_params, dtype=float),
            chi2=float(res @ res), n_bins=int(len(dm)),
            bin_edges=self.bin_edges,
        )

    # ----- validity --------------------------------------------------------

    @property
    def min_usable_bins(self) -> int:
        return max(10, int(0.1 * len(self.bin_centers)))

    def usable(self, calib) -> int:
        return int((self.rebin_data(calib)[1] > 0).sum())

    def is_valid(self, calib) -> bool:
        return self.usable(calib) >= self.min_usable_bins

    def binning_ok(self) -> bool:
        return len(self.bin_centers) >= self.min_usable_bins

    # ----- initialization helpers ------------------------------------------

    def _make_binning(self, width: float | None, calib_orig) -> np.ndarray:
        if width is None:
            # About one bin per native data channel within the fit window.
            e = calib_model(calib_orig, self._ch_edges, self.x_max)
            x_lo = np.interp(self.elow, e, self._ch_edges)
            x_hi = np.interp(self.ehigh, e, self._ch_edges)
            n_ch = max(1.0, x_hi - x_lo)
            width = max(1.0, (self.ehigh - self.elow) / n_ch)
        n = max(2, int(np.ceil((self.ehigh - self.elow) / width)))
        return np.linspace(self.elow, self.ehigh, n + 1)

    def _initial_scale(self) -> float:
        calib = self.init_calib
        resol = self.init_resol
        d, err, m_raw = self.arrays(calib, resol)
        mask = err > 0
        if not np.any(mask):
            raise ValueError("cannot estimate the initial scale: no usable bins "
                             "(no positive statistical error)")
        m_prime = np.gradient(m_raw, self.bin_centers)[mask]
        dm, em, mm = d[mask], err[mask], m_raw[mask]
        var = self.error_model(dm, em, m_prime) ** 2
        smm = float(np.sum(mm * mm / var))
        if smm <= 0:
            raise ValueError("cannot estimate the initial scale: zero model "
                             "normalization in the usable bins")
        s0 = float(np.sum(dm * mm / var) / smm)
        if not np.isfinite(s0) or s0 <= 0.0:
            raise ValueError(
                f"cannot estimate the initial scale: got non-positive or "
                f"non-finite value {s0}")
        return s0
