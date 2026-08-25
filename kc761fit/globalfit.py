"""Global (multi-dataset) chi^2 forward model.

A single :class:`~kc761fit.fitmodel.FitModel` fits one experimental spectrum
against one simulation over one energy range.  A global fit instead fits
*several* (data, simulation, energy-range) pairs at once: the energy
calibration and the detector resolution are *global-fit* parameters common
to all datasets, and each dataset gets its own normalization scale:

    q = [x60, x609, x1461, x2614, r60, r609, r2614, s0, ..., s_{N-1}]

The total chi^2 is the plain sum of the per-dataset chi^2 values.  Every
dataset is weighted by its own statistical errors (plus its fractional
systematic error), so no ad-hoc relative weights are needed.  The soft
monotonicity penalties of the global-fit calibration / resolution are counted
*once* — they are properties of the global-fit parameters, not of the
datasets.

Typical use: fit the Th-232 spectrum over 300-3000 keV (which pins the
high-energy anchors of the global-fit curves) together with the Am-241
spectrum over 20-80 keV (which constrains the low-energy region).  Each
dataset keeps its own native-resolution energy grid (one bin per data
channel), rebuilt between fit passes from the global-fit fitted calibration,
so the comparison is always at the actual channel-to-energy density of every
dataset.

Model evaluation
----------------
Per dataset the model is exactly the :class:`~kc761fit.fitmodel.FitModel`
forward model: the experimental channel spectrum is calibrated with the
global-fit cubic E(x) and exactly rebinned onto the dataset's fixed energy
grid; the dataset's simulation is convolved with the global-fit Gaussian
resolution directly onto the same grid; and chi^2 is summed over grid bins
with positive error using the statistical + fractional-systematic total
errors.  A point is degenerate (chi^2 = inf) if *any* dataset lacks
sufficient data coverage at the global-fit calibration; ordering violations
of the global-fit channels / resolutions are handled by the soft penalties
(see :mod:`kc761fit.fitmodel`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .calibration import channels_to_c, monotonicity_penalty as calib_penalty
from .fitmodel import INIT_PARAMS, PARAM_NAMES, FitModel
from .resolution import (
    BOUNDS_R, monotonicity_penalty as resol_penalty, resol_to_a,
)


@dataclass
class DatasetSpec:
    """One (data, simulation, energy-range) pair of a global fit.

    ``data`` / ``sim`` are :class:`kc761fit.io.Spectrum` objects; ``elow`` /
    ``ehigh`` the fit energy range in keV; ``width`` an optional fixed
    energy-grid bin width (default: about one data channel width, estimated
    per dataset).
    """

    data: object
    sim: object
    elow: float
    ehigh: float
    width: float | None = None


def _as_seq(value, n: int, name: str) -> list:
    """Broadcast ``value`` (scalar / length-1 / length-n) to length ``n``."""
    if np.isscalar(value) or isinstance(value, (str, bytes)):
        return [value] * n
    seq = list(value)
    if len(seq) == n:
        return seq
    if len(seq) == 1:
        return seq * n
    raise ValueError(f"{name}: expected 1 or {n} values, got {len(seq)}")


def _scale_name(label: str, index: int) -> str:
    """Sanitize a dataset label into a scale-parameter name (``s_<label>``)."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(label)).strip("_")
    return f"s_{clean}" if clean else f"s{index}"


class GlobalFitModel:
    """Bundles N (data, simulation, range) pairs and evaluates the summed chi^2.

    ``evaluate`` / ``detail`` / ``residuals`` expect the fit parameter vector
    q = [x60..x2614, r60..r2614, s0..s_{N-1}] (global-fit calibration
    channels, global-fit relative resolutions, one scale per dataset).  The
    reported ``detail`` dict carries the derived coefficients (c, a), the
    global-fit fitted channels / resolutions (x, r), the per-dataset scales,
    and — in ``datasets`` — the per-dataset masked arrays and chi^2.
    """

    # Number of global-fit parameters (calibration + resolution), common to
    # all datasets.
    n_global_params = 7

    def __init__(self, specs, sys_frac: float | list[float] = 0.0,
                 labels: list[str] | None = None):
        self.specs = [s if isinstance(s, DatasetSpec) else DatasetSpec(*s)
                      for s in specs]
        self.n_datasets = len(self.specs)
        if self.n_datasets == 0:
            raise ValueError("GlobalFitModel requires at least one dataset")

        # Per-dataset fractional systematic error (see FitModel.sys_frac).
        self.sys_fracs = [float(v) for v in
                          _as_seq(sys_frac, self.n_datasets, "sys_frac")]

        # Display labels and scale-parameter names for each dataset.
        if labels is None:
            labels = [f"dataset{i + 1}" for i in range(self.n_datasets)]
        self.labels = [str(l) for l in _as_seq(labels, self.n_datasets, "labels")]
        self.scale_names = [_scale_name(l, i)
                            for i, l in enumerate(self.labels)]

        # One FitModel per dataset (native-resolution grid, own initial scale).
        self.models = [
            FitModel(s.data, s.sim, s.elow, s.ehigh, width=s.width,
                     sys_frac=self.sys_fracs[i])
            for i, s in enumerate(self.specs)
        ]

        self.x0 = self._build_x0()
        self.bounds = self._build_bounds()
        self.param_names = list(PARAM_NAMES[:7]) + self.scale_names

    # -- construction helpers ---------------------------------------------
    def _build_x0(self) -> np.ndarray:
        """Initial parameters: global-fit channels / resolutions (initial
        values) plus the per-dataset weighted least-squares scale estimates."""
        return np.concatenate(
            [np.asarray(INIT_PARAMS[:7], dtype=float)] +
            [np.array([m.x0[7]]) for m in self.models])

    def _build_bounds(self) -> list[tuple[float, float]]:
        """Box bounds: global-fit channels within (0, min n_bins) so that
        every dataset's calibration lines fall inside its channel range;
        resolutions in BOUNDS_R; one scale bound per dataset."""
        min_n = min(m.data.n_bins for m in self.models)
        return ([(0.0, min_n)] * 4 + BOUNDS_R
                + [(1e-3, 1e3)] * self.n_datasets)

    # -- fixed comparison grids --------------------------------------------
    @property
    def grid_centers(self) -> np.ndarray:
        """Concatenated grid-bin centers over all datasets (fixed grids)."""
        return np.concatenate([m.grid_centers for m in self.models])

    def rebuilt(self, channels):
        """New GlobalFitModel whose per-dataset grids follow the global-fit
        calibration fixed by the fitted channel positions ``channels``.

        Used to re-bin every dataset after a first pass, so each grid matches
        the actual (global-fit) channel-to-energy density at that dataset's
        native resolution.  The initial resolutions are kept at the initial
        values; the per-dataset initial scales are re-estimated for the new
        grids.
        """
        channels = np.asarray(channels, dtype=float)
        g = GlobalFitModel(self.specs, sys_frac=self.sys_fracs,
                           labels=self.labels)
        g.models = [m.rebuilt(channels) for m in self.models]
        g.x0 = np.concatenate(
            [channels, g.models[0].x0[4:7],
             np.array([mi.x0[7] for mi in g.models])])
        g.bounds = g._build_bounds()
        return g

    # -- validity -----------------------------------------------------------
    def is_valid(self, q) -> bool:
        """True unless the point is degenerate for *any* dataset (insufficient
        data coverage at the global-fit calibration)."""
        q = np.asarray(q, dtype=float)
        return all(m.is_valid(q[:7]) for m in self.models)

    # -- evaluation ---------------------------------------------------------
    def _split_mask(self, mask):
        """Split a concatenated grid mask into per-dataset masks."""
        split = []
        start = 0
        for m in self.models:
            n = len(m.grid_centers)
            split.append(mask[start:start + n])
            start += n
        return split

    def arrays(self, q, c=None, a=None):
        """Concatenated (d, err, m_raw) over all datasets on their grids.

        ``err`` is the statistical + fractional-systematic per-bin error
        (per-dataset ``sys_frac``); the x-direction (finite bin-width) term
        is applied in ``detail`` / ``residuals`` via ``FitModel.model_error``.
        """
        q = np.asarray(q, dtype=float)
        if c is None:
            c = channels_to_c(q[:4])
        if a is None:
            a = resol_to_a(q[4:7])
        d_all, err_all, m_all = [], [], []
        for m in self.models:
            d, err, m_raw = m.arrays(q[:7], c, a)
            err = m.total_errors(d, err)
            d_all.append(d)
            err_all.append(err)
            m_all.append(m_raw)
        return (np.concatenate(d_all), np.concatenate(err_all),
                np.concatenate(m_all))

    def residuals(self, q, mask=None) -> np.ndarray:
        """Weighted residuals (d - s_i m)/sigma, concatenated over datasets.

        Each dataset's bins are weighted with its own scale s_i and its own
        total errors (statistical + fractional-systematic + x-direction,
        via ``FitModel.model_error``); ``mask`` (concatenated, e.g.
        ``detail()["mask"]``) selects the grid bins.  This is the vector
        whose central-difference Jacobian gives the parameter uncertainties.
        """
        q = np.asarray(q, dtype=float)
        c = channels_to_c(q[:4])
        a = resol_to_a(q[4:7])
        masks = (self._split_mask(mask) if mask is not None
                 else [None] * self.n_datasets)
        res = []
        for i, (m, msk) in enumerate(zip(self.models, masks)):
            d, err, m_raw = m.arrays(q[:7], c, a)
            m_prime = np.gradient(m_raw, m.grid_centers)
            if msk is not None:
                d, err, m_raw, m_prime = (d[msk], err[msk], m_raw[msk],
                                          m_prime[msk])
            s_i = float(q[7 + i])
            err = m.model_error(d, err, s_i, m_prime)
            res.append((d - s_i * m_raw) / err)
        return np.concatenate(res)

    @staticmethod
    def _degenerate_detail(p, c, a) -> dict:
        """Placeholder detail for an infeasible parameter point.

        chi^2 = inf rejects the point for any optimizer; a valid fit is never
        affected, so this cannot bias the result.
        """
        return dict(
            datasets=[], d=np.array([]), err=np.array([]), m_raw=np.array([]),
            m=np.array([]), mu=np.array([]), mask=None,
            s=np.asarray(p[7:], dtype=float), x=p[:4], r=p[4:7], c=c, a=a,
            chi2=np.inf, chi2_per_dataset=np.array([]),
            bins_per_dataset=np.array([]), ndof=0, pen=0.0,
            grid_centers=None,
        )

    def detail(self, q) -> dict:
        """Masked evaluation at fit parameters q (always a dict).

        ``q`` = [x60..x2614, r60..r2614, s0..s_{N-1}].  The dict carries the
        global-fit fitted channels (x) / resolutions (r), the derived
        coefficients (c, a), the per-dataset scales (s), the total data chi^2
        (``chi2``), the per-dataset contributions (``chi2_per_dataset`` /
        ``bins_per_dataset``), the soft monotonicity penalty (``pen``, counted
        once) and, in ``datasets``, the per-dataset masked arrays.  Ordering
        violations of the global-fit channels / resolutions add a quadratically
        rising penalty (zero for a physically ordered point); a degenerate
        point (any dataset with insufficient data coverage) returns chi^2=inf.
        """
        q = np.asarray(q, dtype=float)
        x = np.asarray(q[:4], dtype=float)
        r = np.asarray(q[4:7], dtype=float)
        s = np.asarray(q[7:7 + self.n_datasets], dtype=float)
        c = channels_to_c(x)
        a = resol_to_a(r)
        p = np.concatenate([x, r, s])

        entries = []
        d_all, err_all, m_raw_all, m_all, mu_all, mask_all = [], [], [], [], [], []
        chi2_per = np.empty(self.n_datasets)
        bins_per = np.empty(self.n_datasets, dtype=int)
        degenerate = False
        for i, m in enumerate(self.models):
            d, err, m_raw = m.arrays(q[:7], c, a)
            mask_i = err > 0
            if int(np.sum(mask_i)) < m.min_usable_bins:
                degenerate = True
            # Model slope over the full grid (for the x-direction error
            # term), then mask everything consistently.
            m_prime = np.gradient(m_raw, m.grid_centers)
            d, err, m_raw, m_prime = (d[mask_i], err[mask_i], m_raw[mask_i],
                                      m_prime[mask_i])
            s_i = float(s[i])
            err = m.model_error(d, err, s_i, m_prime)
            mu_i = m.grid_centers[mask_i]
            mi = s_i * m_raw
            chi2_per[i] = float(np.sum((d - mi) ** 2 / err**2))
            bins_per[i] = len(d)
            entries.append(dict(
                d=d, err=err, m_raw=m_raw, m=mi, s=s_i, mu=mu_i,
                chi2=float(chi2_per[i]), bins=int(bins_per[i]),
                mask=mask_i, grid_centers=m.grid_centers, model=m,
            ))
            d_all.append(d)
            err_all.append(err)
            m_raw_all.append(m_raw)
            m_all.append(mi)
            mu_all.append(mu_i)
            mask_all.append(mask_i)

        if degenerate:
            return self._degenerate_detail(p, c, a)

        chi2 = float(np.sum(chi2_per))
        ndof = int(np.sum(bins_per)) - (self.n_global_params + self.n_datasets)
        pen = calib_penalty(x) + resol_penalty(r)
        return dict(
            datasets=entries,
            d=np.concatenate(d_all), err=np.concatenate(err_all),
            m_raw=np.concatenate(m_raw_all), m=np.concatenate(m_all),
            mu=np.concatenate(mu_all), mask=np.concatenate(mask_all),
            s=s, x=x, r=r, c=c, a=a,
            chi2=chi2, chi2_per_dataset=chi2_per, bins_per_dataset=bins_per,
            ndof=ndof, pen=pen, grid_centers=self.grid_centers,
        )

    def evaluate(self, q) -> float:
        det = self.detail(q)
        chi2 = det["chi2"]
        if not np.isfinite(chi2):
            return np.inf
        return chi2 + det["pen"]
