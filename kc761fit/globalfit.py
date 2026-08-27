"""Global (multi-dataset) chi^2 forward model — the only fit model.

A fit always runs over N >= 1 (data, simulation, energy-range) pairs: the
energy calibration and the detector resolution are *global-fit* parameters
common to all datasets, and each dataset gets its own normalization scale:

    q = [x60, x609, x1461, x2614, r60, r609, r2614, s0, ..., s_{N-1}]

The total chi^2 is the plain sum of the per-dataset chi^2 values; every
dataset is weighted by its own statistical + fractional-systematic errors, so
no ad-hoc relative weights are needed.  The soft monotonicity penalties of the
global-fit calibration / resolution are counted *once* — they are properties
of the global-fit parameters, not of the datasets.  A single-dataset fit is
simply N = 1.

Each dataset is evaluated by a per-dataset :class:`~kc761fit.fitmodel.FitModel`
(the comparison grid, the exact rebin of the channel data, and the convolution
of the simulation); this class owns the parameter space (including the scale
block), the shared calibration / resolution coefficients, and the sum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .calibration import channels_to_c, monotonicity_penalty as calib_penalty
from .fitmodel import FitModel
from .params import DEFAULT_SYS_FRAC, ParameterSpace, broadcast
from .resolution import monotonicity_penalty as resol_penalty, resol_to_a
from .types import FitDetail


@dataclass
class DatasetSpec:
    """One (data, simulation, energy-range) pair of a fit.

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


def _scale_name(label: str, index: int) -> str:
    """Sanitize a dataset label into a scale-parameter name (``s_<label>``)."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(label)).strip("_")
    return f"s_{clean}" if clean else f"s{index}"


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


class GlobalFitModel:
    """Bundles N (data, simulation, range) pairs and evaluates the summed chi^2.

    ``evaluate`` / ``detail`` / ``residuals`` expect the fit parameter vector
    q = [x60..x2614, r60..r2614, s0..s_{N-1}] (global-fit calibration channels,
    global-fit relative resolutions, one scale per dataset).  ``self.space`` is
    the :class:`~kc761fit.params.ParameterSpace` owning the vector layout;
    ``n_global_params`` is the number of shared (non-scale) parameters.
    """

    def __init__(self, specs, sys_frac: float | list[float] = DEFAULT_SYS_FRAC,
                 labels: list[str] | None = None,
                 *, _models=None, _channels=None):
        """Bundles N (data, simulation, range) pairs and evaluates the summed
        chi^2.  The keyword-only ``_models`` / ``_channels`` arguments are
        internal: ``rebuilt`` passes pre-built per-dataset models and the
        fitted channel positions without re-deriving them from the specs.
        """
        self.specs = [s if isinstance(s, DatasetSpec) else DatasetSpec(*s)
                      for s in specs]
        self.n_datasets = len(self.specs)
        if self.n_datasets == 0:
            raise ValueError("GlobalFitModel requires at least one dataset")

        # Per-dataset fractional systematic error (see FitModel.sys_frac).
        self.sys_fracs = [float(v) for v in
                          broadcast(sys_frac, self.n_datasets, "sys_frac")]

        # Display labels and scale-parameter names for each dataset.
        if labels is None:
            labels = [f"dataset{i + 1}" for i in range(self.n_datasets)]
        self.labels = [str(l) for l in broadcast(labels, self.n_datasets,
                                                 "labels")]
        self.scale_names = [_scale_name(l, i)
                            for i, l in enumerate(self.labels)]

        # One FitModel per dataset (native-resolution grid, own initial scale).
        if _models is None:
            self.models = [
                FitModel(s.data, s.sim, s.elow, s.ehigh, width=s.width,
                         sys_frac=self.sys_fracs[i])
                for i, s in enumerate(self.specs)
            ]
        else:
            self.models = _models

        # The global-fit calibration is a single cubic E(x) shared by every
        # dataset, so all datasets must have the same MCA channel count (and
        # implicitly the same channel->energy gain).
        n_bins = {m.data.n_bins for m in self.models}
        if len(n_bins) != 1:
            raise ValueError(
                "all datasets must have the same channel count (the "
                f"global-fit calibration is shared): got {sorted(n_bins)}")

        # Parameter space: channels within (0, min n_bins) so that every
        # dataset's calibration lines fall inside its channel range;
        # resolutions in BOUNDS_R; one scale bound per dataset (BOUNDS_S).
        # The per-dataset initial scales are the models' own (data-driven WLS
        # estimates at first construction, recomputed by ``rebuilt``).
        min_n = min(m.data.n_bins for m in self.models)
        core = ParameterSpace.from_anchors(min_n)
        self.space = core.with_scales(
            self.n_datasets, names=self.scale_names,
            init=[m.initial_scale for m in self.models])
        self.x0 = self.space.x0
        if _channels is not None:
            self.x0[self.space.channels] = np.asarray(_channels, dtype=float)
        self.bounds = self.space.bounds

    @property
    def n_global_params(self) -> int:
        """Number of shared (calibration + resolution) parameters."""
        return self.space.scale_start

    # -- fixed comparison grids --------------------------------------------
    @property
    def grid_centers(self) -> np.ndarray:
        """Concatenated grid-bin centers over all datasets (fixed grids)."""
        return np.concatenate([m.grid_centers for m in self.models])

    @property
    def channel_max(self) -> float:
        """Largest channel edge over all datasets (plotting range)."""
        return max(m.data.edges[-1] for m in self.models)

    @property
    def n_channel_bins(self) -> int:
        """Smallest channel count over all datasets (calibration plot range)."""
        return min(m.data.n_bins for m in self.models)

    def rebuilt(self, channels):
        """New GlobalFitModel whose per-dataset grids follow the global-fit
        calibration fixed by the fitted channel positions ``channels``.

        Used to re-bin every dataset after a first pass, so each grid matches
        the actual (global-fit) channel-to-energy density at that dataset's
        native resolution.  The fit warm-starts from the previous pass's
        solution anyway, so only the grids matter here.
        """
        channels = np.asarray(channels, dtype=float)
        models = [m.rebuilt(channels) for m in self.models]
        return GlobalFitModel(self.specs, sys_frac=self.sys_fracs,
                              labels=self.labels, _models=models,
                              _channels=channels)

    def grid_ok(self) -> bool:
        """True if every per-dataset comparison grid is large enough to fit.

        A global model concatenates its per-dataset grids, so the degeneracy
        of the *concatenated* grid is not a meaningful check: each dataset's
        own grid must be usable.
        """
        return all(m.grid_ok() for m in self.models)

    # -- validity -----------------------------------------------------------
    def is_valid(self, q) -> bool:
        """True unless the point is degenerate for *any* dataset (insufficient
        data coverage at the global-fit calibration)."""
        q = np.asarray(q, dtype=float)
        return all(m.is_valid(q[:self.n_global_params]) for m in self.models)

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

    def residuals(self, q, mask=None) -> np.ndarray:
        """Weighted residuals (d - s_i m)/sigma, concatenated over datasets.

        Each dataset's bins are weighted with its own scale s_i and its own
        total errors (statistical + fractional-systematic + x-direction, via
        ``FitModel.error_model``); ``mask`` (concatenated, e.g.
        ``detail().mask``) selects the grid bins.  This is the vector whose
        central-difference Jacobian gives the parameter uncertainties.
        """
        q = np.asarray(q, dtype=float)
        c = channels_to_c(q[self.space.channels])
        a = resol_to_a(q[self.space.resolutions])
        q_core = q[:self.n_global_params]
        masks = (self._split_mask(mask) if mask is not None
                 else [None] * self.n_datasets)
        res = []
        for i, (m, msk) in enumerate(zip(self.models, masks)):
            d, err, m_raw = m.arrays(q_core, c, a)
            s_i = float(q[self.space.scale_start + i])
            ds = m.dataset_detail(d, err, m_raw, s_i, mask=msk)
            res.append((ds.d - ds.s * ds.m_raw) / ds.err)
        return np.concatenate(res)

    def detail(self, q) -> FitDetail:
        """Masked evaluation at fit parameters q (always a FitDetail).

        ``q`` = [x60..x2614, r60..r2614, s0..s_{N-1}].  The detail carries the
        global-fit fitted channels (x) / resolutions (r), the derived
        coefficients (c, a), the per-dataset scales (s), the total data chi^2
        (``chi2``), the per-dataset contributions (``chi2_per_dataset`` /
        ``bins_per_dataset``), the soft monotonicity penalty (``pen``, counted
        once) and, in ``datasets``, the per-dataset masked arrays.  Ordering
        violations of the global-fit channels / resolutions add a quadratically
        rising penalty (zero for a physically ordered point); a degenerate
        point (any dataset with insufficient data coverage) returns chi^2=inf
        (with ``valid=False``).
        """
        q = np.asarray(q, dtype=float)
        x = np.asarray(q[self.space.channels], dtype=float)
        r = np.asarray(q[self.space.resolutions], dtype=float)
        s = np.asarray(q[self.space.scales], dtype=float)
        c = channels_to_c(x)
        a = resol_to_a(r)
        p = np.concatenate([x, r, s])
        q_core = q[:self.n_global_params]

        entries = []
        chi2_per = np.empty(self.n_datasets)
        bins_per = np.empty(self.n_datasets, dtype=int)
        degenerate = False
        for i, m in enumerate(self.models):
            d, err, m_raw = m.arrays(q_core, c, a)
            mask = err > 0
            if int(np.sum(mask)) < m.min_usable_bins:
                degenerate = True
            ds = m.dataset_detail(d, err, m_raw, float(s[i]), mask=mask,
                                  label=self.labels[i])
            chi2_per[i] = ds.chi2
            bins_per[i] = ds.n_bins
            entries.append(ds)

        if degenerate:
            return degenerate_detail(self.space, p, c, a)

        chi2 = float(np.sum(chi2_per))
        ndof = int(np.sum(bins_per)) - self.space.size
        pen = calib_penalty(x) + resol_penalty(r)
        return FitDetail(
            datasets=entries,
            s=s, x=x, r=r, c=c, a=a,
            chi2=chi2, chi2_per_dataset=chi2_per, bins_per_dataset=bins_per,
            ndof=ndof, pen=pen,
        )

    def evaluate(self, q) -> float:
        det = self.detail(q)
        chi2 = det.chi2
        if not np.isfinite(chi2):
            return np.inf
        return chi2 + det.pen
