"""Global (multi-dataset) chi-square forward model.

All datasets share one energy calibration and one resolution model; each
dataset owns a normalisation scale.  Feasibility of the shared parameters is
enforced by rejection: non-monotone calibration or negative sigma^2
coefficients evaluate to chi2 = inf, steering the optimizer back without
soft penalty terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitmodel import FitModel
from .params import CHANNELS, DEFAULT_SYS_FRAC, RESOLUTIONS, Space, broadcast
from .response import channels_to_c, coeffs_ok, poly3_is_increasing, resol_to_b
from .types import FitDetail


@dataclass
class DatasetSpec:
    data: object
    sim: object
    elow: float
    ehigh: float
    width: float | None = None


class GlobalFitModel:
    def __init__(self, specs: list[DatasetSpec],
                 sys_frac: float | list[float] = DEFAULT_SYS_FRAC,
                 labels: list[str] | None = None, *,
                 init_channels=None):
        self.specs = list(specs)
        self.n_datasets = len(self.specs)
        if self.n_datasets == 0:
            raise ValueError("GlobalFitModel requires at least one dataset")

        self.sys_fracs = [float(v) for v in
                          broadcast(sys_frac, self.n_datasets, "sys_frac")]
        if labels is None:
            labels = [f"dataset{i + 1}" for i in range(self.n_datasets)]
        self.labels = [str(l) for l in
                       broadcast(labels, self.n_datasets, "labels")]

        self.models = [
            FitModel(s.data, s.sim, s.elow, s.ehigh, width=s.width,
                     sys_frac=self.sys_fracs[i],
                     init_channels=init_channels)
            for i, s in enumerate(self.specs)
        ]

        n_bins = {m.data.n_bins for m in self.models}
        if len(n_bins) != 1:
            raise ValueError(
                "all datasets must have the same channel count (a "
                f"global calibration is shared): got {sorted(n_bins)}")

        self.channel_max = float(max(m.data.edges[-1] for m in self.models))
        self.space = Space(tuple(self.labels),
                           channel_bound=float(min(m.data.n_bins
                                                   for m in self.models)))
        self.x0 = self.space.x0([m.initial_scale for m in self.models])
        self.bounds = self.space.bounds

    @property
    def n_channel_bins(self) -> int:
        return min(m.data.n_bins for m in self.models)

    # ----- feasibility gate -------------------------------------------------

    def _gate(self, q) -> tuple[np.ndarray, np.ndarray] | None:
        """Shared coefficients for q, or None if the state is infeasible."""
        q = np.asarray(q, dtype=float)
        c = channels_to_c(q[CHANNELS])
        b = resol_to_b(q[RESOLUTIONS])
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(b))
                and coeffs_ok(b)):
            return None
        if not poly3_is_increasing(c, self.channel_max):
            return None
        return c, b

    def is_valid(self, q) -> bool:
        """Advisory check that data coverage supports the fit (for x0)."""
        q = np.asarray(q, dtype=float)
        c = channels_to_c(q[CHANNELS])
        return bool(np.all(np.isfinite(c))
                    and all(m.is_valid(c) for m in self.models))

    def grid_ok(self) -> bool:
        return all(m.grid_ok() for m in self.models)

    # ----- evaluation -------------------------------------------------------

    def masks(self, q) -> list[np.ndarray]:
        """Per-dataset usable-bin masks at q (frozen for numerical Jacobians)."""
        gate = self._gate(q)
        if gate is None:
            raise ValueError("cannot freeze masks at an infeasible point")
        c, _ = gate
        return [m.rebin_data(c)[1] > 0 for m in self.models]

    def residuals(self, q, mask_list=None) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if mask_list is not None:
            sizes = [int(np.sum(m)) for m in mask_list]
        else:
            sizes = [len(m.grid_centers) for m in self.models]
        gate = self._gate(q)
        if gate is None:
            return np.full(int(sum(sizes)), np.nan)
        c, b = gate
        s = q[self.space.scales]
        out = []
        for i, m in enumerate(self.models):
            mask = None if mask_list is None else mask_list[i]
            dm, w, mm, _ = m.pulled(c, b, float(s[i]), mask=mask)
            out.append((dm - float(s[i]) * mm) / w)
        return np.concatenate(out)

    def evaluate(self, q) -> float:
        """Chi-square only; infeasible or under-covered states are inf."""
        gate = self._gate(q)
        if gate is None:
            return np.inf
        c, b = gate
        q = np.asarray(q, dtype=float)
        s = q[self.space.scales]
        total = 0.0
        for i, m in enumerate(self.models):
            dm, w, mm, _ = m.pulled(c, b, float(s[i]))
            if len(dm) < m.min_usable_bins:
                return np.inf
            res = (dm - float(s[i]) * mm) / w
            total += float(res @ res)
        return total if np.isfinite(total) else np.inf

    def detail(self, q) -> FitDetail:
        """Full diagnostics; degenerate states yield chi2=inf and valid=False."""
        q = np.asarray(q, dtype=float)
        x = q[CHANNELS]
        r = q[RESOLUTIONS]
        s = np.asarray(q[self.space.scales], dtype=float)
        base = dict(x=x, r=r, s=s, channel_max=self.channel_max,
                    n_channel_bins=self.n_channel_bins)

        gate = self._gate(q)
        c = channels_to_c(x) if gate is None else gate[0]
        b = resol_to_b(r) if gate is None else gate[1]

        entries = []
        for i, m in enumerate(self.models):
            entries.append(m.dataset_detail(self.labels[i], c, b, float(s[i])))
        if gate is None or any(ds.n_bins < m.min_usable_bins
                               for ds, m in zip(entries, self.models)):
            return FitDetail(datasets=[], chi2=np.inf, ndof=0,
                             c=c, b=b, **base, valid=False)

        chi2 = float(sum(ds.chi2 for ds in entries))
        ndof = int(sum(ds.n_bins for ds in entries)) - self.space.size
        return FitDetail(datasets=entries, chi2=chi2, ndof=ndof,
                         c=c, b=b, **base, valid=True)

    # ----- grid rebuild ------------------------------------------------------

    def rebuilt(self, channels) -> "GlobalFitModel":
        """Same datasets on a grid derived from the given channel anchors."""
        return GlobalFitModel(self.specs, sys_frac=self.sys_fracs,
                              labels=self.labels,
                              init_channels=np.asarray(channels, dtype=float))
