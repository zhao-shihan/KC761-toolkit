"""Global (multi-dataset) chi-square forward model."""

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
    data: object
    sim: object
    elow: float
    ehigh: float
    width: float | None = None


def _scale_name(label: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(label)).strip("_")
    return f"s_{clean}" if clean else f"s{index}"


def degenerate_detail(space, p, c, a) -> FitDetail:
    return FitDetail(
        datasets=[], s=np.asarray(p[space.scales], dtype=float),
        chi2=np.inf, ndof=0, pen=0.0,
        c=c, a=a, x=p[space.channels], r=p[space.resolutions],
        valid=False,
    )


class GlobalFitModel:
    def __init__(self, specs, sys_frac: float | list[float] = DEFAULT_SYS_FRAC,
                 labels: list[str] | None = None,
                 *, _models=None, _channels=None):
        self.specs = [s if isinstance(s, DatasetSpec) else DatasetSpec(*s)
                      for s in specs]
        self.n_datasets = len(self.specs)
        if self.n_datasets == 0:
            raise ValueError("GlobalFitModel requires at least one dataset")

        self.sys_fracs = [float(v) for v in
                          broadcast(sys_frac, self.n_datasets, "sys_frac")]

        if labels is None:
            labels = [f"dataset{i + 1}" for i in range(self.n_datasets)]
        self.labels = [str(l) for l in broadcast(labels, self.n_datasets,
                                                 "labels")]
        self.scale_names = [_scale_name(l, i)
                            for i, l in enumerate(self.labels)]

        if _models is None:
            self.models = [
                FitModel(s.data, s.sim, s.elow, s.ehigh, width=s.width,
                         sys_frac=self.sys_fracs[i])
                for i, s in enumerate(self.specs)
            ]
        else:
            self.models = _models

        n_bins = {m.data.n_bins for m in self.models}
        if len(n_bins) != 1:
            raise ValueError(
                "all datasets must have the same channel count (the "
                f"global-fit calibration is shared): got {sorted(n_bins)}")

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
        return self.space.scale_start

    @property
    def grid_centers(self) -> np.ndarray:
        return np.concatenate([m.grid_centers for m in self.models])

    @property
    def channel_max(self) -> float:
        return max(m.data.edges[-1] for m in self.models)

    @property
    def n_channel_bins(self) -> int:
        return min(m.data.n_bins for m in self.models)

    def rebuilt(self, channels):
        channels = np.asarray(channels, dtype=float)
        models = [m.rebuilt(channels) for m in self.models]
        return GlobalFitModel(self.specs, sys_frac=self.sys_fracs,
                              labels=self.labels, _models=models,
                              _channels=channels)

    def grid_ok(self) -> bool:
        return all(m.grid_ok() for m in self.models)

    def is_valid(self, q) -> bool:
        q = np.asarray(q, dtype=float)
        return all(m.is_valid(q[:self.n_global_params]) for m in self.models)

    def _split_mask(self, mask):
        split = []
        start = 0
        for m in self.models:
            n = len(m.grid_centers)
            split.append(mask[start:start + n])
            start += n
        return split

    def residuals(self, q, mask=None) -> np.ndarray:
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
