"""Global (multi-dataset) chi-square forward model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitmodel import DEFAULT_SYS_FRAC, FitModel
from .fitparamspace import CALIB, RESOL, FitParamSpace
from .scaling import scale_model
from .types import FitDetail
from .util import broadcast


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
                 init_calib=None):
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
                     init_calib=init_calib)
            for i, s in enumerate(self.specs)
        ]

        n_bins = {m.data.n_bins for m in self.models}
        if len(n_bins) != 1:
            raise ValueError(
                "all datasets must have the same channel count (a "
                f"global calibration is shared): got {sorted(n_bins)}")

        self.channel_max = float(max(m.data.edges[-1] for m in self.models))
        self.param_space = FitParamSpace(tuple(self.labels))
        self.x0 = self.param_space.x0([m.initial_scale for m in self.models])
        self.bounds = self.param_space.bounds

    @property
    def n_channel_bins(self) -> int:
        return min(m.data.n_bins for m in self.models)

    # ----- feasibility gate -------------------------------------------------

    def _gate(self, q) -> tuple[np.ndarray, np.ndarray] | None:
        """Shared scale/resolution cores for q, or None if non-finite."""
        q = np.asarray(q, dtype=float)
        calib = np.asarray(q[CALIB], dtype=float)
        resol = np.asarray(q[RESOL], dtype=float)
        if not (np.all(np.isfinite(calib)) and np.all(np.isfinite(resol))):
            return None
        return calib, resol

    def is_valid(self, q) -> bool:
        """Advisory check that data coverage supports the fit (for x0)."""
        q = np.asarray(q, dtype=float)
        calib = np.asarray(q[CALIB], dtype=float)
        return bool(np.all(np.isfinite(calib))
                    and all(m.is_valid(calib) for m in self.models))

    def binning_ok(self) -> bool:
        return all(m.binning_ok() for m in self.models)

    # ----- evaluation -------------------------------------------------------

    def masks(self, q) -> list[np.ndarray]:
        """Per-dataset usable-bin masks at q (frozen for numerical Jacobians)."""
        gate = self._gate(q)
        if gate is None:
            raise ValueError("cannot freeze masks at an infeasible point")
        calib, _ = gate
        return [m.rebin_data(calib)[1] > 0 for m in self.models]

    def _per_dataset_predictions(self, calib, resol, mask_list=None):
        predictions = []
        for i, m in enumerate(self.models):
            mask = None if mask_list is None else mask_list[i]
            dm, w, mm, msk = m.pulled(calib, resol, mask=mask)
            if len(dm) < m.min_usable_bins:
                return None
            predictions.append((dm, w, mm, m.bin_centers[msk]))
        return predictions

    def residuals(self, q, mask_list=None) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if mask_list is not None:
            sizes = [int(np.sum(m)) for m in mask_list]
        else:
            sizes = [len(m.bin_centers) for m in self.models]
        gate = self._gate(q)
        if gate is None:
            return np.full(int(sum(sizes)), np.nan)
        calib, resol = gate
        predictions = self._per_dataset_predictions(calib, resol, mask_list)
        if predictions is None:
            return np.full(int(sum(sizes)), np.nan)
        out = []
        for i, (dm, w, mm, bc) in enumerate(predictions):
            sb = scale_model(q[self.param_space.scale(i)], bc,
                             self.models[i].elow, self.models[i].ehigh)
            out.append((dm - sb * mm) / w)
        return np.concatenate(out)

    def evaluate(self, q) -> float:
        """Chi-square only; infeasible or under-covered states are inf."""
        gate = self._gate(q)
        if gate is None:
            return np.inf
        calib, resol = gate
        q = np.asarray(q, dtype=float)
        predictions = self._per_dataset_predictions(calib, resol)
        if predictions is None:
            return np.inf
        total = 0.0
        for i, (dm, w, mm, bc) in enumerate(predictions):
            sb = scale_model(q[self.param_space.scale(i)], bc,
                             self.models[i].elow, self.models[i].ehigh)
            res = (dm - sb * mm) / w
            total += float(res @ res)
        return total if np.isfinite(total) else np.inf

    def detail(self, q) -> FitDetail:
        """Full diagnostics; degenerate states yield chi2=inf and valid=False."""
        q = np.asarray(q, dtype=float)
        calib_params = np.asarray(q[CALIB], dtype=float)
        resol_params = np.asarray(q[RESOL], dtype=float)
        scale_params = np.asarray(q[self.param_space.scale_block], dtype=float)
        base = dict(calib_params=calib_params, resol_params=resol_params,
                    scale_params=scale_params,
                    channel_max=self.channel_max,
                    n_channel_bins=self.n_channel_bins)

        gate = self._gate(q)

        entries = []
        for i, m in enumerate(self.models):
            entries.append(m.dataset_detail(self.labels[i], calib_params,
                                            resol_params,
                                            q[self.param_space.scale(i)]))
        if gate is None or any(ds.n_bins < m.min_usable_bins
                               for ds, m in zip(entries, self.models)):
            return FitDetail(datasets=[], chi2=np.inf, ndof=0,
                             **base, valid=False)

        chi2 = float(sum(ds.chi2 for ds in entries))
        ndof = int(sum(ds.n_bins for ds in entries)) - self.param_space.size
        return FitDetail(datasets=entries, chi2=chi2, ndof=ndof,
                         **base, valid=True)

    # ----- binning rebuild ------------------------------------------------------

    def rebuilt(self, calib_core) -> "GlobalFitModel":
        """Same datasets on a binning derived from the given calibration parameters."""
        return GlobalFitModel(self.specs, sys_frac=self.sys_fracs,
                              labels=self.labels,
                              init_calib=np.asarray(calib_core, dtype=float))
