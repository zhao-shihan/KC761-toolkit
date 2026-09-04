"""Global (multi-dataset) chi-square forward model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .folding import Response
from .fitmodel import DEFAULT_SYS_FRAC, FitModel
from .fitparamspace import CALIB, RESOL, FitParamSpace
from .response import INIT_CALIB, INIT_RESOL
from .types import DatasetArrays, FitDetail
from .util import broadcast

if TYPE_CHECKING:
    from kc761util.spectrum import Spectrum


@dataclass
class DatasetSpec:
    data: Spectrum
    sim: Spectrum
    channel_low: int
    channel_high: int


class GlobalFitModel:
    def __init__(self, specs: list[DatasetSpec],
                 sys_frac: float | list[float] = DEFAULT_SYS_FRAC,
                 labels: list[str] | None = None, *,
                 init_calib_params=None):
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

        n_bins = {s.data.n_bins for s in self.specs}
        if len(n_bins) != 1:
            raise ValueError(
                "all datasets must have the same channel count (a "
                f"global calibration is shared): got {sorted(n_bins)}")

        # Shared channel geometry: the fit range is the union of the datasets'
        # fit ranges; the extended binning and response matrix are built
        # around it once per evaluation.
        self.channel_max = float(max(s.data.edges[-1] for s in self.specs))
        self.last_channel = int(min(n_bins)) - 1
        self.fit_channel_lo = min(s.channel_low for s in self.specs)
        self.fit_channel_hi = max(s.channel_high for s in self.specs)

        init_calib_params = (INIT_CALIB if init_calib_params is None
                             else np.asarray(init_calib_params, dtype=float))
        self.init_response = Response.build(
            init_calib_params, INIT_RESOL, self.channel_max, self.fit_channel_lo,
            self.fit_channel_hi, self.last_channel)

        self.models = [
            FitModel(s.data, s.sim, s.channel_low, s.channel_high,
                     sys_frac=self.sys_fracs[i],
                     init_response=self.init_response)
            for i, s in enumerate(self.specs)
        ]

        self.param_space = FitParamSpace(
            tuple(self.labels), tuple(m.initial_scale for m in self.models),
            tuple((m.channel_low, m.channel_high) for m in self.models))
        self.x0 = self.param_space.x0()
        self.bounds = self.param_space.bounds

        # Memoization state for the per-evaluation response/projection
        # (see _cached_projection).
        self._proj_cache_key = None
        self._proj_cache = None

    # ----- feasibility gate -------------------------------------------------

    def _gate(self, q) -> tuple[np.ndarray, np.ndarray] | None:
        """Shared calibration/resolution cores for q, or None if non-finite."""
        q = np.asarray(q, dtype=float)
        calib_params = np.asarray(q[CALIB], dtype=float)
        resol_params = np.asarray(q[RESOL], dtype=float)
        if not (np.all(np.isfinite(calib_params))
                and np.all(np.isfinite(resol_params))):
            return None
        return calib_params, resol_params

    def is_valid(self, q) -> bool:
        """Advisory check that data coverage supports the fit (for x0)."""
        q = np.asarray(q, dtype=float)
        calib_params = np.asarray(q[CALIB], dtype=float)
        return bool(np.all(np.isfinite(calib_params))
                    and all(m.is_valid(calib_params) for m in self.models))

    # ----- evaluation -------------------------------------------------------

    def masks(self, q) -> list[np.ndarray]:
        """Per-dataset usable-bin masks (fixed; frozen for numerical Jacobians)."""
        return [m.usable_mask for m in self.models]

    def _build_response(self, calib_params: np.ndarray,
                        resol_params: np.ndarray) -> Response:
        """One shared binning + energy-to-channel response matrix per evaluation."""
        return Response.build(
            calib_params, resol_params, self.channel_max, self.fit_channel_lo,
            self.fit_channel_hi, self.last_channel)

    def _cached_projection(self, calib_params: np.ndarray,
                           resol_params: np.ndarray,
                           ) -> tuple[Response, list]:
        """Response and per-dataset projections, memoized on calib/resol.

        The derivative-free optimizers differentiate by probing one
        parameter at a time, so consecutive evaluations share
        ``(calib_params, resol_params)`` whenever only scale parameters
        move; the response matrix build and the projection (the dominant
        per-evaluation cost) are then reused instead of rebuilt per probe.
        """
        key = (calib_params.tobytes(), resol_params.tobytes())
        if key != self._proj_cache_key:
            resp = self._build_response(calib_params, resol_params)
            self._proj_cache = (resp, resp.project_many(
                [m.sim for m in self.models]))
            self._proj_cache_key = key
        return self._proj_cache

    def _per_dataset_arrays(self, calib_params: np.ndarray,
                            resol_params: np.ndarray,
                            mask_list: list[np.ndarray] | None = None,
                            ) -> list[DatasetArrays] | None:
        resp, projections = self._cached_projection(calib_params, resol_params)
        arrays_list = []
        for i, m in enumerate(self.models):
            mask = None if mask_list is None else mask_list[i]
            arrays = m.dataset_arrays(
                resp, mask=mask, projection=projections[i])
            if len(arrays.data_counts) < m.min_usable_bins:
                return None
            arrays_list.append(arrays)
        return arrays_list

    def _per_dataset_pulls(self, q, arrays_list) -> list[np.ndarray]:
        """Per-dataset chi-square pull vectors at ``q`` for the given arrays.

        The single per-dataset loop shared by :meth:`residuals` and
        :meth:`evaluate`, so the minimized objective and the residual
        vector cannot drift apart.
        """
        return [
            self.models[i].dataset_fit(arrays, q[self.param_space.scale(i)])[3]
            for i, arrays in enumerate(arrays_list)
        ]

    def residuals(self, q, mask_list=None) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if mask_list is not None:
            sizes = [int(np.sum(m)) for m in mask_list]
        else:
            sizes = [m.usable_bins for m in self.models]
        gate = self._gate(q)
        if gate is None:
            return np.full(int(sum(sizes)), np.nan)
        calib_params, resol_params = gate
        arrays_list = self._per_dataset_arrays(calib_params, resol_params,
                                               mask_list)
        if arrays_list is None:
            return np.full(int(sum(sizes)), np.nan)
        return np.concatenate(self._per_dataset_pulls(q, arrays_list))

    def evaluate(self, q) -> float:
        """Chi-square only; infeasible or under-covered states are inf."""
        gate = self._gate(q)
        if gate is None:
            return np.inf
        calib_params, resol_params = gate
        q = np.asarray(q, dtype=float)
        arrays_list = self._per_dataset_arrays(calib_params, resol_params)
        if arrays_list is None:
            return np.inf
        total = 0.0
        for pulls in self._per_dataset_pulls(q, arrays_list):
            total += float(pulls @ pulls)
        return total if np.isfinite(total) else np.inf

    def detail(self, q) -> FitDetail:
        """Full diagnostics; degenerate states yield chi2=inf and valid=False."""
        q = np.asarray(q, dtype=float)
        base = dict(scale_params=np.asarray(q[self.param_space.scale_block],
                                            dtype=float),
                    channel_max=self.channel_max)

        gate = self._gate(q)
        if gate is None:
            return FitDetail(datasets=[], chi2=np.inf, ndof=0,
                             **base, valid=False)
        calib_params, resol_params = gate
        resp = self._build_response(calib_params, resol_params)

        entries = []
        for i, m in enumerate(self.models):
            entries.append(m.dataset_detail(self.labels[i], resp,
                                            q[self.param_space.scale(i)]))
        if any(ds.n_bins < m.min_usable_bins
               for ds, m in zip(entries, self.models)):
            return FitDetail(datasets=[], chi2=np.inf, ndof=0,
                             **base, valid=False)

        chi2 = float(sum(ds.chi2 for ds in entries))
        ndof = int(sum(ds.n_bins for ds in entries)) - self.param_space.size
        return FitDetail(datasets=entries, chi2=chi2, ndof=ndof,
                         **base, valid=True)
