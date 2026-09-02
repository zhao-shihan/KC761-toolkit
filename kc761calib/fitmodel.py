"""Per-dataset forward model on the shared response binning.

Each dataset keeps its raw channel counts/errors on a fixed channel range;
energy is a relabeling of the channels via the calibration.  The
resolution-smeared model comes from the shared :class:`Response` that
:class:`GlobalFitModel` builds once per chi-square evaluation and reuses
across datasets.
"""

from __future__ import annotations

import numpy as np

from .folding import Response
from .scaling import scale_model
from .types import DatasetArrays, DatasetDetail

DEFAULT_SYS_FRAC = 0.05


class FitModel:
    """One dataset binned as channels; energy is a relabeling via the calibration.

    The fit window is a fixed channel range ``[channel_low, channel_high]``
    (0-based, inclusive).  Each channel bin maps to one energy bin with edges
    ``E(channel_edges)``, so the data counts/errors are the raw channel values
    and never change with the calibration; only the bin energy positions and
    the smeared model are recomputed each evaluation, the latter from the
    shared response.
    """

    def __init__(self, data, sim, channel_low: int, channel_high: int,
                 sys_frac: float = DEFAULT_SYS_FRAC, *,
                 init_response=None):
        self.data = data
        self.sim = sim
        self.channel_low = int(channel_low)
        self.channel_high = int(channel_high)
        if not (0 <= self.channel_low <= self.channel_high < data.n_bins):
            raise ValueError(
                f"channel range [{self.channel_low}, {self.channel_high}] must "
                f"satisfy 0 <= channel_low <= channel_high < {data.n_bins}")
        if data.errors is None:
            raise ValueError("data spectrum must carry per-bin errors")
        if init_response is None:
            raise ValueError("FitModel requires the shared init_response "
                             "(built by GlobalFitModel)")
        self.sys_frac = float(sys_frac)

        channel_slice = slice(self.channel_low, self.channel_high + 1)
        self.data_counts = data.counts[channel_slice]
        self.data_errors = data.errors[channel_slice]
        self.usable_mask = self.data_errors > 0
        self.min_usable_bins = max(
            10, int(0.1 * (self.channel_high - self.channel_low + 1)))

        self.init_response = init_response
        self.initial_scale = self._initial_scale()

    # ----- data / model assembly -----------------------------------------

    def error_model(self, data_counts, stat_errors) -> np.ndarray:
        """Total per-bin sigma: statistical + fractional systematic."""
        var = stat_errors**2 + (self.sys_frac * data_counts)**2
        return np.sqrt(np.maximum(var, 1.0))  # bound min error to 1

    def scale_refs(self, resp: Response) -> tuple[float, float]:
        """(lo, hi) scale reference energies at the current calibration.

        The linear scale is anchored at the fit-window lower/upper bin
        centers evaluated with the calibration the response was built
        with, so the anchors move with the fit window as the calibration
        changes during the fit; ``s0`` and ``s1`` are the scale values there.
        """
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        centers = resp.binning.energy_centers[bin_slice]
        return float(centers[0]), float(centers[-1])

    def dataset_arrays(self, resp: Response,
                       mask: np.ndarray | None = None,
                       smeared: np.ndarray | None = None) -> DatasetArrays:
        """Assemble the per-dataset data/model/weight arrays on the usable bins.

        ``mask`` defaults to the fixed ``usable_mask``; an explicit mask freezes
        bin selection, as required when differencing residuals numerically.
        ``smeared`` optionally supplies the precomputed full-binning smeared sim
        (from ``Response.smeared_many``); when ``None`` it is computed here.
        """
        if mask is None:
            mask = self.usable_mask
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        scale_lo, scale_hi = self.scale_refs(resp)
        if smeared is None:
            smeared = resp.smeared(self.sim)
        model_counts = smeared[bin_slice]
        total_errors = self.error_model(self.data_counts[mask],
                                        self.data_errors[mask])
        return DatasetArrays(
            data_counts=self.data_counts[mask],
            total_errors=total_errors,
            model_counts=model_counts[mask],
            bin_centers=resp.binning.energy_centers[bin_slice][mask],
            scale_lo=scale_lo,
            scale_hi=scale_hi,
        )

    def unsmeared_sim_on_bins(self, resp: Response) -> np.ndarray:
        """Rebinned sim counts on this dataset's bins, prior to folding with
        the response matrix."""
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        return resp.rebinned(self.sim)[bin_slice]

    def dataset_detail(self, label: str, resp: Response,
                       scale_params: np.ndarray | list[float]) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics.

        The model prediction is ``s(E) * m(E)`` with the per-bin scale curve
        ``s(E) = scale_model(scale_params, E, scale_lo, scale_hi)`` evaluated
        at each bin center; ``scale_lo``/``scale_hi`` are the fit-window
        lower/upper bin centers at the current calibration, so the anchors
        track the fit window.  The ``unsmeared_sim`` field remains unscaled.
        """
        arrays = self.dataset_arrays(resp)
        scale_curve = scale_model(scale_params, arrays.bin_centers,
                                  arrays.scale_lo, arrays.scale_hi)
        prediction = scale_curve * arrays.model_counts
        residuals = (arrays.data_counts - prediction) / arrays.total_errors
        edge_slice = resp.binning.channel_edge_slice(self.channel_low,
                                                     self.channel_high)
        return DatasetDetail(
            label=label,
            channel_low=self.channel_low,
            channel_high=self.channel_high,
            scale_lo=arrays.scale_lo,
            scale_hi=arrays.scale_hi,
            bin_centers=arrays.bin_centers,
            data_counts=arrays.data_counts,
            total_errors=arrays.total_errors,
            model_prediction=prediction,
            unsmeared_sim=self.unsmeared_sim_on_bins(resp),
            scale_params=np.asarray(scale_params, dtype=float),
            chi2=float(residuals @ residuals),
            n_bins=int(len(arrays.data_counts)),
            bin_edges=resp.binning.energy_edges[edge_slice],
        )

    # ----- validity --------------------------------------------------------

    @property
    def usable_bins(self) -> int:
        return int(self.usable_mask.sum())

    def is_valid(self, calib_params) -> bool:
        """Whether the fixed channel range has enough usable bins.

        ``calib_params`` is accepted for ``GlobalFitModel.is_valid`` compatibility
        but is not needed (bin selection is calibration-independent).
        """
        return self.usable_bins >= self.min_usable_bins

    # ----- initialization helpers ------------------------------------------

    def _initial_scale(self) -> float:
        arrays = self.dataset_arrays(self.init_response)
        if arrays.data_counts.size == 0:
            raise ValueError("cannot estimate the initial scale: no usable bins "
                             "(no positive statistical error)")
        variance = arrays.total_errors**2
        model_norm = float(np.sum(arrays.model_counts * arrays.model_counts
                                  / variance))
        if model_norm <= 0:
            raise ValueError("cannot estimate the initial scale: zero model "
                             "normalization in the usable bins")
        scale_estimate = float(
            np.sum(arrays.data_counts * arrays.model_counts / variance)
            / model_norm)
        if not np.isfinite(scale_estimate) or scale_estimate <= 0.0:
            raise ValueError(
                f"cannot estimate the initial scale: got non-positive or "
                f"non-finite value {scale_estimate}")
        return scale_estimate
