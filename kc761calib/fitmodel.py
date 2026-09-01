"""Per-dataset forward model on the shared convolution grid.

Each dataset keeps its raw channel counts/errors on a fixed channel range;
energy is a relabeling of the channels via the calibration.  The
resolution-smeared model comes from the shared :class:`Convolution` that
:class:`GlobalFitModel` builds once per chi-square evaluation and reuses
across datasets.
"""

from __future__ import annotations

import numpy as np

from .convolution import Convolution
from .response import INIT_CALIB, calib_model
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
    shared convolution.
    """

    def __init__(self, data, sim, channel_low: int, channel_high: int,
                 sys_frac: float = DEFAULT_SYS_FRAC, *, init_calib=None,
                 init_convolution=None):
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
        if init_convolution is None:
            raise ValueError("FitModel requires the shared init_convolution "
                             "(built by GlobalFitModel)")
        self.sys_frac = float(sys_frac)

        self._ch_edges = data.edges
        self.channel_max = float(data.edges[-1])

        self.channel_slice = slice(self.channel_low, self.channel_high + 1)
        self.edge_slice = slice(self.channel_low, self.channel_high + 2)
        self.data_counts = data.counts[self.channel_slice]
        self.data_errors = data.errors[self.channel_slice]
        self.usable_mask = self.data_errors > 0
        self.min_usable_bins = max(
            10, int(0.1 * (self.channel_high - self.channel_low + 1)))

        self.init_calib = INIT_CALIB if init_calib is None else np.asarray(
            init_calib, dtype=float)
        self.init_convolution = init_convolution
        self.scale_lo, self.scale_hi = self._scale_refs()
        self.initial_scale = self._initial_scale()

    # ----- data / model assembly -----------------------------------------

    def error_model(self, data_counts, stat_errors) -> np.ndarray:
        """Total per-bin sigma: statistical + fractional systematic."""
        var = stat_errors**2 + (self.sys_frac * data_counts)**2
        return np.sqrt(np.maximum(var, 1.0))  # bound min error to 1

    def energy_edges(self, calib_params) -> np.ndarray:
        """Energy of each selected channel edge (length ``n_selected + 1``).

        Used for the fixed scale reference energies at initialization.
        """
        return calib_model(calib_params, self._ch_edges[self.edge_slice],
                           self.channel_max)

    def dataset_arrays(self, conv: Convolution,
                       mask: np.ndarray | None = None,
                       smeared: np.ndarray | None = None) -> DatasetArrays:
        """Assemble the per-dataset data/model/weight arrays on the usable bins.

        ``mask`` defaults to the fixed ``usable_mask``; an explicit mask freezes
        bin selection, as required when differencing residuals numerically.
        ``smeared`` optionally supplies the precomputed full-grid smeared sim
        (from ``Convolution.smeared_many``); when ``None`` it is computed here.
        """
        if mask is None:
            mask = self.usable_mask
        bin_slice = conv.grid.channel_slice(
            self.channel_low, self.channel_high)
        if smeared is None:
            smeared = conv.smeared(self.sim)
        model_counts = smeared[bin_slice]
        weights = self.error_model(self.data_counts[mask],
                                   self.data_errors[mask])
        return DatasetArrays(
            data_counts=self.data_counts[mask],
            weights=weights,
            model_counts=model_counts[mask],
            bin_centers=conv.grid.energy_centers[bin_slice][mask],
        )

    def unsmeared_sim_on_bins(self, conv: Convolution) -> np.ndarray:
        """Rebinned (unconvolved) sim counts on this dataset's bins."""
        bin_slice = conv.grid.channel_slice(
            self.channel_low, self.channel_high)
        return conv.rebinned(self.sim)[bin_slice]

    def dataset_detail(self, label: str, conv: Convolution,
                       scale_params: np.ndarray | list[float]) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics.

        The model prediction is ``s(E) * m(E)`` with the per-bin scale curve
        ``s(E) = scale_model(scale_params, E, scale_lo, scale_hi)`` evaluated at
        each bin center; the ``smeared_sim``/``unsmeared_sim`` fields remain
        unscaled.
        """
        arrays = self.dataset_arrays(conv)
        scale_curve = scale_model(scale_params, arrays.bin_centers,
                                  self.scale_lo, self.scale_hi)
        prediction = scale_curve * arrays.model_counts
        residuals = (arrays.data_counts - prediction) / arrays.weights
        edge_slice = conv.grid.channel_edge_slice(self.channel_low,
                                                  self.channel_high)
        return DatasetDetail(
            label=label,
            channel_low=self.channel_low,
            channel_high=self.channel_high,
            scale_lo=self.scale_lo,
            scale_hi=self.scale_hi,
            bin_centers=arrays.bin_centers,
            data_counts=arrays.data_counts,
            data_errors=arrays.weights,
            model_prediction=prediction,
            smeared_sim=arrays.model_counts,
            unsmeared_sim=self.unsmeared_sim_on_bins(conv),
            scale_params=np.asarray(scale_params, dtype=float),
            chi2=float(residuals @ residuals),
            n_bins=int(len(arrays.data_counts)),
            bin_edges=conv.grid.energy_edges[edge_slice],
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

    def _scale_refs(self) -> tuple[float, float]:
        """(lo, hi) energy bounds of the initial calibration's selected range.

        The linear scale is anchored at these two fixed reference energies (the
        initial fit-window lower/upper bin centers); the fit parameters ``s0``,
        ``s1`` are the scale values there.
        """
        edges = self.energy_edges(self.init_calib)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers[0], centers[-1]

    def _initial_scale(self) -> float:
        arrays = self.dataset_arrays(self.init_convolution)
        if arrays.data_counts.size == 0:
            raise ValueError("cannot estimate the initial scale: no usable bins "
                             "(no positive statistical error)")
        variance = arrays.weights**2
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
