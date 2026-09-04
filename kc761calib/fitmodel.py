"""Per-dataset forward model on the shared energy-to-channel response.

Each dataset keeps its raw channel counts/errors on a fixed channel range;
the true-energy axis is only the calibration image of those channels, used
to place the simulation before folding.  The resolution-smeared per-channel
model comes from the shared :class:`Response` that :class:`GlobalFitModel`
builds once per chi-square evaluation and reuses across datasets.

The per-bin uncertainty combined in the chi-square denominator has three
terms, added in quadrature: the data's statistical error, a fractional
systematic proportional to the data counts, and the Monte Carlo statistical
error of the model prediction ``scale * m`` -- i.e. ``scale`` times the
per-bin MC sigma of the smeared sim, propagated exactly from the
simulation's per-source-bin variance through the rebin and response matrix
(see :class:`kc761calib.folding.SimProjection`).
"""

from __future__ import annotations

import numba
import numpy as np

from .folding import Response, SimProjection
from .scaling import scale_model
from .types import DatasetArrays, DatasetDetail

DEFAULT_SYS_FRAC = 0.10


@numba.njit(inline="always", cache=True)
def error_model(data_counts, stat_errors, sys_frac):
    """Data-side per-bin sigma: statistical + fractional systematic.

    ``var = stat_errors^2 + (sys_frac * data_counts)^2``, bounded below at 1.
    """
    var = stat_errors**2 + (sys_frac * data_counts)**2
    return np.sqrt(np.maximum(var, 1.0))


@numba.njit(inline="always", cache=True)
def combined_variance(data_errors, mc_errors, scale):
    """Total per-bin variance: data-side plus the scaled MC statistical term.

    ``data_errors^2 + (scale * mc_errors)^2`` -- the single definition of
    the chi-square denominator's variance, shared by the fit (whose per-bin
    sigma is its sqrt, bounded below at 1) and by the initial-scale weights.
    """
    return data_errors**2 + (scale * mc_errors)**2


class FitModel:
    """One dataset binned as channels; the smeared model lives in channel space.

    The fit window is a fixed channel range ``[channel_low, channel_high]``
    (0-based, inclusive).  The data counts/errors are the raw channel values
    and never change with the calibration.  Each evaluation, the simulation
    is rebinned onto the true-energy bins (the calibration image of the
    channels) and folded through the shared energy-to-channel response, so
    the smeared model is predicted per channel bin and compares directly to
    the data; the bin energy positions (calibration image of the channel
    centers) are recomputed for display only.
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
        self.channel_centers = np.arange(self.channel_low, self.channel_high + 1,
                                         dtype=np.float64)
        self.min_usable_bins = max(
            10, int(0.1 * (self.channel_high - self.channel_low + 1)))

        self.init_response = init_response
        self.initial_scale = self._initial_scale()

    # ----- data / model assembly -----------------------------------------

    def dataset_arrays(self, resp: Response,
                       mask: np.ndarray | None = None,
                       projection: SimProjection | None = None) -> DatasetArrays:
        """Assemble the per-dataset data/model/error arrays on the usable bins.

        ``mask`` defaults to the fixed ``usable_mask``; an explicit mask freezes
        bin selection, as required when differencing residuals numerically.
        ``projection`` optionally supplies the precomputed smeared sim counts
        and their MC variances (from ``Response.project``/``project_many``);
        when ``None`` it is computed here.  The returned ``data_errors`` are
        the data-side sigma (stat + sys) only; the scale-dependent MC term is
        combined in by the callers once the scale curve is known.
        """
        if mask is None:
            mask = self.usable_mask
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        if projection is None:
            projection = resp.project(self.sim)
        model_counts = projection.counts[bin_slice]
        mc_errors = np.sqrt(projection.variances[bin_slice])
        data_counts = self.data_counts[mask]
        return DatasetArrays(
            data_counts=data_counts,
            data_errors=error_model(data_counts, self.data_errors[mask],
                                    self.sys_frac),
            mc_errors=mc_errors[mask],
            model_counts=model_counts[mask],
            bin_centers=resp.binning.energy_centers[bin_slice][mask],
            channel_centers=self.channel_centers[mask],
        )

    def unsmeared_sim_with_errors_on_bins(
            self, resp: Response, projection: SimProjection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebinned sim counts and their MC sigmas, prior to folding.

        Sliced from ``projection`` (computed once per evaluation), covering
        the full selected channel range (not the usable-bin mask), so the
        two arrays pair directly for the raw-sim spectrum and its error
        bars.
        """
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        return (projection.rebinned[bin_slice],
                np.sqrt(projection.rebinned_variances[bin_slice]))

    def dataset_fit(self, arrays: DatasetArrays,
                    scale_params: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        """Scaled prediction, model error, combined sigma and pulls of one dataset.

        The single source of truth for the per-dataset fit evaluation: the
        model prediction ``s(ch) * m(ch)`` with the per-bin scale curve
        ``s(ch) = scale_model(scale_params, ch, channel_low, channel_high)``
        at each channel bin center, its Monte Carlo statistical error
        ``s(ch) * mc_errors`` (the band around the best-fit model), the
        combined per-bin sigma
        ``sqrt(data_errors^2 + (s(ch) * mc_errors)^2)`` (data statistical +
        systematic plus the model's MC statistical error), and the pulls
        ``(data_counts - prediction) / sigma``.
        """
        scale_curve = scale_model(scale_params, arrays.channel_centers,
                                  self.channel_low, self.channel_high)
        prediction = scale_curve * arrays.model_counts
        model_errors = scale_curve * arrays.mc_errors
        sigma = np.sqrt(np.maximum(
            combined_variance(arrays.data_errors, arrays.mc_errors,
                              scale_curve),
            1.0))
        residuals = (arrays.data_counts - prediction) / sigma
        return prediction, model_errors, sigma, residuals

    def dataset_detail(self, label: str, resp: Response,
                       scale_params: np.ndarray) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics.

        The channel window is fixed per dataset; the prediction, its Monte
        Carlo statistical error, the combined per-bin sigma and the
        chi-square pulls come from :meth:`dataset_fit`.  The
        ``unsmeared_sim`` and ``unsmeared_sim_errors`` fields remain
        unscaled and cover the full selected channel range (including bins
        excluded by the usable-bin mask).
        """
        projection = resp.project(self.sim)
        arrays = self.dataset_arrays(resp, projection=projection)
        prediction, model_errors, sigma, residuals = self.dataset_fit(
            arrays, scale_params)
        unsmeared, unsmeared_errors = self.unsmeared_sim_with_errors_on_bins(
            resp, projection)
        edge_slice = resp.binning.channel_edge_slice(self.channel_low,
                                                     self.channel_high)
        return DatasetDetail(
            label=label,
            channel_low=self.channel_low,
            channel_high=self.channel_high,
            bin_centers=arrays.bin_centers,
            data_counts=arrays.data_counts,
            data_errors=arrays.data_errors,
            mc_errors=arrays.mc_errors,
            model_errors=model_errors,
            combined_errors=sigma,
            model_prediction=prediction,
            unsmeared_sim=unsmeared,
            unsmeared_sim_errors=unsmeared_errors,
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
        """Overall normalization estimate with the full per-bin variance.

        The variance weights include the Monte Carlo term
        ``(scale * mc_errors)^2``, which depends on the scale being
        estimated, so the weighted least-squares estimate is solved as a
        fixed point: starting from the data-only weights, each iteration
        re-weights with the current scale and damp-averages the new estimate
        (the map is not guaranteed monotone).  The fit itself refines this
        starting value, so a few iterations are ample.
        """
        arrays = self.dataset_arrays(self.init_response)
        if arrays.data_counts.size == 0:
            raise ValueError("cannot estimate the initial scale: no usable bins "
                             "(no positive statistical error)")
        data_var = arrays.data_errors**2
        counts = arrays.data_counts
        model = arrays.model_counts
        model_norm = float(np.sum(model * model / data_var))
        if model_norm <= 0:
            raise ValueError("cannot estimate the initial scale: zero model "
                             "normalization in the usable bins")

        def estimate(scale):
            var = combined_variance(arrays.data_errors, arrays.mc_errors,
                                    scale)
            num = float(np.sum(counts * model / var))
            den = float(np.sum(model * model / var))
            return num / den

        scale = estimate(0.0)
        for _ in range(20):
            new = estimate(scale)
            if not np.isfinite(new) or new <= 0.0:
                break
            if abs(new - scale) <= 1e-12 * max(abs(scale), 1.0):
                scale = new
                break
            scale = 0.5 * (scale + new)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"cannot estimate the initial scale: got non-positive or "
                f"non-finite value {scale}")
        return scale
