"""Per-dataset forward model on the shared deposition-to-channel response.

Each dataset keeps its raw channel counts/uncertainties on a fixed channel
range; the energy-deposition axis is only the calibration image of those
channels, used to place the simulation before folding.  The
resolution-folded per-channel model comes from the shared :class:`Response`
that :class:`GlobalFitModel` builds once per chi-square evaluation and
reuses across datasets.

The chi-square denominator combines three terms in quadrature:

    var = max(stat^2, 1) + (syst_frac * data)^2 + (s * mc_sigma)^2,

with the floor on the statistical term only: the data histogram's
per-bin statistical errors (the file's sumw2 sqrt) floored at 1, a
fractional systematic proportional to the data counts (squared, i.e.
|data|, so background-subtracted negative bins are handled
symmetrically), and the Monte Carlo statistical uncertainty of the
scaled model prediction (``scale`` times the per-bin MC sigma of the
folded MC spectrum, propagated exactly from the simulation's
per-source-bin variance through the rebin and response matrix, see
:class:`kc761calib.folding.McProjection`).  All window bins
participate: the floor gives empty bins of unsubtracted spectra an
approximate Poisson error of 1, while background-subtracted bins keep
their (possibly smaller) subtraction error above the floor -- their
fluctuation is already carried by the bin error.
"""

from __future__ import annotations

import numba
import numpy as np

from .folding import Response, McProjection
from .scaling import scale_model
from .types import DatasetArrays, DatasetDetail

DEFAULT_SYST_FRAC = 0.10


@numba.njit(inline="always", cache=True)
def data_display_sigma(data_counts, stat_uncertainties, syst_frac):
    """Data-band per-bin sigma: statistical + fractional systematic.

    ``var = max(stat_uncertainties^2, 1) + (syst_frac * data_counts)^2``
    (the square makes the fractional term proportional to |data_counts|;
    the floor matches :func:`fit_variance`).  Used for the displayed
    data band only; the fit's chi-square denominator is
    :func:`fit_variance`, which reuses the same two data-side terms plus
    the scaled model MC term.
    """
    var = np.maximum(stat_uncertainties**2, 1.0)
    var += (syst_frac * data_counts) ** 2
    return np.sqrt(var)


@numba.njit(inline="always", cache=True)
def fit_variance(stat_uncertainties, data_counts, model_mc_uncertainties,
                 syst_frac):
    """Per-bin variance of the chi-square denominator.

    ``var = max(stat_uncertainties^2, 1) + (syst_frac * data_counts)^2 +
    model_mc_uncertainties^2`` -- the single definition of the chi-square
    denominator's variance: the data histogram's per-bin statistical
    errors floored at 1 (empty bins of unsubtracted spectra get an
    approximate Poisson error; background-subtracted bins keep their
    possibly smaller subtraction error above the floor), a fractional
    systematic on the data (the square takes |data_counts|, so negative
    background-subtracted bins are handled symmetrically), and the
    scaled model prediction's Monte Carlo statistical uncertainty
    (``model_mc_uncertainties = scale * per-bin MC sigma``).
    """
    var = np.maximum(stat_uncertainties ** 2, 1.0)
    var += (syst_frac * data_counts) ** 2
    var += model_mc_uncertainties ** 2
    return var


class FitModel:
    """One dataset on a fixed channel range ``[channel_low, channel_high]``
    (0-based, inclusive).  The data counts/uncertainties are the raw channel
    values and never change with the calibration; the Monte Carlo spectrum
    is rebinned onto the energy-deposition bins and folded through the
    shared response, so the folded model compares directly to the data.
    """

    def __init__(self, data, mc_spectrum, channel_low: int,
                 channel_high: int,
                 syst_frac: float = DEFAULT_SYST_FRAC, *,
                 init_response=None):
        self.data = data
        self.mc_spectrum = mc_spectrum
        self.channel_low = int(channel_low)
        self.channel_high = int(channel_high)
        if not (0 <= self.channel_low <= self.channel_high < data.n_bins):
            raise ValueError(
                f"channel range [{self.channel_low}, {self.channel_high}] must "
                f"satisfy 0 <= channel_low <= channel_high < {data.n_bins}")
        if data.uncertainties is None:
            raise ValueError(
                "data spectrum must carry per-bin uncertainties")
        if init_response is None:
            raise ValueError("FitModel requires the shared init_response "
                             "(built by GlobalFitModel)")
        self.syst_frac = float(syst_frac)

        channel_slice = slice(self.channel_low, self.channel_high + 1)
        self.data_counts = data.counts[channel_slice]
        self.stat_uncertainties = data.uncertainties[channel_slice]
        # All window bins participate in the fit (the variance floor
        # handles empty bins); the positive-stat count is kept only for
        # the coverage gate (degenerate-window detection).
        self.positive_stat_bins = int(np.sum(self.stat_uncertainties > 0))
        self.channel_centers = np.arange(self.channel_low, self.channel_high + 1,
                                         dtype=np.float64)
        self.min_positive_stat_bins = max(
            10, int(0.1 * (self.channel_high - self.channel_low + 1)))

        self.init_response = init_response
        self.initial_scale = self._initial_scale()

    # --- data / model assembly ---

    def dataset_arrays(self, resp: Response,
                       projection: McProjection | None = None) -> DatasetArrays:
        """Assemble the per-dataset data/model/uncertainty arrays on the fit window.

        All window bins participate (empty bins are handled by the
        variance floor, see :func:`fit_variance`).  ``projection``
        optionally supplies the precomputed folded MC counts and their MC
        variances (from ``Response.project``/``project_many``); when
        ``None`` it is computed here.  The returned
        ``data_display_uncertainties`` are the data-band sigma (stat +
        syst·data) only; the scale-dependent MC term is combined in by the
        callers once the scale curve is known.
        """
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        if projection is None:
            projection = resp.project(self.mc_spectrum)
        model_counts = projection.counts[bin_slice]
        mc_uncertainties = np.sqrt(projection.variances[bin_slice])
        data_counts = self.data_counts
        return DatasetArrays(
            data_counts=data_counts,
            stat_uncertainties=self.stat_uncertainties,
            data_display_uncertainties=data_display_sigma(
                data_counts, self.stat_uncertainties, self.syst_frac),
            mc_uncertainties=mc_uncertainties,
            model_counts=model_counts,
            bin_centers=resp.binning.energy_centers[bin_slice],
            channel_centers=self.channel_centers,
        )

    def raw_mc_with_uncertainties_on_bins(
            self, resp: Response, projection: McProjection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebinned MC counts and their MC sigmas, prior to folding.

        Sliced from ``projection`` (computed once per evaluation), covering
        the full selected channel range (not the usable-bin mask), so the
        two arrays pair directly for the raw-MC spectrum and its
        uncertainty bars.
        """
        bin_slice = resp.binning.channel_slice(
            self.channel_low, self.channel_high)
        return (projection.rebinned[bin_slice],
                np.sqrt(projection.rebinned_variances[bin_slice]))

    def dataset_fit(self, arrays: DatasetArrays,
                    scale_params: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        """Scaled prediction, model MC uncertainty, fit sigma, pulls.

        The single source of truth for the per-dataset fit evaluation: the
        model prediction ``s(ch) * m(ch)`` with the per-bin scale curve
        ``s(ch) = scale_model(scale_params, ch, channel_low, channel_high)``
        at each channel center, its Monte Carlo statistical uncertainty
        ``s(ch) * mc_uncertainties`` (the band around the best-fit model),
        the chi-square denominator sigma
        ``sqrt(max(stat^2, 1) + (syst*data)^2 + mc_unc^2)``, and the pulls
        ``(data_counts - prediction) / fit_sigma``.
        """
        scale_curve = scale_model(scale_params, arrays.channel_centers,
                                  self.channel_low, self.channel_high)
        prediction = scale_curve * arrays.model_counts
        model_mc_uncertainties = scale_curve * arrays.mc_uncertainties
        var = fit_variance(arrays.stat_uncertainties, arrays.data_counts,
                           model_mc_uncertainties, self.syst_frac)
        fit_sigma = np.sqrt(var)
        residuals = (arrays.data_counts - prediction) / fit_sigma
        return prediction, model_mc_uncertainties, fit_sigma, residuals

    def dataset_detail(self, label: str, resp: Response,
                       scale_params: np.ndarray,
                       projection: McProjection | None = None) -> DatasetDetail:
        """Package one dataset's pulls into plot/report diagnostics.

        The channel window is fixed per dataset; the prediction, its Monte
        Carlo statistical uncertainty, the chi-square denominator sigma
        and the pulls come from :meth:`dataset_fit`.  All fields cover the
        full selected channel range (``bin_edges`` binning).
        ``projection`` optionally
        supplies the precomputed folded MC spectrum (and its MC variances)
        so callers with a cached projection (e.g.
        :meth:`kc761calib.globalfit.GlobalFitModel.detail`) skip the
        rebuild.
        """
        if projection is None:
            projection = resp.project(self.mc_spectrum)
        arrays = self.dataset_arrays(resp, projection=projection)
        prediction, model_mc_uncertainties, fit_sigma, residuals = (
            self.dataset_fit(arrays, scale_params))
        raw, raw_unc = self.raw_mc_with_uncertainties_on_bins(
            resp, projection)
        edge_slice = resp.binning.channel_edge_slice(self.channel_low,
                                                     self.channel_high)
        return DatasetDetail(
            label=label,
            channel_low=self.channel_low,
            channel_high=self.channel_high,
            bin_centers=arrays.bin_centers,
            data_counts=arrays.data_counts,
            data_display_uncertainties=arrays.data_display_uncertainties,
            mc_uncertainties=arrays.mc_uncertainties,
            model_mc_uncertainties=model_mc_uncertainties,
            fit_sigma=fit_sigma,
            model_prediction=prediction,
            raw_mc_counts=raw,
            raw_mc_uncertainties=raw_unc,
            scale_params=np.asarray(scale_params, dtype=float),
            chi2=float(residuals @ residuals),
            n_bins=int(len(arrays.data_counts)),
            bin_edges=resp.binning.energy_edges[edge_slice],
        )

    # --- validity ---

    def is_valid(self, calib_params) -> bool:
        """Whether the fixed channel range has enough positive-stat bins.

        ``calib_params`` is accepted for ``GlobalFitModel.is_valid``
        compatibility but is not needed (bin coverage is
        calibration-independent).
        """
        return self.positive_stat_bins >= self.min_positive_stat_bins

    # --- initialization helpers ---

    def _initial_scale(self) -> float:
        """Overall normalization estimate with the full per-bin variance.

        The variance weights use the fit variance (the data histogram's
        statistical errors, the data-side fractional systematic and the
        scaled MC terms; regular at scale -> 0 because the variance floor
        keeps the stat term >= 1), solved as a fixed point: starting
        from the data-only weights, each iteration
        re-weights with the current scale and damp-averages the new estimate
        (the map is not guaranteed monotone).  The fit itself refines this
        starting value, so a few iterations are ample.
        """
        arrays = self.dataset_arrays(self.init_response)
        data_var = arrays.data_display_uncertainties**2
        counts = arrays.data_counts
        model = arrays.model_counts
        model_norm = float(np.sum(model * model / data_var))
        if model_norm <= 0:
            raise ValueError("cannot estimate the initial scale: zero model "
                             "normalization in the usable bins")

        def estimate(scale):
            var = fit_variance(arrays.stat_uncertainties,
                               arrays.data_counts,
                               scale * arrays.mc_uncertainties,
                               self.syst_frac)
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
