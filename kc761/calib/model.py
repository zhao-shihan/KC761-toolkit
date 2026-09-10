"""Calibration forward model on the shared response (F-CAL-1/F-CAL-3/F-CAL-4).

Per dataset ``d`` the model is

    ``data_d ~= diag(scale_d(ch)) . (C_fit(q) @ mc_d)``

with the shared internal core ``q = (c0, k1, k2, k3, b0, b1, b2)`` and the
per-dataset quadratic-Bezier scale ``scale_d`` (F-CAL-2). The fit-time
``C_fit`` uses the fixed uniform deposition axis ``0..4096 keV / 4096 bins``
(the source-mode MC axis, D-33) so its geometry does not depend on the
parameters (D-79/D-101); the export matrix is rebuilt separately on the
channel-derived axis (D-101).

The chi-square uses the frozen F-CAL-1 weights

    ``var = max(stat, 1) + (syst_frac * data)**2 + MC``

where the Monte Carlo term is the folded prediction's MC variance,
``(scale_d * sqrt((C_fit**2) @ var_mc_d))**2`` (D-48/D-102). Value, Jacobian
and gradient are all derived analytically from this one expression
(F-CAL-4); no finite differences enter production. The gradient includes the
parameter dependence of the variance, so the optimizer's gradient is the exact
gradient of the objective it minimizes.

Limitations (documented in ``docs/derivations.md``): the non-strict
F-MODEL-5 clamp makes the objective only C0 at a clamp boundary, so the
quasi-Newton step may see a kink (D-104).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.calib.scaling import (
    N_SCALE,
    scale_bounds,
    scale_curve,
    scale_curve_grad,
)
from kc761.calib.types import DatasetSpec, FitSettings
from kc761.core._checks import as_float_array
from kc761.core.binning import ChannelGrid
from kc761.core.model import (
    InternalCalibration,
    energy_kev,
    internal_jacobian,
)
from kc761.core.response import (
    build_response_matrix,
    response_parameter_jacobian,
)
from kc761.errors import SolverError, ValidationError

N_CORE: Final = 7
"""Fit-basis core ``(c0, k1, k2, k3, b0, b1, b2)`` (F-MODEL-2)."""

FIXED_DEPOSITION_MAX_KEV: Final = 4096.0
FIXED_DEPOSITION_BINS: Final = 4096
"""Fixed source-mode deposition axis (D-33/D-101)."""

INIT_CALIB: Final[tuple[float, float, float, float]] = (-180.0, 1.5, 2.5, 3.5)
BOUNDS_CALIB: Final[tuple[tuple[float, float], ...]] = (
    (-300.0, -100.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
)
INIT_RESOL: Final[tuple[float, float, float]] = (2.0, 20.0, 40.0)
BOUNDS_RESOL: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 10.0),
    (0.0, 80.0),
    (0.0, 100.0),
)
"""Start values and bounds (F-CAL-3; specification reference to the rewrite).

The bounds keep ``E(ch)`` monotone (F-MODEL-3) and contain the resolution
parameters used by the pre-rewrite fits (docs/plan.md D-103).
"""


def fixed_deposition_edges_kev() -> NDArray[np.float64]:
    """The fixed uniform source-mode deposition axis (D-33/D-101)."""
    return np.linspace(0.0, FIXED_DEPOSITION_MAX_KEV, FIXED_DEPOSITION_BINS + 1)


def core_bounds() -> tuple[tuple[float, float], ...]:
    """Bounds of the internal core ``(c0, k1, k2, k3, b0, b1, b2)`` (F-CAL-3)."""
    return (*BOUNDS_CALIB, *BOUNDS_RESOL)


def initial_core() -> NDArray[np.float64]:
    """Start value of the internal core (F-CAL-3)."""
    return np.array([*INIT_CALIB, *INIT_RESOL], dtype=np.float64)


@dataclass(frozen=True)
class DatasetProjection:
    """One dataset's model quantities on its window (depends only on ``q``)."""

    channel_low: int
    channel_high: int
    channel_centers: NDArray[np.float64]
    model_counts: NDArray[np.float64]
    mc_variance: NDArray[np.float64]
    d_model: tuple[NDArray[np.float64], ...]
    d_mc_variance: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class _ResponseCache:
    core: NDArray[np.float64]
    matrix: sparse.csr_matrix
    jacobian: tuple[sparse.csr_matrix, ...]
    datasets: tuple[DatasetProjection, ...]


@dataclass(frozen=True)
class DatasetDetail:
    """Per-dataset diagnostics used by the report and the figure."""

    label: str
    channel_low: int
    channel_high: int
    channel_centers: NDArray[np.float64]
    data_counts: NDArray[np.float64]
    data_display_sigma: NDArray[np.float64]
    scale_curve: NDArray[np.float64]
    model_counts: NDArray[np.float64]
    model_mc_sigma: NDArray[np.float64]
    prediction: NDArray[np.float64]
    fit_sigma: NDArray[np.float64]
    residual: NDArray[np.float64]
    energy_centers_kev: NDArray[np.float64]
    energy_edges_kev: NDArray[np.float64]
    chi2: float


class CalibrationModel:
    """Global calibration objective over one or more datasets (F-CAL-1/4)."""

    def __init__(
        self,
        datasets: tuple[DatasetSpec, ...] | list[DatasetSpec],
        *,
        channel_max: float | None = None,
        settings: FitSettings | None = None,
    ) -> None:
        self.settings = settings if settings is not None else FitSettings()
        specs = tuple(datasets)
        if not specs:
            raise ValidationError("at least one dataset is required (F-CAL-1)")
        self.specs = specs

        n_channels = {spec.data.axis.n_bins for spec in specs}
        if len(n_channels) != 1:
            raise ValidationError(
                "all datasets must share the same acquisition channel count "
                f"(a global calibration is shared); got {sorted(n_channels)}"
            )
        self.n_channels = int(next(iter(n_channels)))
        inferred = float(self.n_channels - 1)
        self.channel_max = inferred if channel_max is None else float(channel_max)
        if not np.isfinite(self.channel_max) or self.channel_max <= 0.0:
            raise ValidationError(f"channel_max must be positive and finite, got {channel_max!r}")
        if self.channel_max < inferred:
            raise ValidationError(
                f"channel_max={self.channel_max!r} is below the last channel index {inferred}"
            )

        self.fixed_edges = fixed_deposition_edges_kev()
        self.data_counts: list[NDArray[np.float64]] = []
        self.stat_variance: list[NDArray[np.float64]] = []
        self.mc_counts: list[NDArray[np.float64]] = []
        self.mc_variance_source: list[NDArray[np.float64]] = []
        self.syst_frac: list[float] = []
        self.channel_low: list[int] = []
        self.channel_high: list[int] = []

        for spec in specs:
            self._add_dataset(spec)

        self._cache_key: bytes | None = None
        self._cache: _ResponseCache | None = None
        self.initial_scales: tuple[float, ...] = tuple(
            self._estimate_initial_scale(index) for index in range(len(specs))
        )
        self.bounds = self._build_bounds()
        self.x0 = self._build_start()
        self.n_bins = int(sum(spec.channel_high - spec.channel_low + 1 for spec in specs))
        self.n_free = N_CORE + N_SCALE * len(specs)
        self.dof = self.n_bins - self.n_free
        if self.dof < 1:
            raise ValidationError(
                f"dof = {self.dof} < 1: {self.n_bins} fitted bins cannot constrain "
                f"{self.n_free} free parameters (F-CAL-1)"
            )

    # ------------------------------------------------------------------ setup
    def _add_dataset(self, spec: DatasetSpec) -> None:
        data = spec.data
        mc = spec.mc
        if data.axis.unit != "channel":
            raise ValidationError(
                f"dataset {spec.label!r}: data spectrum must carry the channel axis, "
                f"got unit {data.axis.unit!r}"
            )
        if mc.axis.unit != "kev":
            raise ValidationError(
                f"dataset {spec.label!r}: MC spectrum must carry an energy axis in keV, "
                f"got unit {mc.axis.unit!r}"
            )
        low = int(spec.channel_low)
        high = int(spec.channel_high)
        if not 0 <= low <= high < data.axis.n_bins:
            raise ValidationError(
                f"dataset {spec.label!r}: window [{low}, {high}] outside "
                f"[0, {data.axis.n_bins - 1}]"
            )
        if not np.isfinite(spec.syst_frac) or spec.syst_frac < 0.0:
            raise ValidationError(
                f"dataset {spec.label!r}: syst_frac must be finite and >= 0, got {spec.syst_frac!r}"
            )
        if data.variances is None:
            raise ValidationError(
                f"dataset {spec.label!r}: data spectrum has no fSumw2 variance buffer"
            )
        values = as_float_array(f"{spec.label}.data.counts", data.values, ndim=1)
        variances = as_float_array(f"{spec.label}.data.variances", data.variances, ndim=1)
        if np.any(variances < 0.0):
            raise ValidationError(f"dataset {spec.label!r}: data variances must be non-negative")
        mc_values = as_float_array(f"{spec.label}.mc.counts", mc.values, ndim=1)
        if np.any(mc_values < 0.0):
            raise ValidationError(f"dataset {spec.label!r}: MC counts must be non-negative")
        if not np.array_equal(mc.axis.edges, self.fixed_edges):
            raise ValidationError(
                f"dataset {spec.label!r}: MC spectrum must use the fixed source-mode "
                f"deposition axis (D-33/D-101), got {mc.axis.n_bins} bins"
            )
        if mc.variances is None:
            mc_var = np.maximum(mc_values, 0.0)
        else:
            mc_var = as_float_array(f"{spec.label}.mc.variances", mc.variances, ndim=1)
            if np.any(mc_var < 0.0):
                raise ValidationError(f"dataset {spec.label!r}: MC variances must be non-negative")
        # Keep the window quantities the model actually fits: the scale curve
        # and the folded model are window-sized, so the data side must be too.
        self.data_counts.append(values[low : high + 1])
        self.stat_variance.append(np.maximum(variances[low : high + 1], 1.0))
        self.mc_counts.append(mc_values)
        self.mc_variance_source.append(mc_var)
        self.syst_frac.append(float(spec.syst_frac))
        self.channel_low.append(low)
        self.channel_high.append(high)

    def _build_bounds(self) -> list[tuple[float, float]]:
        bounds = list(core_bounds())
        for index in range(len(self.specs)):
            bounds.extend(
                scale_bounds(
                    self.initial_scales[index],
                    self.channel_low[index],
                    self.channel_high[index],
                )
            )
        return bounds

    def _build_start(self) -> NDArray[np.float64]:
        start = [initial_core()]
        for index in range(len(self.specs)):
            start.append(np.asarray(self._seed_scale(index), dtype=np.float64))
        return np.concatenate(start)

    def _seed_scale(self, index: int) -> tuple[float, float, float, float]:
        """Non-degenerate scale start ``(s0, s1, s2, s3)`` (F-CAL-3).

        A constant scale makes ``ds/ds0`` vanish identically, so the ``s0``
        control channel is unidentifiable at the start (and a gradient-based
        optimizer cannot leave the plateau). The seed is therefore the
        data/model ratio averaged over the lower, middle and upper thirds of
        the window (weighted by ``model**2 / var`` at the initial core), which
        starts the fit on a genuine parabola; ``s0`` starts at the midpoint.
        """
        cache = self._response_cache(initial_core())
        projection = cache.datasets[index]
        model = projection.model_counts
        data = self.data_counts[index]
        variance = self.stat_variance[index] + (self.syst_frac[index] * data) ** 2
        fallback = self.initial_scales[index]
        size = model.size
        edges = (0, size // 3, (2 * size) // 3, size)
        values: list[float] = []
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            window_model = model[start:stop]
            mask = window_model > 0.0
            if not mask.any():
                values.append(fallback)
                continue
            numerator = float(
                np.sum(data[start:stop][mask] * window_model[mask] / variance[start:stop][mask])
            )
            denominator = float(
                np.sum(window_model[mask] ** 2 / variance[start:stop][mask])
            )
            values.append(numerator / denominator if denominator > 0.0 else fallback)
        value_bounds = scale_bounds(fallback, self.channel_low[index], self.channel_high[index])[1]
        clamped = tuple(float(np.clip(value, *value_bounds)) for value in values)
        midpoint = 0.5 * (self.channel_low[index] + self.channel_high[index])
        return (midpoint, clamped[0], clamped[1], clamped[2])

    def _estimate_initial_scale(self, index: int) -> float:
        """Overall normalization ``sum(data) / sum(model)`` (F-CAL-3).

        Unweighted and immune to the empty-bin domination of a plain weighted
        moment; it sets the relative bounds of ``s1, s2, s3``. The scale shape
        (and ``s0``) is seeded separately by :meth:`_seed_scale`.
        """
        cache = self._response_cache(initial_core())
        projection = cache.datasets[index]
        normalization = float(np.sum(projection.model_counts))
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValidationError(
                f"dataset {self.specs[index].label!r}: cannot estimate the initial "
                "scale (zero model normalization in the fit window)"
            )
        scale = float(np.sum(self.data_counts[index])) / normalization
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValidationError(
                f"dataset {self.specs[index].label!r}: non-positive initial scale {scale!r}"
            )
        return scale

    # --------------------------------------------------------------- response
    def _active_deposition_columns(self) -> tuple[int, int]:
        """Contiguous slice of the fixed axis touched by any MC spectrum.

        The fit only ever contracts ``C_fit`` with ``mc`` and its variance, so
        columns outside the union support contribute exactly zero. Building the
        response on the contiguous hull of the support is therefore the exact
        sub-matrix of the full fixed-axis response (each deposition column's
        kernel and renormalization depend on that column alone), while keeping
        the deposition bin edges identical to the fixed axis (D-101).
        """
        active = np.zeros(FIXED_DEPOSITION_BINS, dtype=bool)
        for counts, variance in zip(self.mc_counts, self.mc_variance_source, strict=True):
            active |= (counts != 0.0) | (variance != 0.0)
        if not active.any():
            raise ValidationError("MC spectra have no non-zero deposition bin (D-101)")
        low = int(np.argmax(active))
        high = int(active.size - 1 - np.argmax(active[::-1]))
        return low, high

    def _response_cache(self, core: NDArray[np.float64]) -> _ResponseCache:
        q = as_float_array("core", core, ndim=1)
        if q.size != N_CORE:
            raise ValidationError(f"core must have {N_CORE} entries, got {q.size}")
        key = q.tobytes()
        if key == self._cache_key and self._cache is not None:
            return self._cache

        low, high = self._active_deposition_columns()
        deposition_edges = np.asarray(self.fixed_edges[low : high + 2], dtype=np.float64)
        mc_slices = [counts[low : high + 1] for counts in self.mc_counts]
        variance_slices = [variance[low : high + 1] for variance in self.mc_variance_source]

        calibration = InternalCalibration.from_array(q[:4])
        resol = np.asarray(q[4:], dtype=np.float64)
        grid = ChannelGrid(self.n_channels)
        response = build_response_matrix(
            deposition_edges,
            calibration,
            resol,
            channel_grid=grid,
            channel_max=self.channel_max,
        )
        reported_jacobian = response_parameter_jacobian(
            deposition_edges,
            calibration,
            resol,
            channel_grid=grid,
            channel_max=self.channel_max,
        )
        # Chain the reported-basis derivatives into the internal slope basis
        # through the constant F-MODEL-2 Jacobian M = d(reported)/d(internal).
        transform = internal_jacobian(channel_max=self.channel_max)
        internal_jacobian_matrices: list[sparse.csr_matrix] = []
        for internal_index in range(4):
            accumulator = reported_jacobian[0] * float(transform[0, internal_index])
            for reported_index in range(1, 4):
                weight = float(transform[reported_index, internal_index])
                if weight != 0.0:
                    accumulator = accumulator + reported_jacobian[reported_index] * weight
            internal_jacobian_matrices.append(accumulator.tocsr())
        internal_jacobian_matrices.extend(matrix.tocsr() for matrix in reported_jacobian[4:])

        matrix = response.matrix
        projections: list[DatasetProjection] = []
        for index in range(len(self.specs)):
            rows = slice(self.channel_low[index], self.channel_high[index] + 1)
            window = matrix[rows, :]
            mc = mc_slices[index]
            mc_var = variance_slices[index]
            model_counts = np.asarray(window @ mc, dtype=np.float64)
            mc_variance = np.asarray(window.power(2) @ mc_var, dtype=np.float64)
            d_model: list[NDArray[np.float64]] = []
            d_mc_variance: list[NDArray[np.float64]] = []
            for parameter in range(N_CORE):
                derivative = internal_jacobian_matrices[parameter][rows, :]
                d_model.append(np.asarray(derivative @ mc, dtype=np.float64))
                d_mc_variance.append(
                    2.0 * np.asarray(window.multiply(derivative) @ mc_var, dtype=np.float64)
                )
            projections.append(
                DatasetProjection(
                    channel_low=self.channel_low[index],
                    channel_high=self.channel_high[index],
                    channel_centers=np.arange(
                        self.channel_low[index], self.channel_high[index] + 1, dtype=np.float64
                    ),
                    model_counts=model_counts,
                    mc_variance=mc_variance,
                    d_model=tuple(d_model),
                    d_mc_variance=tuple(d_mc_variance),
                )
            )
        cache = _ResponseCache(
            core=q.copy(),
            matrix=matrix,
            jacobian=tuple(internal_jacobian_matrices),
            datasets=tuple(projections),
        )
        self._cache_key = key
        self._cache = cache
        return cache

    # ------------------------------------------------------------ evaluation
    def _check_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        values = as_float_array("theta", theta, ndim=1)
        if values.size != self.n_free:
            raise ValidationError(
                f"theta has {values.size} entries, the model expects {self.n_free}"
            )
        return values

    def _prediction(
        self, index: int, scale: NDArray[np.float64], projection: DatasetProjection
    ) -> NDArray[np.float64]:
        """F-CAL-1 prediction ``scale . (C_fit @ mc)`` of one dataset."""
        return scale * projection.model_counts

    def _fit_variance(
        self, index: int, scale: NDArray[np.float64], projection: DatasetProjection
    ) -> NDArray[np.float64]:
        """F-CAL-1 fit variance of one dataset (single source of the formula)."""
        return (
            self.stat_variance[index]
            + (self.syst_frac[index] * self.data_counts[index]) ** 2
            + scale * scale * projection.mc_variance
        )

    def _dataset_terms(
        self,
        index: int,
        theta: NDArray[np.float64],
        scale: NDArray[np.float64],
    ) -> tuple[DatasetProjection, NDArray[np.float64]]:
        cache = self._response_cache(theta[:N_CORE])
        projection = cache.datasets[index]
        prediction = self._prediction(index, scale, projection)
        variance = self._fit_variance(index, scale, projection)
        residual = (self.data_counts[index] - prediction) / np.sqrt(variance)
        return projection, residual

    def _scale_values_for(self, theta: NDArray[np.float64], index: int) -> NDArray[np.float64]:
        start = N_CORE + N_SCALE * index
        return scale_curve(
            theta[start : start + N_SCALE],
            np.arange(self.channel_low[index], self.channel_high[index] + 1, dtype=np.float64),
            self.channel_low[index],
            self.channel_high[index],
        )

    def _scale_values(self, theta: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        return [self._scale_values_for(theta, index) for index in range(len(self.specs))]

    def _residual_blocks(self, theta: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        scales = self._scale_values(theta)
        return [
            self._dataset_terms(index, theta, scales[index])[1] for index in range(len(self.specs))
        ]

    def evaluate(self, theta: NDArray[np.float64]) -> float:
        """Chi-square at ``theta`` (F-CAL-1)."""
        values = self._check_theta(theta)
        total = 0.0
        for residual in self._residual_blocks(values):
            total += float(residual @ residual)
        if not np.isfinite(total):
            raise SolverError("calibration objective is not finite")
        return total

    def residuals(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Concatenated per-bin residuals ``(data - prediction) / sigma``."""
        return np.concatenate(self._residual_blocks(self._check_theta(theta)))

    def jacobian(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prediction Jacobian ``d(prediction)/d(theta)`` (F-CAL-4).

        Shape ``(n_bins, n_free)``; the core columns chain ``dC/dq`` through
        the dataset scale and the scale columns multiply the folded model by
        the Bezier derivative (F-CAL-2).
        """
        values = self._check_theta(theta)
        jacobian = np.zeros((self.n_bins, self.n_free), dtype=np.float64)
        offset = 0
        for index in range(len(self.specs)):
            projection = self._response_cache(values[:N_CORE]).datasets[index]
            scale, scale_grad = scale_curve_grad(
                values[N_CORE + N_SCALE * index : N_CORE + N_SCALE * (index + 1)],
                projection.channel_centers,
                self.channel_low[index],
                self.channel_high[index],
            )
            size = self.channel_high[index] - self.channel_low[index] + 1
            block = slice(offset, offset + size)
            for parameter in range(N_CORE):
                jacobian[block, parameter] = scale * projection.d_model[parameter]
            for parameter in range(N_SCALE):
                jacobian[block, N_CORE + N_SCALE * index + parameter] = (
                    scale_grad[parameter] * projection.model_counts
                )
            offset += size
        return jacobian

    def variance(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Concatenated per-bin fit variance (F-CAL-1)."""
        values = self._check_theta(theta)
        scales = self._scale_values(values)
        variances: list[NDArray[np.float64]] = []
        for index in range(len(self.specs)):
            projection = self._response_cache(values[:N_CORE]).datasets[index]
            variances.append(self._fit_variance(index, scales[index], projection))
        return np.concatenate(variances)

    def variance_gradient(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the F-CAL-1 variance ``d v / d theta`` (F-CAL-4).

        ``dv/dq_k = scale**2 d(MC var)/dq_k`` and
        ``dv/ds_p = 2 scale (d scale/ds_p) MC var``. Consumed by the exact
        objective gradient and the trust-region residual Jacobian.
        """
        values = self._check_theta(theta)
        gradient = np.zeros((self.n_bins, self.n_free), dtype=np.float64)
        offset = 0
        for index in range(len(self.specs)):
            projection = self._response_cache(values[:N_CORE]).datasets[index]
            scale, scale_grad = scale_curve_grad(
                values[N_CORE + N_SCALE * index : N_CORE + N_SCALE * (index + 1)],
                projection.channel_centers,
                self.channel_low[index],
                self.channel_high[index],
            )
            size = self.channel_high[index] - self.channel_low[index] + 1
            block = slice(offset, offset + size)
            for parameter in range(N_CORE):
                gradient[block, parameter] = scale * scale * projection.d_mc_variance[parameter]
            for parameter in range(N_SCALE):
                gradient[block, N_CORE + N_SCALE * index + parameter] = (
                    2.0 * scale * scale_grad[parameter] * projection.mc_variance
                )
            offset += size
        return gradient

    def gradient(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Exact gradient of :meth:`evaluate` including the variance terms.

        With ``r = (data - p) / sigma`` and ``v = sigma**2``,
        ``d chi2 / d theta = -2 J^T ((data - p)/v) - (dv/dtheta)^T ((data-p)^2/v^2)``.
        The variance derivative (F-CAL-4) is ``dv/dq_k = scale**2 d(MC var)/dq_k``
        and ``dv/ds_p = 2 scale (d scale/ds_p) MC var``.
        """
        values = self._check_theta(theta)
        gradient = np.zeros(self.n_free, dtype=np.float64)
        for index in range(len(self.specs)):
            projection = self._response_cache(values[:N_CORE]).datasets[index]
            scale, scale_grad = scale_curve_grad(
                values[N_CORE + N_SCALE * index : N_CORE + N_SCALE * (index + 1)],
                projection.channel_centers,
                self.channel_low[index],
                self.channel_high[index],
            )
            data = self.data_counts[index]
            model = projection.model_counts
            prediction = self._prediction(index, scale, projection)
            variance = self._fit_variance(index, scale, projection)
            difference = data - prediction
            first = difference / variance
            second = difference * difference / (variance * variance)
            for parameter in range(N_CORE):
                d_prediction = scale * projection.d_model[parameter]
                d_variance = scale * scale * projection.d_mc_variance[parameter]
                gradient[parameter] += -2.0 * float(d_prediction @ first) - float(
                    d_variance @ second
                )
            for parameter in range(N_SCALE):
                d_scale = scale_grad[parameter]
                d_prediction = d_scale * model
                d_variance = 2.0 * scale * d_scale * projection.mc_variance
                gradient[N_CORE + N_SCALE * index + parameter] += -2.0 * float(
                    d_prediction @ first
                ) - float(d_variance @ second)
        return gradient

    # ------------------------------------------------------------- diagnostics
    def dataset_details(self, theta: NDArray[np.float64]) -> tuple[DatasetDetail, ...]:
        """Report/plot diagnostics with the fitted model at ``theta`` (F-CAL-4)."""
        values = self._check_theta(theta)
        details: list[DatasetDetail] = []
        calibration = InternalCalibration.from_array(values[:4])
        scales = self._scale_values(values)
        for index, spec in enumerate(self.specs):
            projection, residual = self._dataset_terms(index, values, scales[index])
            scale = scales[index]
            variance = self._fit_variance(index, scale, projection)
            energy_edges = energy_kev(
                np.arange(
                    self.channel_low[index] - 0.5,
                    self.channel_high[index] + 1.5,
                    dtype=np.float64,
                ),
                calibration,
                channel_max=self.channel_max,
            )
            details.append(
                DatasetDetail(
                    label=spec.label,
                    channel_low=self.channel_low[index],
                    channel_high=self.channel_high[index],
                    channel_centers=projection.channel_centers.copy(),
                    data_counts=self.data_counts[index].copy(),
                    data_display_sigma=np.sqrt(
                        self.stat_variance[index]
                        + (self.syst_frac[index] * self.data_counts[index]) ** 2
                    ),
                    scale_curve=scale.copy(),
                    model_counts=projection.model_counts.copy(),
                    model_mc_sigma=scale * np.sqrt(projection.mc_variance),
                    prediction=scale * projection.model_counts,
                    fit_sigma=np.sqrt(variance),
                    residual=residual.copy(),
                    energy_centers_kev=0.5 * (energy_edges[:-1] + energy_edges[1:]),
                    energy_edges_kev=energy_edges,
                    chi2=float(residual @ residual),
                )
            )
        return tuple(details)


__all__ = [
    "BOUNDS_CALIB",
    "BOUNDS_RESOL",
    "FIXED_DEPOSITION_BINS",
    "FIXED_DEPOSITION_MAX_KEV",
    "INIT_CALIB",
    "INIT_RESOL",
    "N_CORE",
    "CalibrationModel",
    "DatasetDetail",
    "DatasetProjection",
    "core_bounds",
    "fixed_deposition_edges_kev",
    "initial_core",
]
