"""Shared deterministic synthetic builders for the calibration tests.

Not a test module: it only provides helpers. The generated data are auxiliary
(D-65) and never define correctness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761tool.calib.model import CalibrationModel, fixed_deposition_edges_kev
from kc761tool.calib.scaling import scale_curve
from kc761tool.calib.types import DatasetSpec
from kc761tool.schema.axes import channel_axis, energy_axis
from kc761tool.schema.products import Histogram1D

Q_TRUE = np.array([-140.0, 1.5, 2.5, 3.5, 3.0, 22.0, 38.0])
SCALE_TRUE = np.array([256.0, 0.9, 1.0, 1.1])
S0_FRACTION = 0.30
"""True control channel as a fraction of the window (not the midpoint, so the
truth is a genuine Bezier parabola rather than a quadratic polynomial)."""

# Feature sets sized so the mapped peaks stay inside the channel range of the
# matching channel count (see the E(ch) bounds in F-CAL-3).
FEATURES_MINI = ((40.0, 3000.0, 2.0), (90.0, 1500.0, 3.0), (150.0, 900.0, 4.0))
FEATURES_SMALL = ((60.0, 3000.0, 3.0), (180.0, 1500.0, 4.0), (350.0, 900.0, 5.0))
DEPOSITION_AXIS_NAME = "deposition_energy_kev"


def true_scale_params(n_channels: int) -> NDArray[np.float64]:
    """True Bezier parameters ``(s0, s1, s2, s3)`` for a synthetic dataset."""
    return np.array(
        [
            S0_FRACTION * (n_channels - 1),
            SCALE_TRUE[1],
            SCALE_TRUE[2],
            SCALE_TRUE[3],
        ],
        dtype=np.float64,
    )


def make_mc_spectrum(
    features: tuple[tuple[float, float, float], ...],
    *,
    with_variance: bool = False,
) -> Histogram1D:
    """Compact MC deposition spectrum on the fixed axis (D-33/D-101).

    Subnormal Monte-Carlo tails are zeroed so the synthetic support hull (and
    with it the active-column response build) stays small; production spectra
    are untouched.
    """
    edges = fixed_deposition_edges_kev()
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.zeros_like(centers)
    for mu, amplitude, sigma in features:
        values += amplitude * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
    values[values < 1e-12 * max(1.0, float(values.max()))] = 0.0
    variances = np.maximum(values, 0.0) if with_variance else np.zeros_like(values)
    return Histogram1D(
        axis=energy_axis(edges, name=DEPOSITION_AXIS_NAME),
        values=values,
        variances=variances,
    )


def make_dataset(
    n_channels: int,
    features: tuple[tuple[float, float, float], ...],
    *,
    seed: int,
    q_true: NDArray[np.float64] = Q_TRUE,
    noise: str = "gaussian",
    with_mc_variance: bool = False,
) -> tuple[DatasetSpec, NDArray[np.float64]]:
    """Build a synthetic dataset with a known truth.

    ``noise="gaussian"`` draws ``pred + N(0, sqrt(max(pred, 1)))`` and stores
    that variance, so the F-CAL-1 weights are exact; ``noise="poisson"`` draws
    Poisson counts with their ``fSumw2``. Returns the spec and the true
    prediction vector.
    """
    mc = make_mc_spectrum(features, with_variance=with_mc_variance)
    dummy = DatasetSpec(
        label="d0",
        data=Histogram1D(
            axis=channel_axis(n_channels),
            values=np.ones(n_channels),
            variances=np.ones(n_channels),
        ),
        mc=mc,
        channel_low=0,
        channel_high=n_channels - 1,
        syst_frac=0.0,
    )
    model = CalibrationModel([dummy])
    projection = model._response_cache(np.asarray(q_true, dtype=np.float64)).datasets[0]
    full_scale = true_scale_params(n_channels)
    scale = scale_curve(
        full_scale,
        projection.channel_centers,
        0,
        n_channels - 1,
    )
    prediction = scale * projection.model_counts
    variance = np.maximum(prediction, 1.0)
    rng = np.random.default_rng(seed)
    if noise == "gaussian":
        counts = prediction + np.sqrt(variance) * rng.standard_normal(n_channels)
        data_variance = variance
    elif noise == "poisson":
        counts = rng.poisson(prediction).astype(np.float64)
        data_variance = np.maximum(counts, 1.0)
    else:
        raise ValueError(f"unknown noise model {noise!r}")
    spec = DatasetSpec(
        label="d0",
        data=Histogram1D(
            axis=channel_axis(n_channels),
            values=counts,
            variances=data_variance,
        ),
        mc=mc,
        channel_low=0,
        channel_high=n_channels - 1,
        syst_frac=0.0,
    )
    return spec, prediction


__all__ = [
    "FEATURES_MINI",
    "FEATURES_SMALL",
    "Q_TRUE",
    "S0_FRACTION",
    "SCALE_TRUE",
    "make_dataset",
    "make_mc_spectrum",
    "true_scale_params",
]
