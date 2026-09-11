"""Calibration forward-model tests (F-CAL-1/F-CAL-3/F-CAL-4).

Analytic Jacobians and gradients are checked against central differences; the
restricted active-column response is checked against the full fixed-axis
response (D-101). Finite differences are auxiliary only (D-65).
"""

from __future__ import annotations

import numpy as np
import pytest

from kc761tool.calib.model import (
    CalibrationModel,
    fixed_deposition_edges_kev,
)
from kc761tool.calib.types import DatasetSpec
from kc761tool.core.binning import ChannelGrid
from kc761tool.core.model import InternalCalibration
from kc761tool.core.response import build_response_matrix
from kc761tool.errors import ValidationError
from kc761tool.schema.axes import channel_axis, energy_axis
from kc761tool.schema.products import Histogram1D
from tests.test_calib_support import (
    FEATURES_MINI,
    Q_TRUE,
    make_dataset,
    make_mc_spectrum,
)

N_CHANNELS = 128


def _model(**kwargs) -> CalibrationModel:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7, **kwargs)
    return CalibrationModel([spec])


def test_jacobian_matches_central_differences() -> None:
    model = _model()
    theta = 0.5 * (model.x0 + np.array([b[1] for b in model.bounds]))
    theta = np.clip(theta, [b[0] for b in model.bounds], [b[1] for b in model.bounds])
    jacobian = model.jacobian(theta)
    assert jacobian.shape == (model.n_bins, model.n_free)
    step = 1e-6
    for index in range(model.n_free):
        up = theta.copy()
        down = theta.copy()
        up[index] += step
        down[index] -= step
        finite_difference = (
            model.dataset_details(up)[0].prediction - model.dataset_details(down)[0].prediction
        ) / (2.0 * step)
        magnitude = max(1.0, float(np.max(np.abs(jacobian[:, index]))))
        assert np.max(np.abs(jacobian[:, index] - finite_difference)) < 1e-5 * magnitude


def test_variance_gradient_matches_central_differences() -> None:
    model = _model()
    theta = np.clip(
        0.5 * (model.x0 + np.array([b[1] for b in model.bounds])),
        [b[0] for b in model.bounds],
        [b[1] for b in model.bounds],
    )
    gradient = model.variance_gradient(theta)
    step = 1e-6
    for index in range(model.n_free):
        up = theta.copy()
        down = theta.copy()
        up[index] += step
        down[index] -= step
        finite_difference = (
            model.dataset_details(up)[0].fit_sigma ** 2
            - model.dataset_details(down)[0].fit_sigma ** 2
        ) / (2.0 * step)
        assert np.allclose(gradient[:, index], finite_difference, rtol=1e-5, atol=1e-7)


def test_objective_gradient_matches_central_differences() -> None:
    model = _model()
    theta = model.x0.copy()
    gradient = model.gradient(theta)
    step = 1e-6
    for index in range(model.n_free):
        up = theta.copy()
        down = theta.copy()
        up[index] += step
        down[index] -= step
        finite_difference = (model.evaluate(up) - model.evaluate(down)) / (2.0 * step)
        assert gradient[index] == pytest.approx(finite_difference, rel=1e-5, abs=1e-6)


def test_active_columns_equal_the_full_fixed_axis_response() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    model = CalibrationModel([spec])
    projection = model._response_cache(Q_TRUE).datasets[0]
    full = build_response_matrix(
        fixed_deposition_edges_kev(),
        InternalCalibration.from_array(Q_TRUE[:4]),
        Q_TRUE[4:],
        channel_grid=ChannelGrid(N_CHANNELS),
        channel_max=N_CHANNELS - 1,
    )
    expected = np.asarray(full.matrix @ spec.mc.values.astype(np.float64), dtype=np.float64)
    assert np.allclose(projection.model_counts, expected, rtol=1e-10, atol=1e-10)


def test_mc_variance_is_the_propagated_diagonal() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7, with_mc_variance=True)
    model = CalibrationModel([spec])
    projection = model._response_cache(Q_TRUE).datasets[0]
    full = build_response_matrix(
        fixed_deposition_edges_kev(),
        InternalCalibration.from_array(Q_TRUE[:4]),
        Q_TRUE[4:],
        channel_grid=ChannelGrid(N_CHANNELS),
        channel_max=N_CHANNELS - 1,
    )
    expected = np.asarray(
        full.matrix.power(2) @ spec.mc.variances.astype(np.float64), dtype=np.float64
    )
    assert np.allclose(projection.mc_variance, expected, rtol=1e-10, atol=1e-10)


def test_fit_variance_follows_f_cal_1() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7, noise="poisson")
    model = CalibrationModel([spec])
    theta = model.x0.copy()
    variance = model.variance(theta)
    projection = model._response_cache(theta[:7]).datasets[0]
    scale = model._scale_values(theta)[0]
    expected = (
        np.maximum(spec.data.variances, 1.0)
        + (spec.syst_frac * spec.data.values) ** 2
        + scale * scale * projection.mc_variance
    )
    assert np.allclose(variance, expected, rtol=1e-12, atol=1e-12)


def test_rejects_mismatched_mc_axis() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    wrong_mc = Histogram1D(
        axis=energy_axis(np.linspace(0.0, 1000.0, 11), name="deposition_energy_kev"),
        values=np.ones(10),
        variances=np.ones(10),
    )
    bad = DatasetSpec(
        label="bad",
        data=spec.data,
        mc=wrong_mc,
        channel_low=0,
        channel_high=N_CHANNELS - 1,
    )
    with pytest.raises(ValidationError, match="fixed source-mode deposition axis"):
        CalibrationModel([bad])


def test_rejects_missing_data_variance() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    bad = DatasetSpec(
        label="bad",
        data=Histogram1D(axis=spec.data.axis, values=spec.data.values, variances=None),
        mc=spec.mc,
        channel_low=0,
        channel_high=N_CHANNELS - 1,
    )
    with pytest.raises(ValidationError, match="fSumw2"):
        CalibrationModel([bad])


def test_rejects_mismatched_channel_counts() -> None:
    first, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    other = DatasetSpec(
        label="second",
        data=Histogram1D(
            axis=channel_axis(N_CHANNELS // 2),
            values=np.ones(N_CHANNELS // 2),
            variances=np.ones(N_CHANNELS // 2),
        ),
        mc=first.mc,
        channel_low=0,
        channel_high=N_CHANNELS // 2 - 1,
    )
    with pytest.raises(ValidationError, match="same acquisition channel count"):
        CalibrationModel([first, other])


def test_rejects_underdetermined_window() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    bad = DatasetSpec(
        label="tiny",
        data=spec.data,
        mc=spec.mc,
        channel_low=90,
        channel_high=95,
    )
    with pytest.raises(ValidationError, match="dof"):
        CalibrationModel([bad])


def test_rejects_window_outside_channels() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    bad = DatasetSpec(
        label="out",
        data=spec.data,
        mc=spec.mc,
        channel_low=0,
        channel_high=N_CHANNELS,
    )
    with pytest.raises(ValidationError, match="outside"):
        CalibrationModel([bad])


def test_mc_spectrum_without_variance_uses_poisson() -> None:
    mc = make_mc_spectrum(FEATURES_MINI, with_variance=False)
    stripped = Histogram1D(axis=mc.axis, values=mc.values, variances=None)
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    dataset = DatasetSpec(
        label="poisson-mc",
        data=spec.data,
        mc=stripped,
        channel_low=0,
        channel_high=N_CHANNELS - 1,
    )
    model = CalibrationModel([dataset])
    projection = model._response_cache(Q_TRUE).datasets[0]
    assert np.all(projection.mc_variance >= 0.0)


def test_default_channel_axis_must_be_channel() -> None:
    spec, _ = make_dataset(N_CHANNELS, FEATURES_MINI, seed=7)
    bad_data = Histogram1D(
        axis=energy_axis(np.linspace(0.0, float(N_CHANNELS), N_CHANNELS + 1), name="energy_kev"),
        values=spec.data.values,
        variances=spec.data.variances,
    )
    bad = DatasetSpec(
        label="wrong-axis",
        data=bad_data,
        mc=spec.mc,
        channel_low=0,
        channel_high=N_CHANNELS - 1,
    )
    with pytest.raises(ValidationError, match="channel axis"):
        CalibrationModel([bad])


def test_channel_axis_helper_is_the_expected_one() -> None:
    axis = channel_axis(N_CHANNELS)
    assert axis.unit == "channel"
