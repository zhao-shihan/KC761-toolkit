"""W3 fit orchestration and product tests (F-CAL-1..F-CAL-5, D-106).

The synthetic recovery check is auxiliary (D-65); it uses data drawn with the
fitted variance so the F-CAL-1 weights are exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kc761.calib.fit import STATUS_CONVERGED, STATUS_STOPPED, _scale_bound_flags, run_fit
from kc761.calib.model import N_CORE, CalibrationModel, fixed_deposition_edges_kev
from kc761.calib.scaling import scale_curve
from kc761.calib.types import DatasetSpec, FitSettings
from kc761.core.model import InternalCalibration, internal_to_reported
from kc761.errors import SolverError, ValidationError
from kc761.schema.axes import energy_axis
from kc761.schema.io import read_product
from kc761.schema.products import CalibProduct, Histogram1D
from tests.test_calib_support import (
    FEATURES_MINI,
    FEATURES_SMALL,
    Q_TRUE,
    make_dataset,
    make_mc_spectrum,
    true_scale_params,
)

RECOVERY_CHANNELS = 256
PRODUCT_CHANNELS = 128


def _reported_truth(channel_max: float) -> np.ndarray:
    reported = internal_to_reported(
        InternalCalibration.from_array(Q_TRUE[:4]), channel_max=channel_max
    )
    return np.array([reported.c0, reported.c1, reported.c2, reported.c3, *Q_TRUE[4:]])


def test_synthetic_recovery_and_chi2() -> None:
    spec, prediction = make_dataset(RECOVERY_CHANNELS, FEATURES_SMALL, seed=2024)
    result = run_fit([spec], plot=False)
    assert result.status == STATUS_CONVERGED
    assert result.success
    reduced = result.chi2 / result.dof
    assert 0.4 < reduced < 1.8
    truth = _reported_truth(result.channel_max)
    sigma = np.sqrt(np.maximum(np.diag(result.param_cov), 0.0))
    fitted = np.array([*result.params_reported, *result.resol_params])
    assert np.all(np.abs(fitted - truth) <= 5.0 * sigma)
    channels = np.linspace(
        spec.channel_low, spec.channel_high, spec.channel_high - spec.channel_low + 1
    )
    fitted_curve = scale_curve(
        result.scales[0].params, channels, spec.channel_low, spec.channel_high
    )
    true_curve = scale_curve(
        true_scale_params(RECOVERY_CHANNELS), channels, spec.channel_low, spec.channel_high
    )
    # The scale is only constrained where the folded model has support.
    supported = prediction > 0.01 * float(np.max(prediction))
    assert np.max(np.abs(fitted_curve[supported] - true_curve[supported])) < 0.05
    assert "chi2/dof" in result.report


def test_product_roundtrip_and_metadata(tmp_path: Path) -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    output = tmp_path / "calib.root"
    result = run_fit(
        [spec],
        output=output,
        strict=True,
        plot=False,
        command="kc761 calib --data data.root",
        arguments=(("--data", "data.root"),),
    )
    assert output.is_file()
    assert result.product_path == output
    product = read_product(output, strict=True)
    assert isinstance(product, CalibProduct)
    assert product.chi2 == pytest.approx(result.chi2)
    assert product.dof == result.dof
    assert product.covariance_scale == pytest.approx(result.covariance_scale)
    assert product.fit_status == STATUS_CONVERGED
    assert product.params_reported == pytest.approx(result.params_reported)
    assert product.resol_params == pytest.approx(result.resol_params)
    assert product.channel_max == pytest.approx(result.channel_max)
    assert product.resol_clamp_count == result.resol_clamp_count
    assert len(product.scales) == 1
    label, params = product.scales[0]
    assert label == "d0"
    assert params == pytest.approx(result.scales[0].params)
    assert len(product.scale_bound_flags) == 1
    flag_label, flag_values = product.scale_bound_flags[0]
    assert flag_label == "d0"
    assert len(flag_values) == 4
    assert all(isinstance(flag, bool) for flag in flag_values)
    assert np.allclose(product.deposition_to_channel.values.sum(axis=0), 1.0, atol=1e-10)


def test_product_is_not_overwritten_without_force(tmp_path: Path) -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    output = tmp_path / "calib.root"
    run_fit([spec], output=output, plot=False)
    from kc761.errors import SchemaError

    with pytest.raises(SchemaError, match="refusing to overwrite"):
        run_fit([spec], output=output, plot=False)
    run_fit([spec], output=output, force=True, plot=False)


def test_non_convergence_is_recorded_and_written(tmp_path: Path) -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    output = tmp_path / "early.root"
    result = run_fit(
        [spec],
        output=output,
        plot=False,
        settings=FitSettings(maxiter=1),
    )
    assert result.status == STATUS_STOPPED
    assert not result.success
    product = read_product(output)
    assert product.fit_status == STATUS_STOPPED


def test_non_convergence_raises_in_strict_mode() -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    with pytest.raises(SolverError, match="did not converge"):
        run_fit(
            [spec],
            plot=False,
            strict=True,
            settings=FitSettings(maxiter=1),
        )


def test_covariance_is_symmetric_psd_and_scaled() -> None:
    spec, _ = make_dataset(RECOVERY_CHANNELS, FEATURES_SMALL, seed=11)
    result = run_fit([spec], plot=False)
    covariance = result.param_cov
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    assert eigenvalues.min() >= -1e-8 * max(1.0, float(np.max(eigenvalues)))
    assert result.covariance_scale > 0.0
    assert np.isfinite(result.covariance_scale)


def test_plot_is_written_and_refuses_overwrite(tmp_path: Path) -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    output = tmp_path / "calib.root"
    result = run_fit([spec], output=output, plot=True)
    assert result.plot_path is not None
    assert result.plot_path.is_file()
    from kc761.errors import UsageError

    with pytest.raises(UsageError, match="refusing to overwrite"):
        run_fit([spec], output=output, plot=True, force=True)


def test_invalid_inputs_raise_classified_errors() -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    tiny = DatasetSpec(
        label="tiny",
        data=spec.data,
        mc=spec.mc,
        channel_low=100,
        channel_high=105,
    )
    with pytest.raises(ValidationError, match="dof"):
        run_fit([tiny], plot=False)
    with pytest.raises(ValidationError, match="at least one dataset"):
        run_fit([], plot=False)
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
        channel_high=PRODUCT_CHANNELS - 1,
    )
    with pytest.raises(ValidationError, match="fixed source-mode deposition axis"):
        run_fit([bad], plot=False)


def test_mc_variance_absence_still_fits() -> None:
    mc = make_mc_spectrum(FEATURES_MINI, with_variance=False)
    stripped = Histogram1D(axis=mc.axis, values=mc.values, variances=None)
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    dataset = DatasetSpec(
        label="poisson",
        data=spec.data,
        mc=stripped,
        channel_low=0,
        channel_high=PRODUCT_CHANNELS - 1,
    )
    result = run_fit([dataset], plot=False)
    assert np.isfinite(result.chi2)


def test_two_datasets_share_the_core() -> None:
    first, _ = make_dataset(RECOVERY_CHANNELS, FEATURES_SMALL, seed=31)
    second, _ = make_dataset(RECOVERY_CHANNELS, FEATURES_SMALL, seed=32)
    second = DatasetSpec(
        label="d1",
        data=second.data,
        mc=second.mc,
        channel_low=25,
        channel_high=RECOVERY_CHANNELS - 1,
        syst_frac=0.05,
    )
    result = run_fit([first, second], plot=False)
    assert result.status == STATUS_CONVERGED
    assert len(result.scales) == 2
    assert {scale.label for scale in result.scales} == {"d0", "d1"}
    assert result.n_free == 7 + 4 * 2
    assert result.dof == (RECOVERY_CHANNELS + RECOVERY_CHANNELS - 25) - result.n_free
    assert np.linalg.eigvalsh(result.param_cov).min() >= -1e-9


def test_scale_bound_flags_mark_bound_hits() -> None:
    spec, _ = make_dataset(PRODUCT_CHANNELS, FEATURES_MINI, seed=11)
    model = CalibrationModel([spec])
    theta = model.x0.copy()
    theta[N_CORE] = model.bounds[N_CORE][0]  # s0 on its lower bound
    flags = _scale_bound_flags(theta, model)
    assert flags[0][0] == "d0"
    assert flags[0][1][0] is True
    assert flags[0][1][1:] == (False, False, False)


def test_fixed_axis_helper_shape() -> None:
    assert fixed_deposition_edges_kev().size == 4097
