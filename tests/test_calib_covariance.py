"""Calibration covariance tests (F-CAL-5) and a pull-coverage check.

The pull distribution is auxiliary (D-65): it samples several synthetic
datasets drawn with the fitted variance and checks that the F-CAL-5 covariance
covers the recovered parameters within generous bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from kc761tool.calib.covariance import calibration_covariance
from kc761tool.calib.fit import run_fit
from kc761tool.core.covariance import fisher_information
from kc761tool.core.model import InternalCalibration, internal_jacobian, internal_to_reported
from tests.test_calib_support import FEATURES_SMALL, Q_TRUE, make_dataset

N_CHANNELS = 256
N_SEEDS = 3


def _transform_7x7() -> np.ndarray:
    transform = np.zeros((7, 7), dtype=np.float64)
    transform[:4, :4] = internal_jacobian(channel_max=N_CHANNELS - 1)
    transform[4:, 4:] = np.eye(3)
    return transform


def test_f_cal_5_transform_matches_the_manual_block() -> None:
    """On a full-rank Fisher the Schur core block equals the full inverse block.

    A hand-built well-conditioned Jacobian keeps this independent of the fit
    and of the Bezier gauge directions.
    """
    rng = np.random.default_rng(99)
    n_bins = 400
    n_params = 11
    jacobian = rng.standard_normal((n_bins, n_params))
    # Give the last four (scale) columns a weaker but full-rank signature.
    jacobian[:, 7:] *= 0.1
    variance = 1.0 + rng.random(n_bins)
    chi2 = float(rng.uniform(300.0, 600.0))
    dof = n_bins - n_params
    estimate = calibration_covariance(
        jacobian, variance, chi2=chi2, dof=dof, channel_max=N_CHANNELS - 1
    )
    fisher = np.asarray(fisher_information(jacobian, np.sqrt(variance)), dtype=np.float64)
    full = (chi2 / dof) * np.linalg.inv(fisher)
    expected = _transform_7x7() @ full[:7, :7] @ _transform_7x7().T
    assert estimate.estimator == "fisher-x2dof-marginalized-reported"
    assert estimate.scale == pytest.approx(chi2 / dof)
    assert np.allclose(estimate.matrix, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_parameter_pull_distribution_is_covered() -> None:
    pulls: list[np.ndarray] = []
    for seed in range(N_SEEDS):
        spec, _ = make_dataset(N_CHANNELS, FEATURES_SMALL, seed=100 + seed)
        result = run_fit([spec], plot=False)
        reduced = result.chi2 / result.dof
        assert 0.3 < reduced < 2.0
        reported = internal_to_reported(
            InternalCalibration.from_array(Q_TRUE[:4]),
            channel_max=result.channel_max,
        )
        truth = np.array([reported.c0, reported.c1, reported.c2, reported.c3, *Q_TRUE[4:]])
        sigma = np.sqrt(np.maximum(np.diag(result.param_cov), 0.0))
        fitted = np.array([*result.params_reported, *result.resol_params])
        usable = (sigma > 0.0) & np.array([True, True, True, True, True, True, False])
        pulls.append((fitted[usable] - truth[usable]) / sigma[usable])
    combined = np.concatenate(pulls)
    assert combined.size >= 4
    assert abs(float(np.mean(combined))) < 1.5
    assert 0.2 < float(np.std(combined)) < 4.0
