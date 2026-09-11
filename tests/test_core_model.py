"""Auxiliary checks for the energy/resolution model (F-MODEL-1..5)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761tool.core import model
from kc761tool.errors import CertificateError


def _calibration() -> model.InternalCalibration:
    return model.InternalCalibration(c0=1.5, k1=2.0, k2=2.05, k3=2.1)


def test_internal_reported_roundtrip() -> None:
    calibration = _calibration()
    maximum = 2048.0
    reported = model.internal_to_reported(calibration, channel_max=maximum)
    back = model.reported_to_internal(reported, channel_max=maximum)
    assert np.allclose(back.as_array(), calibration.as_array(), rtol=1e-12)
    channels = np.linspace(0.0, maximum, 17)
    assert np.allclose(
        model.energy_kev(channels, calibration, channel_max=maximum),
        model.energy_kev(channels, back, channel_max=maximum),
    )


def test_energy_gradient_matches_finite_difference() -> None:
    calibration = _calibration()
    maximum = 2048.0
    channels = np.linspace(0.0, maximum, 9)
    gradient = model.energy_grad_internal(channels, channel_max=maximum)
    for index in range(4):
        step = 1e-6
        plus = list(calibration.as_array())
        minus = list(calibration.as_array())
        plus[index] += step
        minus[index] -= step
        finite = (
            model.energy_kev(
                channels, model.InternalCalibration(*plus), channel_max=maximum
            )
            - model.energy_kev(
                channels, model.InternalCalibration(*minus), channel_max=maximum
            )
        ) / (2.0 * step)
        assert np.allclose(gradient[index], finite, rtol=1e-6, atol=1e-6)


def test_reported_gradient_matches_finite_difference() -> None:
    reported = model.ReportedCalibration(c0=1.5, c1=2.0, c2=1e-3, c3=1e-8)
    channels = np.array([0.0, 1.0, 17.5, 100.0])
    gradient = model.energy_grad_reported(channels)
    for index in range(4):
        step = 1e-6
        values = list(reported.as_array())
        values[index] += step
        plus = model.ReportedCalibration(*values)
        values[index] -= 2.0 * step
        minus = model.ReportedCalibration(*values)
        internal_plus = model.reported_to_internal(plus, channel_max=100.0)
        internal_minus = model.reported_to_internal(minus, channel_max=100.0)
        finite = (
            model.energy_kev(channels, internal_plus, channel_max=100.0)
            - model.energy_kev(channels, internal_minus, channel_max=100.0)
        ) / (2.0 * step)
        assert np.allclose(gradient[index], finite, rtol=1e-6, atol=1e-6)


def test_resolution_gradient_matches_finite_difference() -> None:
    energies = np.array([0.0, 50.0, 500.0, 2000.0, 3500.0])
    params = np.array([2.0, 10.0, 20.0])
    sigma, gradient = model.resolution_sigma_grad(energies, params)
    assert np.all(sigma > 0.0)
    for index in range(3):
        step = 1e-6
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        finite = (
            model.resolution_sigma_kev(energies, plus)
            - model.resolution_sigma_kev(energies, minus)
        ) / (2.0 * step)
        assert np.allclose(gradient[index], finite, rtol=1e-7, atol=1e-9)


def test_resolution_non_strict_clamps_with_warning() -> None:
    params = np.array([1.0, 5.0, 1.0])
    energies = np.array([3000.0, 4000.0])
    assert np.any(model.resolution_variance_kev2(energies, params) < 0.0)
    with pytest.warns(RuntimeWarning, match="F-MODEL-5"):
        sigma = model.resolution_sigma_kev(energies, params)
    assert np.all(sigma == model.SIGMA_FLOOR_KEV)
    with pytest.warns(RuntimeWarning, match="F-MODEL-5"):
        sigma_grad, gradient = model.resolution_sigma_grad(energies, params)
    assert np.all(sigma_grad == model.SIGMA_FLOOR_KEV)
    assert np.all(gradient == 0.0)
    assert model.resolution_clamped_mask(energies, params).all()


def test_resolution_strict_raises_on_negative_variance() -> None:
    params = np.array([1.0, 5.0, 1.0])
    energies = np.array([4000.0])
    with pytest.raises(CertificateError, match="F-MODEL-5"):
        model.resolution_sigma_kev(energies, params, strict=True)
    model.verify_resolution_positivity(np.array([662.0]), params, strict=True)


def test_energy_monotonicity_certificate() -> None:
    good = model.InternalCalibration(c0=0.0, k1=10.0, k2=10.0, k3=10.0)
    model.verify_energy_monotonicity(good, channel_max=63.0, strict=True)
    bad = model.InternalCalibration(c0=0.0, k1=-1.0, k2=-1.0, k3=-1.0)
    with pytest.raises(CertificateError, match="F-MODEL-3"):
        model.verify_energy_monotonicity(bad, channel_max=63.0, strict=True)
