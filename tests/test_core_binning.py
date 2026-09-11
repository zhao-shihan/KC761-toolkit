"""Auxiliary checks for grids and the working window (F-BIN-1..3, D-79)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761.core import binning
from kc761.core.model import InternalCalibration
from kc761.errors import ValidationError


def test_channel_grid() -> None:
    grid = binning.ChannelGrid(8)
    assert np.array_equal(grid.edges(), np.arange(-0.5, 8.5, 1.0))
    assert np.array_equal(grid.centers(), np.arange(8.0))
    with pytest.raises(ValidationError):
        binning.ChannelGrid(0)
    with pytest.raises(ValidationError, match="GiB"):
        binning.ChannelGrid(binning.MAX_CHANNELS + 1)
    # D-52: the over-limit failure carries a footprint estimate so an 8192-bin
    # request is rejected before any dense allocation is attempted.
    message = binning.channels_limit_message(8192)
    assert "8192" in message and "GiB" in message
    assert binning.dense_matrix_bytes(8192) == 8192 * 8192 * 8


def test_energy_grid_validation_and_helpers() -> None:
    grid = binning.EnergyGrid(edges_kev=np.array([0.0, 1.0, 3.0]))
    assert grid.n_bins == 2
    assert np.allclose(grid.centers_kev(), [0.5, 2.0])
    assert np.allclose(grid.widths_kev(), [1.0, 2.0])
    with pytest.raises(ValidationError):
        binning.EnergyGrid(edges_kev=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValidationError):
        binning.EnergyGrid(edges_kev=np.array([0.0]))


def _identity_calibration() -> InternalCalibration:
    # E(channel) = 10 keV per channel.
    return InternalCalibration(c0=0.0, k1=10.0, k2=10.0, k3=10.0)


def test_working_window_converts_sigma_to_channels() -> None:
    low, high = binning.working_window(
        10,
        20,
        pad_nsigma=5.0,
        calibration=_identity_calibration(),
        resol_params=np.array([2.0, 2.0, 2.0]),
        channel_max=63.0,
        n_channels=64,
    )
    # pad = ceil(5 * 2 keV / 10 keV per channel) = 1 channel on each side.
    assert (low, high) == (9, 21)


def test_working_window_clips_to_detector() -> None:
    low, high = binning.working_window(
        0,
        1,
        pad_nsigma=5.0,
        calibration=_identity_calibration(),
        resol_params=np.array([5.0, 5.0, 5.0]),
        channel_max=63.0,
        n_channels=4,
    )
    assert (low, high) == (0, 3)


def test_working_window_rejects_bad_window() -> None:
    with pytest.raises(ValidationError):
        binning.working_window(
            5,
            2,
            pad_nsigma=5.0,
            calibration=_identity_calibration(),
            resol_params=np.array([2.0, 2.0, 2.0]),
            channel_max=63.0,
            n_channels=64,
        )
