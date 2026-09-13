"""Auxiliary checks for the channel and energy grids (F-BIN-1/F-BIN-2, D-79)."""

from __future__ import annotations

import numpy as np
import pytest

from kc761tool.core import binning
from kc761tool.errors import ValidationError


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
