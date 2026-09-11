"""Window selection and data-side weights (F-UNF-1/F-UNF-2, D-111/D-112)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kc761tool.errors import ValidationError
from kc761tool.unfold.selection import fit_sigma, select_window
from tests.test_unfold_support import (
    N_CHANNELS,
    calibration,
    make_calib_product,
    primary_edges_kev,
)


def _selection(energy_low: float, energy_high: float):
    calib = make_calib_product()
    return select_window(
        calibration=calibration(calib),
        resol_params=np.asarray(calib.resol_params),
        channel_max=calib.channel_max,
        n_channels=N_CHANNELS,
        primary_edges_kev=primary_edges_kev(),
        energy_low_kev=energy_low,
        energy_high_kev=energy_high,
        pad_nsigma=5.0,
    )


def test_window_maps_energy_to_channel_and_primary_ranges() -> None:
    selection = _selection(220.0, 520.0)
    assert selection.channel_low == 18
    assert selection.channel_high == 29
    assert selection.solve_low < selection.channel_low
    assert selection.solve_high > selection.channel_high
    assert selection.report_low == 22
    assert selection.report_high == 51
    assert selection.n_report_bins == 30


def test_window_rejects_inverted_and_out_of_range() -> None:
    with pytest.raises(ValidationError, match="energy_low_kev"):
        _selection(500.0, 500.0)
    with pytest.raises(ValidationError, match="outside"):
        _selection(-10.0, 200.0)
    with pytest.raises(ValidationError, match="outside"):
        _selection(200.0, 1000.0)


def test_window_rejects_empty_channel_or_primary_selection() -> None:
    # A window narrower than the channel pitch and between primary centers.
    with pytest.raises(ValidationError):
        _selection(220.0, 221.0)


def test_window_at_the_acquisition_edge_is_clipped() -> None:
    selection = _selection(540.0, 585.0)
    assert 0 <= selection.solve_low <= selection.channel_low
    assert selection.channel_high <= selection.solve_high <= N_CHANNELS - 1
    assert selection.solve_high == selection.channel_high or (
        selection.solve_high == N_CHANNELS - 1
    )


def test_window_entirely_above_channel_range_is_rejected() -> None:
    """D-146: no silent degenerate fit on the top channel."""
    calib = make_calib_product()
    wide_primary = np.linspace(0.0, 1.0e4, 101)
    with pytest.raises(ValidationError, match="outside the channel energy range"):
        select_window(
            calibration=calibration(calib),
            resol_params=np.asarray(calib.resol_params),
            channel_max=calib.channel_max,
            n_channels=N_CHANNELS,
            primary_edges_kev=wide_primary,
            energy_low_kev=9000.0,
            energy_high_kev=9500.0,
            pad_nsigma=5.0,
        )


def test_fit_sigma_matches_documented_data_side_formula() -> None:
    stat = np.array([0.0, 4.0, 9.0])
    data = np.array([10.0, 20.0, 30.0])
    sigma = fit_sigma(stat, data, 0.1)
    expected = np.sqrt(np.maximum(stat, 1.0) + (0.1 * data) ** 2)
    assert np.allclose(sigma, expected)
    with pytest.raises(ValidationError, match="syst_frac"):
        fit_sigma(stat, data, -0.1)
    with pytest.raises(ValidationError):
        fit_sigma(stat, data[:2], 0.1)


@given(
    data=st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    ),
    syst=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_fit_sigma_is_at_least_the_statistical_floor(data, syst) -> None:
    counts = np.asarray(data, dtype=np.float64)
    stat = np.maximum(counts * 0.25, 0.0)
    sigma = fit_sigma(stat, counts, syst)
    assert np.all(sigma >= np.sqrt(np.maximum(stat, 1.0)))
    assert np.all(np.isfinite(sigma))
