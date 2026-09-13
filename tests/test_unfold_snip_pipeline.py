"""End-to-end SNIP mask coverage (F-SOLVE-4/5/6, D-157/D-160/D-187/D-188).

The shared unfold fixture cannot carry the mask (its primary axis is not the
channel axis, D-121), so these tests use :func:`make_snip_sim_product` and run
the real ``run_unfold`` path with the mask enabled in every combination that
the settings expose.
"""

from __future__ import annotations

import numpy as np
import pytest

import kc761tool.unfold.solve as solve_module
from kc761tool.schema.io import read_product
from kc761tool.unfold import run_unfold
from kc761tool.unfold.inputs import internal_calibration, primary_edges_kev
from kc761tool.unfold.selection import select_window
from tests.test_unfold_support import (
    N_CHANNELS,
    make_calib_product,
    make_snip_sim_product,
    make_spectrum_product,
    response_of,
    write,
)

ALPHA = 1e-3
TRUTH = {10: 900.0, 15: 1500.0, 20: 700.0}


def _run(tmp_path, **kwargs):
    calib = make_calib_product()
    sim = make_snip_sim_product(calib)
    signal = response_of(calib, sim) @ _truth_vector()
    edges = np.asarray(calib.deposition_to_channel.y.edges, dtype=np.float64)
    data = write(tmp_path / "d.root", make_spectrum_product(signal))
    calib_path = write(tmp_path / "c.root", calib)
    sim_path = write(tmp_path / "s.root", sim)
    options: dict[str, object] = {
        "energy_low_kev": float(edges[8]),
        "energy_high_kev": float(edges[24]),
        "alpha": ALPHA,
        "output": tmp_path / "u.root",
        "strict": True,
        "plot": False,
    }
    options.update(kwargs)
    return run_unfold(data, calib_path, sim_path, **options)


def _truth_vector() -> np.ndarray:
    truth = np.zeros(N_CHANNELS)
    for index, amplitude in TRUTH.items():
        truth[index] = amplitude
    return truth


def test_snip_enabled_pipeline_is_valid_and_auditable(tmp_path) -> None:
    """D-160/D-188: the mask runs, the certificates pass and the meta round-trips."""
    result = _run(tmp_path)
    assert result.certificate is not None and result.certificate.ok
    assert np.all(np.isfinite(result.mu)) and np.all(result.mu >= 0.0)
    meta = dict(read_product(tmp_path / "u.root", strict=True).settings)
    assert int(meta["snip_enabled"]) == 1
    assert int(meta["snip_protect_bins"]) == 3
    assert int(meta["snip_iterations"]) >= 1
    assert int(meta["snip_candidates"]) >= 1
    assert int(meta["snip_protected_bins"]) >= 1
    assert len(meta["snip_mask_sha256"]) == 64
    assert len(meta["snip_baseline_sha256"]) == 64


def test_snip_meta_counters_describe_the_full_measured_axis(tmp_path) -> None:
    """The recorded counters and mask hash are full-axis; the applied mask is sliced.

    ``snip_candidates``/``snip_protected_bins``/``snip_mask_sha256`` describe the
    mask on the full measured axis (the hash is taken before the reported-window
    slice), while the penalty only sees the slice. A candidate outside the window
    therefore raises the counters without entering the operator.
    """
    result = _run(tmp_path)
    meta = dict(read_product(tmp_path / "u.root", strict=True).settings)
    window = result.window
    protected_full = int(meta["snip_protected_bins"])
    assert protected_full > window.n_report_bins  # the measured axis is larger
    assert int(meta["snip_candidates"]) >= 1
    assert not hasattr(window, "solve_low")


def test_snip_mask_actually_reaches_the_operator(tmp_path) -> None:
    """A mask-on solve must differ from the mask-off solve on the same data."""
    on = _run(tmp_path / "on")
    off = _run(tmp_path / "off", snip_enabled=False)
    assert not np.array_equal(on.mu, off.mu)
    assert np.all(np.isfinite(off.mu))


def test_snip_iteration_reference_is_the_window_midpoint(tmp_path, monkeypatch) -> None:
    """D-157/D-188(c): the pipeline passes the reported-window midpoint bin."""
    seen: list[int] = []
    original = solve_module.snip_peak_mask

    def spy(*args, **kwargs):
        seen.append(int(kwargs["iteration_reference_index"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(solve_module, "snip_peak_mask", spy)
    result = _run(tmp_path)
    calib = make_calib_product()
    sim = make_snip_sim_product(calib)
    selection = select_window(
        calibration=internal_calibration(calib),
        channel_max=calib.channel_max,
        n_channels=N_CHANNELS,
        primary_edges_kev=primary_edges_kev(sim),
        energy_low_kev=result.settings.energy_low_kev,
        energy_high_kev=result.settings.energy_high_kev,
    )
    edges = np.asarray(calib.deposition_to_channel.y.edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected = int(np.argmin(np.abs(centers - selection.midpoint_kev)))
    assert seen == [expected]
    assert expected != centers.size // 2  # not the blind middle of the axis


@pytest.mark.parametrize(
    "options",
    (
        {"snip_floor": 0.0},
        {"snip_floor": 1.0},
        {"snip_protect_bins": 1},
        {"snip_protect_bins": 100},
        {"snip_threshold_sigma": 1.0},
        {"snip_iterations": 5},
        {"snip_max_iterations": 1},
        {"difference_order": 1},
    ),
)
def test_snip_setting_combinations_stay_valid(tmp_path, options) -> None:
    result = _run(tmp_path, **options)
    assert result.certificate is not None and result.certificate.ok
    assert np.all(np.isfinite(result.mu))
    assert np.all(result.mu >= 0.0)


def test_protected_footprint_does_not_scale_with_the_resolution() -> None:
    """D-188(a): the protected half-width is in primary bins, not resolution widths.

    A single peak as wide as the detector resolution (the case the iteration
    count is derived for) must protect exactly ``2 * protect_bins + 1`` bins for
    a resolution of one, six and thirty bins, while the iteration count -- the
    detection step -- follows the resolution.
    """
    from kc761tool.core.solver import SnipSettings, snip_peak_mask

    size = 256
    grid = np.arange(size, dtype=np.float64)
    settings = SnipSettings()
    footprints: set[int] = set()
    iterations: set[int] = set()
    for resolution_bins in (1.0, 6.0, 30.0):
        values = np.full(size, 300.0) + (3000.0 * resolution_bins) * np.exp(
            -0.5 * ((grid - size // 2) / resolution_bins) ** 2
        )
        mask = snip_peak_mask(
            values, np.sqrt(values), np.full(size, resolution_bins), np.ones(size), settings
        )
        assert mask.n_candidates == 1
        assert mask.n_protected == 2 * settings.protect_bins + 1
        footprints.add(mask.n_protected)
        iterations.add(mask.iterations)
    assert len(footprints) == 1  # protection is resolution-independent
    assert len(iterations) > 1  # the SNIP iteration count is not
