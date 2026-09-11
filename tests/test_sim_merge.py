"""No-Geant4 tests: worker histogram merging and thread estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot

from kc761tool import errors
from kc761tool.schema import _uproot
from kc761tool.schema.axes import energy_axis
from kc761tool.schema.products import Histogram1D, Histogram2D
from kc761tool.sim import runner
from kc761tool.sim.config import MATRIX_HIST_NAME, SPECTRUM_HIST_NAME, ZERO_DEPOSITION_HIST_NAME


def _deposition_axis():
    return energy_axis(np.linspace(0.0, 200.0, 3), name="deposition_energy_kev")


def _primary_axis():
    return energy_axis(np.linspace(0.0, 400.0, 5), name="primary_energy_kev")


def _write_matrix_worker(
    path: Path, counts: np.ndarray, zeros: np.ndarray, *, primary_axis=None
) -> None:
    primary = primary_axis if primary_axis is not None else _primary_axis()
    with uproot.recreate(path) as file:
        _uproot.write_hist2d(
            file,
            MATRIX_HIST_NAME,
            Histogram2D(
                x=_deposition_axis(), y=primary, values=counts, variances=None
            ),
        )
        _uproot.write_hist1d(
            file,
            ZERO_DEPOSITION_HIST_NAME,
            Histogram1D(axis=primary, values=zeros),
        )


def _write_spectrum_worker(path: Path, values: np.ndarray, *, edges=None) -> None:
    axis_edges = edges if edges is not None else np.linspace(0.0, 16.0, 9)
    with uproot.recreate(path) as file:
        _uproot.write_hist1d(
            file,
            SPECTRUM_HIST_NAME,
            Histogram1D(axis=energy_axis(axis_edges, name="energy_kev"), values=values),
        )


def test_matrix_merge_sums_workers_and_keeps_axes(tmp_path: Path) -> None:
    first = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 1.0]])
    second = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    _write_matrix_worker(tmp_path / "w0.root", first, np.array([5.0, 0.0, 1.0, 3.0]))
    _write_matrix_worker(tmp_path / "w1.root", second, np.array([0.0, 2.0, 0.0, 1.0]))
    counts, dep, primary, zeros = runner.merge_matrix_worker_histograms(
        [str(tmp_path / "w0.root"), str(tmp_path / "w1.root")]
    )
    assert np.array_equal(counts, first + second)
    assert np.array_equal(zeros, np.array([5.0, 2.0, 1.0, 4.0]))
    assert np.array_equal(dep, _deposition_axis().edges)
    assert np.array_equal(primary, _primary_axis().edges)


def test_matrix_merge_rejects_mismatched_primary_axes(tmp_path: Path) -> None:
    counts = np.ones((2, 4))
    _write_matrix_worker(tmp_path / "w0.root", counts, np.zeros(4))
    other = energy_axis(np.linspace(0.0, 500.0, 5), name="primary_energy_kev")
    _write_matrix_worker(
        tmp_path / "w1.root", counts, np.zeros(4), primary_axis=other
    )
    with pytest.raises(errors.ValidationError, match="primary edges differ"):
        runner.merge_matrix_worker_histograms(
            [str(tmp_path / "w0.root"), str(tmp_path / "w1.root")]
        )


def test_matrix_merge_rejects_missing_histograms(tmp_path: Path) -> None:
    path = tmp_path / "empty.root"
    with uproot.recreate(path):
        pass
    with pytest.raises(errors.ValidationError, match="missing the matrix histograms"):
        runner.merge_matrix_worker_histograms([str(path)])
    with pytest.raises(errors.ValidationError, match="no worker files"):
        runner.merge_matrix_worker_histograms([])


def test_spectrum_merge_sums_workers(tmp_path: Path) -> None:
    first = np.arange(8.0)
    second = np.ones(8)
    _write_spectrum_worker(tmp_path / "s0.root", first)
    _write_spectrum_worker(tmp_path / "s1.root", second)
    values, edges = runner.merge_source_worker_histograms(
        [str(tmp_path / "s0.root"), str(tmp_path / "s1.root")]
    )
    assert np.array_equal(values, first + second)
    assert np.array_equal(edges, np.linspace(0.0, 16.0, 9))


def test_spectrum_merge_rejects_mismatched_edges(tmp_path: Path) -> None:
    _write_spectrum_worker(tmp_path / "s0.root", np.ones(8))
    _write_spectrum_worker(tmp_path / "s1.root", np.ones(8), edges=np.linspace(0.0, 8.0, 9))
    with pytest.raises(errors.ValidationError, match="energy edges differ"):
        runner.merge_source_worker_histograms(
            [str(tmp_path / "s0.root"), str(tmp_path / "s1.root")]
        )


def test_estimate_threads_override_and_bounds() -> None:
    assert runner.estimate_threads(10, 10, explicit=3) == 3
    with pytest.raises(errors.UsageError, match="threads must be"):
        runner.estimate_threads(10, 10, explicit=0)
    estimated = runner.estimate_threads(10, 10)
    assert estimated >= 1
    source_estimated = runner.estimate_threads(0, 0, source_mode=True)
    assert source_estimated >= 1
    memory = runner.available_memory_bytes()
    assert memory is None or memory > 0


def test_macro_paths_exist() -> None:
    for name in ("vis.mac", "init_vis.mac", "gui.mac"):
        assert runner.macro_path(name).is_file()
    with pytest.raises(errors.UsageError, match="unknown simulation macro"):
        runner.macro_path("nope.mac")


def test_vis_macro_stores_trajectories() -> None:
    text = runner.macro_path("vis.mac").read_text(encoding="utf-8")
    assert "/tracking/storeTrajectory 1" in text
