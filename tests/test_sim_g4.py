"""W5 Geant4 integration tests (marker ``g4``; skipped without Geant4).

Each simulation run happens in a spawned subprocess (one Geant4 run manager per
process), so these tests exercise the real runner, merge and product output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kc761.core.binning import (
    SOURCE_MODE_DEPOSITION_BINS,
    source_mode_deposition_edges_kev,
)
from kc761.schema import io
from kc761.schema.products import McSpectrumProduct, SimProduct
from kc761.sim import DEFAULT_SEED, certificates, runner
from tests.fixtures import synthetic

pytestmark = [
    pytest.mark.g4,
    pytest.mark.skipif(
        importlib.util.find_spec("geant4_pybind") is None,
        reason="Geant4 (geant4_pybind) is not installed",
    ),
]


def _write_calib(tmp_path: Path) -> Path:
    return io.write_product(
        synthetic.make_calib_product(), tmp_path / "calib.root", force=True, strict=True
    )


def _read_sim(path: Path) -> SimProduct:
    product = io.read_product(path, strict=True)
    assert isinstance(product, SimProduct)
    return product


def test_matrix_run_writes_a_strict_product(tmp_path: Path) -> None:
    calib = _write_calib(tmp_path)
    path = runner.run_matrix(
        "plane-front-gamma",
        calib,
        tmp_path / "G.root",
        n_events=128,
        seed=DEFAULT_SEED,
        threads=1,
        strict=True,
    )
    product = _read_sim(path)
    counts = product.primary_to_deposition.values
    assert counts.shape == (24, SOURCE_MODE_DEPOSITION_BINS)
    assert product.mode == 1
    assert product.geometry_name == "plane"
    assert product.n_events == 128
    assert int(product.primary_column_totals.values.sum()) == 128
    assert np.all(counts >= 0.0)
    assert np.array_equal(
        product.primary_to_deposition.y.edges, source_mode_deposition_edges_kev()
    )
    # F-SIM-6: no entry deposits more energy than its primary column carries.
    certificates.verify_physical_boundary(
        counts,
        product.primary_to_deposition.x.edges,
        product.primary_to_deposition.y.edges,
        strict=True,
    )


def test_matrix_same_seed_is_bitwise_reproducible(tmp_path: Path) -> None:
    calib = _write_calib(tmp_path)
    first = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "a.root", n_events=96, seed=11, threads=1
    )
    second = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "b.root", n_events=96, seed=11, threads=1
    )
    a = _read_sim(first)
    b = _read_sim(second)
    assert np.array_equal(a.primary_to_deposition.values, b.primary_to_deposition.values)
    assert np.array_equal(
        a.primary_column_totals.values, b.primary_column_totals.values
    )


def test_matrix_two_workers_match_one_worker(tmp_path: Path) -> None:
    calib = _write_calib(tmp_path)
    # 2176 events = one full 2048-column slice plus one short slice, so both
    # workers actually receive events.
    single = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "one.root", n_events=2176, seed=3, threads=1
    )
    split = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "two.root", n_events=2176, seed=3, threads=2
    )
    a = _read_sim(single)
    b = _read_sim(split)
    assert np.array_equal(a.primary_to_deposition.values, b.primary_to_deposition.values)
    assert np.array_equal(a.primary_column_totals.values, b.primary_column_totals.values)
    assert b.workers == 2


def test_source_run_writes_a_strict_mc_spectrum(tmp_path: Path) -> None:
    path = runner.run_source(
        "am241", tmp_path / "mc.root", n_events=400, seed=5, threads=1, strict=True
    )
    product = io.read_product(path, strict=True)
    assert isinstance(product, McSpectrumProduct)
    assert product.source_key == "am241"
    assert product.spectrum.axis.n_bins == SOURCE_MODE_DEPOSITION_BINS
    assert product.spectrum.axis.unit == "kev"
    variances = product.spectrum.variances
    assert variances is not None
    assert np.all(variances >= 0.0)


def test_source_same_seed_is_bitwise_reproducible(tmp_path: Path) -> None:
    first = runner.run_source(
        "am241", tmp_path / "a.root", n_events=160, seed=9, threads=1
    )
    second = runner.run_source(
        "am241", tmp_path / "b.root", n_events=160, seed=9, threads=1
    )
    a = io.read_product(first)
    b = io.read_product(second)
    assert isinstance(a, McSpectrumProduct)
    assert isinstance(b, McSpectrumProduct)
    assert np.array_equal(a.spectrum.values, b.spectrum.values)
    assert np.array_equal(a.spectrum.variances, b.spectrum.variances)
