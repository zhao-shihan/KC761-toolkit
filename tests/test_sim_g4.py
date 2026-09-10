"""W5 Geant4 integration tests (marker ``g4``; skipped without Geant4).

Each simulation run happens in a spawned subprocess (one Geant4 run manager per
process), so these tests exercise the real runner, merge and product output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kc761.core.binning import SOURCE_MODE_DEPOSITION_BINS
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
    c_edges = np.asarray(synthetic.make_calib_product().deposition_to_channel.y.edges)
    # D-121 revised: primary and deposition are both C.y, so G is square.
    assert counts.shape == (c_edges.size - 1, c_edges.size - 1)
    assert product.mode == 1
    assert product.mode_name == "plane_front_gamma"
    assert product.geometry_name == "plane"
    assert product.n_events == 128
    assert int(product.primary_column_totals.values.sum()) == 128
    assert np.all(counts >= 0.0)
    # Non-vacuous: the scoring path must actually record deposits.
    assert float(counts.sum()) > 0.0
    assert np.array_equal(product.primary_to_deposition.x.edges, c_edges)
    assert np.array_equal(product.primary_to_deposition.y.edges, c_edges)
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
    # Stronger than the D-123 contract: matrix columns carry independent
    # streams, so the merged product is partition-independent in practice.
    calib = _write_calib(tmp_path)
    # 48 events over the 24 columns: the two workers split the columns 12/12.
    single = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "one.root", n_events=48, seed=3, threads=1
    )
    split = runner.run_matrix(
        "plane-front-gamma", calib, tmp_path / "two.root", n_events=48, seed=3, threads=2
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
    # Non-vacuous: the source-mode scoring path must record deposits.
    assert float(product.spectrum.values.sum()) > 0.0


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


def test_source_two_workers_same_partition_is_reproducible(tmp_path: Path) -> None:
    # D-123 (revised): same seed + same worker partition is bit-for-bit.
    first = runner.run_source(
        "am241", tmp_path / "one.root", n_events=2048, seed=7, threads=2
    )
    second = runner.run_source(
        "am241", tmp_path / "two.root", n_events=2048, seed=7, threads=2
    )
    a = io.read_product(first)
    b = io.read_product(second)
    assert isinstance(a, McSpectrumProduct)
    assert isinstance(b, McSpectrumProduct)
    assert np.array_equal(a.spectrum.values, b.spectrum.values)
    assert np.array_equal(a.spectrum.variances, b.spectrum.variances)
    assert b.workers == 2


def test_matrix_folds_out_of_range_deposits_into_zero(tmp_path: Path) -> None:
    # Decision 1 makes the deposition axis channel-derived, so its first edge
    # can exceed 0 keV; out-of-range deposits must be undetected (zero) rather
    # than lost to G4 under/overflow, or F-SIM-1 fails.
    from kc761.calib.product import build_calib_product
    from kc761.schema.io import build_provenance

    provenance = build_provenance(
        producer="test", command="test", arguments=(), inputs=[]
    )
    calib = build_calib_product(
        core_internal=np.array([60.0, 5.0, 5.0, 5.0]),
        resol_params=np.array([2.0, 1.0, 0.0]),
        param_cov=np.diag([0.25, 1e-4, 1e-6, 1e-8, 0.04, 0.09, 0.16]),
        chi2=1.0,
        dof=1,
        covariance_scale=1.0,
        fit_status="converged",
        scales=(),
        scale_bound_flags=(),
        n_channels=8,
        channel_max=7.0,
        provenance=provenance,
        strict=True,
    )
    calib_path = io.write_product(
        calib, tmp_path / "calib_positive.root", force=True, strict=True
    )
    path = runner.run_matrix(
        "plane-front-gamma",
        calib_path,
        tmp_path / "G_positive.root",
        n_events=64,
        seed=12345,
        threads=1,
        strict=True,
    )
    product = _read_sim(path)
    counts = product.primary_to_deposition.values
    totals = product.primary_column_totals.values
    assert counts.shape == (8, 8)
    assert np.all(counts >= 0.0)
    # strict=True already ran F-SIM-1 (sum(counts)+zero == N_j); the folded
    # out-of-range deposits make the product non-vacuous without underflow.
    assert float(counts.sum()) > 0.0
    assert np.all(counts.sum(axis=0) <= totals + 1e-9)
