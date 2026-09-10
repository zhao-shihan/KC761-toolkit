"""Compose orchestration and artifact contract (F-RESP-2/F-RESP-3, D-114)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kc761.errors import ProvenanceError, SchemaError, ValidationError
from kc761.schema.axes import energy_axis
from kc761.schema.io import input_sha256, read_product, sha256_file
from kc761.schema.products import (
    Histogram1D,
    Histogram2D,
    InputFingerprint,
    Provenance,
)
from kc761.unfold import run_compose
from tests.test_unfold_support import (
    make_calib_product,
    make_sim_product,
    make_spectrum_product,
    primary_edges_kev,
    response_of,
    write,
)


def _inputs(tmp_path: Path):
    calib = make_calib_product()
    sim = make_sim_product(calib)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    return calib, sim, calib_path, sim_path


def test_run_compose_matches_explicit_product(tmp_path: Path) -> None:
    calib, sim, calib_path, sim_path = _inputs(tmp_path)
    result = run_compose(calib_path, sim_path, output=tmp_path / "compose.root", strict=True)
    assert result.product_path is not None
    product = read_product(result.product_path, strict=True)
    expected = response_of(calib, sim)
    assert np.allclose(product.response_matrix.values, expected, rtol=0, atol=1e-12)
    assert np.array_equal(
        product.response_matrix.x.edges, calib.deposition_to_channel.x.edges
    )
    assert np.array_equal(
        product.response_matrix.y.edges, sim.primary_to_deposition.y.edges
    )
    counts = np.asarray(sim.primary_to_deposition.values)
    totals = np.asarray(sim.primary_column_totals.values)
    assert np.allclose(product.primary_efficiency.values, counts.sum(axis=0) / totals)
    assert input_sha256(product.provenance, calib_path) == sha256_file(calib_path)
    assert input_sha256(product.provenance, sim_path) == sha256_file(sim_path)


def test_compose_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _, _, calib_path, sim_path = _inputs(tmp_path)
    output = tmp_path / "compose.root"
    run_compose(calib_path, sim_path, output=output)
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        run_compose(calib_path, sim_path, output=output)
    run_compose(calib_path, sim_path, output=output, force=True)


def test_deposition_axis_mismatch_is_rejected(tmp_path: Path) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    edges = np.asarray(calib.deposition_to_channel.y.edges, dtype=np.float64)
    matrix = Histogram2D(
        x=energy_axis(edges * 1.001, name="deposition_energy_kev"),
        y=sim.primary_to_deposition.y,
        values=np.asarray(sim.primary_to_deposition.values),
        variances=np.asarray(sim.primary_to_deposition.variances),
    )
    broken = replace(sim, primary_to_deposition=matrix)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", broken)
    with pytest.raises(ValidationError, match="deposition axis mismatch"):
        run_compose(calib_path, sim_path)


def test_primary_totals_axis_mismatch_is_rejected(tmp_path: Path) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    shifted = Histogram1D(
        axis=energy_axis(primary_edges_kev() * 1.001, name="primary_energy_kev"),
        values=np.asarray(sim.primary_column_totals.values),
    )
    broken = replace(sim, primary_column_totals=shifted)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", broken)
    with pytest.raises(ValidationError, match="primary axis mismatch"):
        run_compose(calib_path, sim_path)


def test_recorded_input_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    calib_path = write(tmp_path / "calib.root", calib)
    provenance = Provenance(
        created_utc="1970-01-01T00:00:00Z",
        producer="test",
        command="test",
        arguments=(),
        git_revision="0" * 7,
        git_dirty=False,
        python_version="3.12",
        dependency_versions=(),
        inputs=(InputFingerprint(path=str(calib_path), sha256="0" * 64),),
    )
    broken = replace(sim, provenance=provenance)
    sim_path = write(tmp_path / "sim.root", broken)
    with pytest.raises(ProvenanceError, match="digest mismatch"):
        run_compose(calib_path, sim_path)


def test_wrong_product_kind_is_rejected(tmp_path: Path) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    sim_path = write(tmp_path / "sim.root", sim)
    spectrum_path = write(tmp_path / "spectrum.root", make_spectrum_product(np.ones(32)))
    with pytest.raises(SchemaError, match="expected a calib product"):
        run_compose(spectrum_path, sim_path)


def test_default_compose_product_uses_common_meta_only(tmp_path: Path) -> None:
    _, _, calib_path, sim_path = _inputs(tmp_path)
    result = run_compose(calib_path, sim_path, output=tmp_path / "compose.root")
    product = read_product(result.product_path, strict=True)
    assert product.provenance.command == "kc761 compose"
    assert product.provenance.producer == "kc761-compose"
