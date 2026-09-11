"""Schema IO tests: atomic write, overwrite policy, provenance and round-tripping.

These tests are auxiliary (D-65): they codify the F-IO-1 protocol but do not
define correctness. Every product is written, reopened and compared field by
field, including variances and provenance JSON.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kc761.errors import Kc761Error, ProvenanceError, SchemaError
from kc761.schema import io
from kc761.schema.products import (
    OBJECT_NAMES,
    TRACKED_DEPENDENCIES,
    CalibProduct,
    ComposeProduct,
    Histogram1D,
    Histogram2D,
    Product,
    SimProduct,
    SpectrumProduct,
    UnfoldProduct,
    dispatch_key_for,
)
from tests.fixtures import synthetic

BUILDERS: dict[str, object] = {
    "calib": synthetic.make_calib_product,
    "sim": synthetic.make_sim_product,
    "compose": synthetic.make_compose_product,
    "unfold": synthetic.make_unfold_product,
    "unfold_calib_only": synthetic.make_unfold_calib_only_product,
    "spectrum": synthetic.make_spectrum_product,
}


def _axes_equal(left, right) -> bool:
    return left.name == right.name and left.unit == right.unit and np.array_equal(
        left.edges, right.edges
    )


def _hist1d_equal(left: Histogram1D | None, right: Histogram1D | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not _axes_equal(left.axis, right.axis) or not np.array_equal(left.values, right.values):
        return False
    if left.variances is None or right.variances is None:
        return left.variances is None and right.variances is None
    return np.array_equal(left.variances, right.variances)


def _hist2d_equal(left: Histogram2D, right: Histogram2D) -> bool:
    if not _axes_equal(left.x, right.x) or not _axes_equal(left.y, right.y):
        return False
    if not np.array_equal(left.values, right.values):
        return False
    if left.variances is None or right.variances is None:
        return left.variances is None and right.variances is None
    return np.array_equal(left.variances, right.variances)


def assert_product_equal(original: Product, reloaded: Product) -> None:
    assert type(original) is type(reloaded)
    assert original.format_version == reloaded.format_version
    assert original.provenance == reloaded.provenance
    if isinstance(original, CalibProduct):
        assert isinstance(reloaded, CalibProduct)
        assert _hist2d_equal(original.deposition_to_channel, reloaded.deposition_to_channel)
        assert _hist2d_equal(original.param_cov, reloaded.param_cov)
        assert original.params_reported == reloaded.params_reported
        assert original.resol_params == reloaded.resol_params
        assert original.channel_max == reloaded.channel_max
    elif isinstance(original, SimProduct):
        assert isinstance(reloaded, SimProduct)
        assert _hist2d_equal(original.primary_to_deposition, reloaded.primary_to_deposition)
        assert _hist1d_equal(original.primary_column_totals, reloaded.primary_column_totals)
        assert (
            original.mode,
            original.mode_name,
            original.geometry_name,
            original.geometry_param_mm,
            original.angular_distribution,
            original.seed,
            original.n_events,
            original.workers,
        ) == (
            reloaded.mode,
            reloaded.mode_name,
            reloaded.geometry_name,
            reloaded.geometry_param_mm,
            reloaded.angular_distribution,
            reloaded.seed,
            reloaded.n_events,
            reloaded.workers,
        )
    elif isinstance(original, ComposeProduct):
        assert isinstance(reloaded, ComposeProduct)
        for name in ("response_matrix", "deposition_to_channel", "primary_to_deposition"):
            assert _hist2d_equal(getattr(original, name), getattr(reloaded, name))
        assert _hist1d_equal(original.primary_column_totals, reloaded.primary_column_totals)
        assert _hist1d_equal(original.primary_efficiency, reloaded.primary_efficiency)
    elif isinstance(original, UnfoldProduct):
        assert isinstance(reloaded, UnfoldProduct)
        assert original.mode == reloaded.mode
        assert _hist1d_equal(original.spectrum, reloaded.spectrum)
        for name in ("sigma_statistical", "sigma_systematic", "sigma_total", "refolded"):
            assert _hist1d_equal(getattr(original, name), getattr(reloaded, name))
        assert original.settings == reloaded.settings
    elif isinstance(original, SpectrumProduct):
        assert isinstance(reloaded, SpectrumProduct)
        assert _hist1d_equal(original.spectrum, reloaded.spectrum)
        assert original.daq_time_s == reloaded.daq_time_s
        assert original.source_file == reloaded.source_file


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_roundtrip(tmp_path: Path, name: str) -> None:
    product = BUILDERS[name]()
    path = io.write_product(product, tmp_path / f"{name}.root", force=True)
    reloaded = io.read_product(path)
    assert_product_equal(product, reloaded)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_roundtrip_strict(tmp_path: Path, name: str) -> None:
    product = BUILDERS[name]()
    path = io.write_product(product, tmp_path / f"{name}.root", force=True, strict=True)
    reloaded = io.read_product(path, strict=True)
    io.verify_product(path, strict=True)
    assert_product_equal(product, reloaded)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_write_refuses_overwrite_and_force_succeeds(tmp_path: Path, name: str) -> None:
    product = BUILDERS[name]()
    path = tmp_path / f"{name}.root"
    io.write_product(product, path)
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        io.write_product(product, path)
    io.write_product(product, path, force=True)
    io.verify_product(path)


def test_parent_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "product.root"
    io.write_product(synthetic.make_spectrum_product(), target)
    assert target.is_file()
    io.verify_product(target)


def test_stale_part_is_replaced(tmp_path: Path) -> None:
    target = tmp_path / "product.root"
    stale = target.with_name(target.name + io.PART_SUFFIX)
    stale.write_bytes(b"stale")
    io.write_product(synthetic.make_spectrum_product(), target)
    assert target.is_file()
    assert not stale.exists()


def test_atomic_write_failure_removes_part_and_preserves_target(tmp_path: Path) -> None:
    target = tmp_path / "product.root"
    original = synthetic.make_spectrum_product()
    io.write_product(original, target)
    assert_product_equal(original, io.read_product(target))

    def _boom(_: Path) -> None:
        raise SchemaError("injected verification failure")

    with pytest.raises(SchemaError, match="injected"):
        io.atomic_write(
            target,
            writer=lambda part: part.write_bytes(b"partial"),
            verify=_boom,
        )
    assert not target.with_name(target.name + io.PART_SUFFIX).exists()
    assert_product_equal(original, io.read_product(target))


def test_read_missing_file_raises_classified_error(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="no such product file"):
        io.read_product(tmp_path / "absent.root")


def test_build_provenance_hashes_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(b"channel,counts\n0,1\n")
    provenance = io.build_provenance(
        producer="test",
        command="kc761 test",
        arguments=(("a", "1"), ("b", "2")),
        inputs=[source],
    )
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance.inputs[0].sha256 == expected
    assert io.input_sha256(provenance, source) == expected
    assert io.input_sha256(provenance, source.resolve()) == expected
    assert io.input_sha256(provenance, tmp_path / "other.csv") is None
    assert tuple(name for name, _ in provenance.dependency_versions) == TRACKED_DEPENDENCIES
    assert provenance.arguments == (("a", "1"), ("b", "2"))


def test_build_provenance_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="input file not found"):
        io.build_provenance(
            producer="test",
            command="kc761 test",
            arguments=(),
            inputs=[tmp_path / "absent.csv"],
        )


def test_build_provenance_non_git_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(io.subprocess, "run", _fail)
    with pytest.warns(RuntimeWarning, match="git metadata unavailable"):
        provenance = io.build_provenance(
            producer="test", command="kc761 test", arguments=(), inputs=[]
        )
    assert provenance.git_revision == "unknown"
    assert provenance.git_dirty is False


def test_provenance_arguments_survive_roundtrip(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    custom = io.build_provenance(
        producer="test-producer",
        command="kc761 spectrum --flag value",
        arguments=(("z", "last"), ("a", "first"), ("m", "middle")),
        inputs=[],
    )
    product = type(product)(
        format_version=product.format_version,
        spectrum=product.spectrum,
        daq_time_s=product.daq_time_s,
        source_file=product.source_file,
        provenance=custom,
    )
    path = io.write_product(product, tmp_path / "spectrum.root")
    reloaded = io.read_product(path)
    assert isinstance(reloaded, SpectrumProduct)
    assert reloaded.provenance.command == custom.command
    assert reloaded.provenance.producer == custom.producer
    assert reloaded.provenance.arguments == custom.arguments
    assert reloaded.provenance.dependency_versions == custom.dependency_versions


def test_write_rejects_nonfinite_values(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    bad = Histogram1D(
        axis=product.spectrum.axis,
        values=np.full(product.spectrum.values.shape, np.nan),
        variances=product.spectrum.variances,
    )
    bad_product = SpectrumProduct(
        format_version=product.format_version,
        spectrum=bad,
        daq_time_s=product.daq_time_s,
        source_file=product.source_file,
        provenance=product.provenance,
    )
    with pytest.raises(Kc761Error):
        io.write_product(bad_product, tmp_path / "nan.root")


def test_write_rejects_shape_mismatch(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    wrong = Histogram1D(
        axis=product.spectrum.axis,
        values=np.zeros(product.spectrum.values.size - 1),
        variances=None,
    )
    bad_product = SpectrumProduct(
        format_version=product.format_version,
        spectrum=wrong,
        daq_time_s=product.daq_time_s,
        source_file=product.source_file,
        provenance=product.provenance,
    )
    with pytest.raises(SchemaError, match="does not match axis"):
        io.write_product(bad_product, tmp_path / "shape.root")


def test_required_variance_is_enforced(tmp_path: Path) -> None:
    product = synthetic.make_sim_product()
    stripped = Histogram2D(
        x=product.primary_to_deposition.x,
        y=product.primary_to_deposition.y,
        values=product.primary_to_deposition.values,
        variances=None,
    )
    bad_product = SimProduct(
        format_version=product.format_version,
        primary_to_deposition=stripped,
        primary_column_totals=product.primary_column_totals,
        mode=product.mode,
        mode_name=product.mode_name,
        geometry_name=product.geometry_name,
        geometry_param_mm=product.geometry_param_mm,
        angular_distribution=product.angular_distribution,
        seed=product.seed,
        n_events=product.n_events,
        workers=product.workers,
        provenance=product.provenance,
    )
    with pytest.raises(SchemaError, match="missing required fSumw2"):
        io.write_product(bad_product, tmp_path / "novar.root")


def test_object_names_cover_all_builders() -> None:
    for name in BUILDERS:
        product = BUILDERS[name]()
        assert dispatch_key_for(product) in OBJECT_NAMES


def test_regular_product_has_no_part_left(tmp_path: Path) -> None:
    path = io.write_product(synthetic.make_spectrum_product(), tmp_path / "clean.root")
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(io.PART_SUFFIX)]
    assert leftovers == []
    assert path.is_file()


def test_human_titles_and_machine_axis_names(tmp_path: Path) -> None:
    """D-170: ROOT titles are human-readable; fName keeps the machine name."""
    import uproot

    from kc761.schema.axes import human_axis_title
    from kc761.schema.products import HUMAN_TITLES, OBJ_SPECTRUM

    product = synthetic.make_spectrum_product()
    path = tmp_path / "spectrum.root"
    io.write_product(product, path)
    with uproot.open(path) as file:
        hist = file[OBJ_SPECTRUM]
        assert hist.member("fTitle") == HUMAN_TITLES[OBJ_SPECTRUM]
        axis = hist.axis(0)
        assert axis.member("fName") == product.spectrum.axis.name
        assert axis.member("fTitle") == human_axis_title(product.spectrum.axis)
