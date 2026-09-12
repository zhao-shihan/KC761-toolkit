"""F-SPEC-1/F-SPEC-2: spectrum addition and DAQ-time scaled subtraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kc761tool.errors import Kc761toolError, SchemaError, UsageError, ValidationError
from kc761tool.schema.axes import energy_axis
from kc761tool.schema.io import read_product, write_product
from kc761tool.spectra import ADD, SUB, combine_spectra, run_specadd, run_specsub
from tests.fixtures import synthetic


def _write(
    path: Path,
    values: list[float],
    variances: list[float],
    daq_time_s: float,
) -> Path:
    product = synthetic.make_spectrum(values, variances, daq_time_s=daq_time_s)
    return write_product(product, path, force=True)


def test_addition_sums_values_variances_and_daq_times() -> None:
    first = synthetic.make_spectrum([10, 0, 4], [9, 0, 4], daq_time_s=100.0, source_file="a.root")
    second = synthetic.make_spectrum([4, 2, 0], [4, 4, 1], daq_time_s=50.0, source_file="b.root")
    combination = combine_spectra(first, second, operation=ADD)
    # Errors (3, 0, 2) and (2, 2, 1) are floored to (3, 1, 2) and (2, 2, 1).
    assert combination.values.tolist() == [14.0, 2.0, 4.0]
    assert combination.variances.tolist() == [13.0, 5.0, 5.0]
    assert combination.daq_time_s == 150.0
    assert combination.scale == 1.0
    assert combination.source_file("a.root", "b.root") == "a.root + b.root"


def test_addition_is_commutative_in_values_variances_and_daq_time() -> None:
    first = synthetic.make_spectrum([10, 0, 4], [9, 0, 4], daq_time_s=100.0)
    second = synthetic.make_spectrum([4, 2, 0], [4, 4, 1], daq_time_s=50.0)
    forward = combine_spectra(first, second, operation=ADD)
    backward = combine_spectra(second, first, operation=ADD)
    assert forward.values.tolist() == backward.values.tolist()
    assert forward.variances.tolist() == backward.variances.tolist()
    assert forward.daq_time_s == backward.daq_time_s


def test_subtraction_scales_the_second_operand_by_the_daq_ratio() -> None:
    first = synthetic.make_spectrum(
        [10, 0, 4], [10, 1, 4], daq_time_s=100.0, source_file="sig.root"
    )
    second = synthetic.make_spectrum(
        [4, 2, 0], [4, 2, 1], daq_time_s=50.0, source_file="bkg.root"
    )
    combination = combine_spectra(first, second, operation=SUB)
    # r = 100 / 50 = 2 and the zero-error bin of the first operand is floored to 1.
    assert np.allclose(combination.values, [2.0, -4.0, 4.0])
    assert np.allclose(combination.variances, [26.0, 9.0, 8.0])
    assert combination.daq_time_s == 100.0
    assert combination.scale == 2.0
    assert combination.source_file("sig.root", "bkg.root") == "sig.root - bkg.root"


def test_addition_floors_a_zero_error_bin_at_one_count() -> None:
    first = synthetic.make_spectrum([0], [0], daq_time_s=10.0)
    second = synthetic.make_spectrum([0], [0], daq_time_s=10.0)
    combination = combine_spectra(first, second, operation=ADD)
    assert combination.values.tolist() == [0.0]
    assert combination.variances.tolist() == [2.0]


@pytest.mark.parametrize("operation", [ADD, SUB])
def test_channel_axis_mismatch_is_a_validation_error(operation: str) -> None:
    first = synthetic.make_spectrum([1, 2, 3], [1, 2, 3], daq_time_s=10.0)
    second = synthetic.make_spectrum([1, 2, 3, 4], [1, 2, 3, 4], daq_time_s=10.0)
    with pytest.raises(ValidationError, match="binning mismatch"):
        combine_spectra(first, second, operation=operation)


def test_unit_mismatch_is_a_validation_error() -> None:
    energy = energy_axis(np.arange(3, dtype=np.float64))
    first = synthetic.make_spectrum([1, 2], [1, 2], daq_time_s=10.0, axis=energy)
    second = synthetic.make_spectrum([1, 2], [1, 2], daq_time_s=10.0)
    with pytest.raises(ValidationError, match="unit mismatch"):
        combine_spectra(first, second, operation=ADD)


@pytest.mark.parametrize("daq_time_s", [0.0, -1.0, float("nan")])
def test_non_positive_or_non_finite_daq_time_is_a_validation_error(daq_time_s: float) -> None:
    first = synthetic.make_spectrum([1], [1], daq_time_s=10.0)
    second = synthetic.make_spectrum([1], [1], daq_time_s=daq_time_s)
    with pytest.raises(ValidationError, match="daq_time_s"):
        combine_spectra(first, second, operation=SUB)


def test_unknown_operation_is_a_validation_error() -> None:
    operand = synthetic.make_spectrum([1], [1], daq_time_s=10.0)
    with pytest.raises(ValidationError, match="unknown spectrum operation"):
        combine_spectra(operand, operand, operation="multiply")


def test_wrong_product_kind_is_a_schema_error(tmp_path: Path) -> None:
    calib = write_product(synthetic.make_calib_product(), tmp_path / "calib.root", force=True)
    spectrum = _write(tmp_path / "s.root", [1, 2, 3], [1, 2, 3], 10.0)
    with pytest.raises(SchemaError, match="expected a spectrum product"):
        run_specadd(calib, spectrum, output=tmp_path / "out.root")


def test_missing_operand_is_a_runtime_error(tmp_path: Path) -> None:
    spectrum = _write(tmp_path / "s.root", [1], [1], 10.0)
    with pytest.raises(Kc761toolError, match="second spectrum file not found"):
        run_specadd(spectrum, tmp_path / "absent.root", output=tmp_path / "out.root")


def test_output_target_is_validated_before_the_operands_are_read(tmp_path: Path) -> None:
    target = _write(tmp_path / "existing.root", [1], [1], 10.0)
    with pytest.raises(UsageError):
        run_specsub(tmp_path / "absent-a.root", tmp_path / "absent-b.root", output=target)


def test_run_specadd_writes_the_product_with_provenance(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.root", [10, 0], [9, 0], 100.0)
    second = _write(tmp_path / "b.root", [4, 2], [4, 4], 50.0)
    result = run_specadd(
        first,
        second,
        output=tmp_path / "sum.root",
        strict=True,
        arguments=(("specadd", ""), (str(first), "")),
    )
    product = read_product(result.product_path, strict=True)
    assert product.spectrum.values.tolist() == [14.0, 2.0]
    assert product.spectrum.variances.tolist() == [13.0, 5.0]
    assert product.daq_time_s == 150.0
    assert product.source_file == f"{first} + {second}"
    assert product.provenance.producer == "kc761tool-specadd"
    assert product.provenance.command == "kc761tool specadd"
    assert [item.path for item in product.provenance.inputs] == [str(first), str(second)]
    assert result.first_path == first
    assert result.second_path == second


def test_run_specsub_without_output_writes_nothing(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.root", [10, 0], [10, 1], 100.0)
    second = _write(tmp_path / "b.root", [4, 2], [4, 2], 50.0)
    result = run_specsub(first, second)
    assert result.product is None
    assert result.product_path is None
    assert result.combination.values.tolist() == [2.0, -4.0]
    assert {path.name for path in tmp_path.glob("*.root")} == {"a.root", "b.root"}


def test_in_memory_operands_use_their_own_source_file_as_the_audit_label(tmp_path: Path) -> None:
    first = synthetic.make_spectrum([1, 2], [1, 2], daq_time_s=10.0, source_file="one.csv")
    second = synthetic.make_spectrum([1, 2], [1, 2], daq_time_s=10.0, source_file="two.csv")
    result = run_specsub(first, second, output=tmp_path / "net.root")
    product = read_product(result.product_path, strict=True)
    assert result.first_path is None
    assert result.second_path is None
    assert product.source_file == "one.csv - two.csv"
    assert product.provenance.inputs == ()
