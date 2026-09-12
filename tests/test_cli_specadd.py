"""``kc761tool specadd``: positional operands, summed DAQ time, default naming."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kc761tool.cli import main
from kc761tool.schema.io import read_product, write_product
from tests.fixtures import synthetic


def _write_spectrum(path: Path, values, variances, daq_time_s: float) -> Path:
    product = synthetic.make_spectrum(values, variances, daq_time_s=daq_time_s)
    return write_product(product, path, force=True)


def test_specadd_sums_values_variances_and_daq_times(tmp_path: Path) -> None:
    first = _write_spectrum(tmp_path / "run1.root", [10, 0, 4], [9, 0, 4], 100.0)
    second = _write_spectrum(tmp_path / "run2.root", [4, 2, 0], [4, 4, 1], 50.0)
    output = tmp_path / "sum.root"
    assert main(["specadd", str(first), str(second), "-o", str(output)]) == 0
    product = read_product(output, strict=True)
    assert np.allclose(product.spectrum.values, [14.0, 2.0, 4.0])
    assert np.allclose(product.spectrum.variances, [13.0, 5.0, 5.0])
    assert product.daq_time_s == 150.0
    assert product.source_file == f"{first} + {second}"
    assert product.provenance.producer == "kc761tool-specadd"
    assert product.provenance.command == "kc761tool specadd"


def test_specadd_default_output_is_the_combined_name(tmp_path: Path) -> None:
    first = _write_spectrum(tmp_path / "run1.root", [4], [4], 10.0)
    second = _write_spectrum(tmp_path / "run2.root", [1], [1], 20.0)
    assert main(["specadd", str(first), str(second)]) == 0
    expected = tmp_path / "run1-add-run2.root"
    assert expected.is_file()
    product = read_product(expected, strict=True)
    assert product.daq_time_s == 30.0


def test_specadd_rejects_axis_mismatch(tmp_path: Path) -> None:
    first = _write_spectrum(tmp_path / "a.root", [1, 2, 3], [1, 2, 3], 10.0)
    second = _write_spectrum(tmp_path / "b.root", [1, 2, 3, 4], [1, 2, 3, 4], 10.0)
    assert main(["specadd", str(first), str(second), "-o", str(tmp_path / "s.root")]) == 1


def test_specadd_missing_file_is_runtime_failure(tmp_path: Path) -> None:
    first = _write_spectrum(tmp_path / "a.root", [1], [1], 10.0)
    assert (
        main(
            [
                "specadd",
                str(first),
                str(tmp_path / "absent.root"),
                "-o",
                str(tmp_path / "sum.root"),
            ]
        )
        == 1
    )
