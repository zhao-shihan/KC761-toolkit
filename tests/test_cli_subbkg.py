"""subbkg semantics: DAQ-time scaling, error floor, net variance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kc761tool.cli import main
from kc761tool.schema.axes import channel_axis
from kc761tool.schema.io import read_product, write_product
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    Histogram1D,
    Provenance,
    SpectrumProduct,
)
from tests.fixtures import synthetic


def _write_spectrum(path: Path, values, variances, daq_time_s: float) -> Path:
    product = SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=channel_axis(len(values)),
            values=np.asarray(values, dtype=np.float64),
            variances=np.asarray(variances, dtype=np.float64),
        ),
        daq_time_s=daq_time_s,
        source_file="fixture.csv",
        provenance=Provenance(
            created_utc="2026-09-10T00:00:00Z",
            producer="test",
            command="test",
            arguments=(),
            git_revision="unknown",
            git_dirty=False,
            python_version="3",
            dependency_versions=(),
            inputs=(),
        ),
    )
    return write_product(product, path, force=True)


def test_subbkg_scales_and_floors_errors(tmp_path: Path) -> None:
    signal = _write_spectrum(tmp_path / "sig.root", [10, 0, 4], [10, 1, 4], 100.0)
    background = _write_spectrum(tmp_path / "bkg.root", [4, 2, 0], [4, 2, 1], 50.0)
    output = tmp_path / "net.root"
    assert (
        main(
            [
                "subbkg",
                "--signal",
                str(signal),
                "--background",
                str(background),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    product = read_product(output, strict=True)
    # r = 100 / 50 = 2.
    assert np.allclose(product.spectrum.values, [2.0, -4.0, 4.0])
    assert np.allclose(product.spectrum.variances, [26.0, 9.0, 8.0])
    assert product.daq_time_s == 100.0
    assert product.source_file == str(signal)


def test_subbkg_error_floor_of_one(tmp_path: Path) -> None:
    signal = _write_spectrum(tmp_path / "sig.root", [0], [0], 10.0)
    background = _write_spectrum(tmp_path / "bkg.root", [0], [0], 10.0)
    output = tmp_path / "net.root"
    assert (
        main(
            [
                "subbkg",
                "--signal",
                str(signal),
                "--background",
                str(background),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    product = read_product(output, strict=True)
    # Both zero-bin errors are floored to 1, r = 1 -> variance 1 + 1 = 2.
    assert product.spectrum.values.tolist() == [0.0]
    assert product.spectrum.variances.tolist() == [2.0]


def test_subbkg_rejects_axis_mismatch(tmp_path: Path) -> None:
    signal = _write_spectrum(tmp_path / "sig.root", [1, 2, 3], [1, 2, 3], 10.0)
    background = _write_spectrum(tmp_path / "bkg.root", [1, 2, 3, 4], [1, 2, 3, 4], 10.0)
    assert (
        main(
            [
                "subbkg",
                "--signal",
                str(signal),
                "--background",
                str(background),
                "-o",
                str(tmp_path / "net.root"),
            ]
        )
        == 1
    )


def test_subbkg_rejects_wrong_product_kind(tmp_path: Path) -> None:
    signal = _write_spectrum(tmp_path / "sig.root", [1, 2, 3], [1, 2, 3], 10.0)
    calib = write_product(
        synthetic.make_calib_product(), tmp_path / "calib.root", force=True
    )
    assert (
        main(
            [
                "subbkg",
                "--signal",
                str(calib),
                "--background",
                str(signal),
                "-o",
                str(tmp_path / "net.root"),
            ]
        )
        == 1
    )


def test_subbkg_missing_file_is_runtime_failure(tmp_path: Path) -> None:
    signal = _write_spectrum(tmp_path / "sig.root", [1], [1], 10.0)
    assert (
        main(
            [
                "subbkg",
                "--signal",
                str(signal),
                "--background",
                str(tmp_path / "absent.root"),
                "-o",
                str(tmp_path / "net.root"),
            ]
        )
        == 1
    )
