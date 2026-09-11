"""Strict csv2root parser and product output (D-72, Appendix A item 8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kc761tool.cli import main
from kc761tool.cli.csv2root import parse_kc761_csv
from kc761tool.errors import Kc761toolError
from kc761tool.schema.io import read_product

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "data"
SMALL_CSV = FIXTURES / "kc761_small.csv"

VALID = "Channel,Count #0d0h0m10s\n0,3,\n1,0,\n2,5,\n"


def test_parse_valid_small_csv() -> None:
    counts, daq_time_s = parse_kc761_csv(VALID)
    assert counts.tolist() == [3.0, 0.0, 5.0]
    assert daq_time_s == 10.0


def test_parse_skips_blank_lines_and_handles_crlf() -> None:
    text = VALID.replace("\n", "\r\n") + "\r\n"
    counts, _ = parse_kc761_csv(text)
    assert counts.size == 3


@pytest.mark.parametrize(
    "bad",
    [
        "Channels,Count #0d0h0m10s\n0,1,\n",
        "Channel,Counts #0d0h0m10s\n0,1,\n",
        "Channel,Count 0d0h0m10s\n0,1,\n",
        "Channel,Count #0d0h0m10\n0,1,\n",
        "Channel,Count #0d0h0m70s\n0,1,\n",
        "Channel,Count #0d0h60m0s\n0,1,\n",
        "Channel,Count #0d0h0m0s\n0,1,\n",
    ],
)
def test_bad_header_or_time_is_an_error(bad: str) -> None:
    with pytest.raises(Kc761toolError):
        parse_kc761_csv(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "Channel,Count #0d0h0m10s\n0,1\n",  # two fields
        "Channel,Count #0d0h0m10s\n0,1,0\n",  # non-empty third field
        "Channel,Count #0d0h0m10s\n0,1,x,\n",  # four fields
        "Channel,Count #0d0h0m10s\n0,-1,\n",  # negative count
        "Channel,Count #0d0h0m10s\n0,1.5,\n",  # non-integer count
        "Channel,Count #0d0h0m10s\n-1,1,\n",  # negative channel
    ],
)
def test_bad_data_line_is_an_error(bad: str) -> None:
    with pytest.raises(Kc761toolError):
        parse_kc761_csv(bad)


def test_non_contiguous_channels_are_an_error() -> None:
    with pytest.raises(Kc761toolError, match="contiguous"):
        parse_kc761_csv("Channel,Count #0d0h0m10s\n0,1,\n2,1,\n")
    with pytest.raises(Kc761toolError, match="contiguous"):
        parse_kc761_csv("Channel,Count #0d0h0m10s\n0,1,\n0,2,\n")


def test_empty_file_is_an_error() -> None:
    with pytest.raises(Kc761toolError, match="empty"):
        parse_kc761_csv("")


def test_cli_writes_a_verified_spectrum(tmp_path: Path) -> None:
    output = tmp_path / "small.root"
    assert main(["csv2root", str(SMALL_CSV), "-o", str(output)]) == 0
    product = read_product(output, strict=True)
    assert product.spectrum.axis.n_bins == 8
    assert float(product.spectrum.values.sum()) == 30.0
    assert product.daq_time_s == 10.0
    assert product.source_file.endswith("kc761_small.csv")
    assert np.array_equal(product.spectrum.variances, product.spectrum.values)


def test_cli_reports_malformed_csv(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("nope\n0,1,\n", encoding="utf-8")
    assert main(["csv2root", str(bad), "-o", str(tmp_path / "bad.root")]) == 1
    assert not (tmp_path / "bad.root").exists()
