"""``kc761 csv2root``: strict CSV to spectrum product conversion (W6).

Parses the KC761 MCA export ``Channel,Count #<D>d<H>h<M>m<S>s`` (the sample
files in ``data/exp/2609a/``). The parser is strict by decision (D-72/D-130):
a malformed header, an invalid acquisition time, a wrong column count, a
duplicated/non-monotonic/non-contiguous channel or a negative count is an
error. The grammar and ranges are recorded in ``docs/formats.md``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kc761.cli._common import (
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output,
)
from kc761.errors import Kc761Error
from kc761.runtime import configure_logging

_HEADER_RE = re.compile(r"^Channel\s*,\s*Count\s*#(?P<time>.+?)\s*$")
_DURATION_RE = re.compile(
    r"^(?P<days>\d+)d(?P<hours>\d+)h(?P<minutes>\d+)m(?P<seconds>\d+(?:\.\d+)?)s$"
)
_INTEGER_RE = re.compile(r"^\d+$")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "csv2root",
        help="convert a KC761 MCA CSV export into a spectrum product",
        description=(
            "Convert a KC761 MCA CSV export into a spectrum product "
            "(kc761_spectrum counts and fSumw2, daq_time_s in the meta "
            "RNTuple). Parsing is strict: malformed input fails with a "
            "classified error."
        ),
    )
    parser.add_argument(
        "input",
        type=str,
        metavar="CSV",
        help="input CSV file (for example data/exp/2609a/am241.csv)",
    )
    add_output_options(parser, with_plot=False)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _parse_duration(value: str, source: str) -> float:
    match = _DURATION_RE.fullmatch(value)
    if match is None:
        raise Kc761Error(
            f"{source}: malformed acquisition time '#{value}'; expected "
            "<D>d<H>h<M>m<S>s"
        )
    days = int(match.group("days"))
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if hours > 23 or minutes > 59 or seconds >= 60.0:
        raise Kc761Error(
            f"{source}: acquisition time '#{value}' is out of range "
            "(hours <= 23, minutes <= 59, seconds < 60)"
        )
    total_seconds = (days * 24.0 + hours + minutes / 60.0 + seconds / 3600.0) * 3600.0
    if total_seconds <= 0.0:
        raise Kc761Error(f"{source}: acquisition time '#{value}' must be positive")
    return total_seconds


def parse_kc761_csv(
    text: str, *, source: str = "<input>"
) -> tuple[NDArray[np.float64], float]:
    """Parse a KC761 MCA CSV export into ``(counts, daq_time_s)`` (strict)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise Kc761Error(f"{source}: empty CSV file")
    header = lines[index].strip()
    match = _HEADER_RE.fullmatch(header)
    if match is None:
        raise Kc761Error(
            f"{source}: malformed header {header!r}; expected "
            "'Channel,Count #<D>d<H>h<M>m<S>s'"
        )
    daq_time_s = _parse_duration(match.group("time"), source)

    counts: list[float] = []
    for line_number, raw in enumerate(lines[index + 1 :], start=index + 2):
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[2] != "":
            raise Kc761Error(
                f"{source}: line {line_number}: expected the columns "
                f"'channel,count,' (3 fields, empty last), got {line!r}"
            )
        if _INTEGER_RE.fullmatch(fields[0]) is None:
            raise Kc761Error(
                f"{source}: line {line_number}: channel must be a non-negative "
                f"integer, got {fields[0]!r}"
            )
        if _INTEGER_RE.fullmatch(fields[1]) is None:
            raise Kc761Error(
                f"{source}: line {line_number}: count must be a non-negative "
                f"integer, got {fields[1]!r}"
            )
        channel = int(fields[0])
        if channel != len(counts):
            raise Kc761Error(
                f"{source}: line {line_number}: channel {channel} is not the "
                f"expected contiguous value {len(counts)} (channels must start "
                "at 0 and increase by 1)"
            )
        counts.append(float(int(fields[1])))
    if not counts:
        raise Kc761Error(f"{source}: no data rows parsed")
    return np.asarray(counts, dtype=np.float64), daq_time_s


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("csv2root", args.log_level)
    input_path = Path(args.input).expanduser()
    output = (
        Path(args.output).expanduser()
        if args.output is not None
        else default_output("csv2root", input_path.stem + ".root")
    )
    if not input_path.is_file():
        raise Kc761Error(f"input CSV not found: {input_path}")
    try:
        text = input_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Kc761Error(f"{input_path}: not a valid UTF-8 CSV file: {exc}") from exc
    counts, daq_time_s = parse_kc761_csv(text, source=str(input_path))
    from kc761.schema.axes import channel_axis
    from kc761.schema.io import build_provenance, write_product
    from kc761.schema.products import (
        SCHEMA_VERSION,
        Histogram1D,
        SpectrumProduct,
    )

    product = SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=channel_axis(int(counts.size)),
            values=counts,
            variances=counts.copy(),
        ),
        daq_time_s=daq_time_s,
        source_file=str(input_path),
        provenance=build_provenance(
            producer="kc761-csv2root",
            command="kc761 csv2root",
            arguments=argv_arguments(args.argv),
            inputs=[input_path],
        ),
    )
    path = write_product(product, output, force=args.force, strict=strict)
    logger.info("wrote %s (%d channels, daq_time_s=%.6g)", path, counts.size, daq_time_s)
    return 0


__all__ = ["add_parser", "parse_kc761_csv"]
