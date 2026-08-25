#!/usr/bin/env python3
"""Frontend that assembles the ROOT command line and runs the csv2root.cxx macro.

Converts a KC761 multichannel-analyzer CSV export (e.g. bkg-260821-data.csv)
into a ROOT file containing:
  - TH1D              "kc761_spectrum" : one bin per channel
  - TParameter<double> "daq_time"      : acquisition time in hours

Examples:
    python csv2root.py bkg-260821-data.csv
    python csv2root.py bkg-260821-data.csv -o out.root
    python csv2root.py bkg-260821-data.csv --root /opt/root/6.40.02/bin/root
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kc761util.frontend import add_root_option, run_macro  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a KC761 MCA CSV export into a ROOT file via csv2root.cxx."
    )
    parser.add_argument(
        "input", type=Path,
        help="input CSV file (e.g. bkg-260821-data.csv)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output ROOT file (default: input filename with suffix replaced by .root)",
    )
    add_root_option(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_file = args.input.expanduser().resolve()
    if not input_file.is_file():
        print(f"[csv2root] error: input file not found: {input_file}", file=sys.stderr)
        return 1

    output_file = (input_file.with_suffix(".root")
                   if args.output is None else args.output.expanduser().resolve())

    return run_macro(
        "kc761util/csv2root.cxx",
        [str(input_file), str(output_file)],
        root_exe=args.root,
        cwd=input_file.parent,
        echo_prefix="csv2root",
    )


if __name__ == "__main__":
    sys.exit(main())
