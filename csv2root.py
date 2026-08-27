#!/usr/bin/env python3
"""Convert a KC761 MCA CSV export into a ROOT file via csv2root.cxx."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        print(
            f"[csv2root] error: input file not found: {input_file}", file=sys.stderr)
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
