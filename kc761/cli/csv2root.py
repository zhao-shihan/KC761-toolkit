"""``kc761 csv2root``: strict CSV to spectrum product conversion (W6).

The parser is strict by decision (D-72): malformed headers, duplicated or
non-monotonic channel numbers and column-count mismatches are errors. The
exact header grammar awaits the sample CSV files (docs/plan.md Appendix A
item 8).
"""

from __future__ import annotations

import argparse

from kc761.cli._common import add_output_options, add_runtime_options


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "csv2root",
        help="convert a KC761 MCA CSV export into a spectrum product",
        description=(
            "Convert a KC761 MCA CSV export into a spectrum product "
            "(kc761_spectrum counts, fSumw2 and daq_time_s in the meta "
            "RNTuple). Parsing is strict: malformed input fails with a "
            "schema error."
        ),
    )
    parser.add_argument(
        "input",
        type=str,
        metavar="CSV",
        help="input CSV file (for example bkg-260821-data.csv)",
    )
    add_output_options(parser, with_plot=False)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 csv2root is implemented in W6")
