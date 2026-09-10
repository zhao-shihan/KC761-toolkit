"""``kc761 subbkg``: DAQ-time scaled background subtraction (W6).

Keeps the pre-rewrite algorithm (DAQ-time scaling, error floor of 1 count,
net spectrum with ``fSumw2``); only the IO layer changes (uproot instead of
the C++ macro).
"""

from __future__ import annotations

import argparse

from kc761.cli._common import add_output_options, add_runtime_options


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "subbkg",
        help="subtract a background spectrum scaled by DAQ time",
        description=(
            "Subtract a background spectrum product from a signal product, "
            "scaled by their daq_time_s values, and write the net spectrum "
            "product (error floor of 1 count, as before)."
        ),
    )
    parser.add_argument(
        "--signal",
        "--sig",
        dest="signal",
        type=str,
        required=True,
        metavar="FILE",
        help="signal (data) spectrum product",
    )
    parser.add_argument(
        "--background",
        "--bkg",
        dest="background",
        type=str,
        required=True,
        metavar="FILE",
        help="background spectrum product with the same channel axis",
    )
    add_output_options(parser, with_plot=False)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 subbkg is implemented in W6")
