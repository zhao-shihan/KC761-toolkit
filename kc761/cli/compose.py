"""``kc761 compose``: build the primary-to-channel response (W2/W4).

Writes the inspection artifact ``R = C . p_tilde . diag(eta)`` from a
calibration product and a matrix-mode simulation product (D-43). The command
is a thin orchestration layer; composition itself is the ``core`` function
F-RESP-2.
"""

from __future__ import annotations

import argparse

from kc761.cli._common import add_output_options, add_runtime_options


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compose",
        help="compose the primary-to-channel response matrix R",
        description=(
            "Compose R = C . p_tilde . diag(eta) over the full primary axis "
            "and write the inspection product (R plus the C/G inputs and the "
            "derived efficiency)."
        ),
    )
    parser.add_argument(
        "--calib",
        type=str,
        required=True,
        metavar="FILE",
        help="calibration product with the deposition-to-channel matrix",
    )
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        metavar="FILE",
        help="matrix-mode simulation product with the primary-to-deposition matrix",
    )
    add_output_options(parser, with_plot=False)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 compose is implemented in W2/W4")
