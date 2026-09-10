"""``kc761 calib``: energy/resolution calibration (W3).

Parses the frozen option surface and raises :class:`NotImplementedError` until
W3 implements the fit. The data-spectrum option keeps the pre-rewrite spelling
``--sim``; a rename is an open contract point (docs/plan.md Appendix A item 3).
"""

from __future__ import annotations

import argparse

from kc761.cli._common import (
    add_channel_window,
    add_output_options,
    add_runtime_options,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calib",
        help="calibrate energy and resolution from data/MC spectrum pairs",
        description=(
            "Fit the energy calibration and resolution model to one or more "
            "data/simulation dataset pairs, then export the calibration "
            "product. Repeat --data/--sim/--channel-low/--channel-high/--label "
            "once per dataset."
        ),
    )
    parser.add_argument(
        "--data",
        dest="data",
        action="append",
        required=True,
        metavar="FILE",
        help="data spectrum ROOT file; repeat once per dataset",
    )
    parser.add_argument(
        "--sim",
        dest="sim",
        action="append",
        required=True,
        metavar="FILE",
        help="simulated source spectrum ROOT file; repeat once per dataset",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        metavar="NAME",
        help="dataset label (plot titles, scale parameter names); repeat per dataset",
    )
    parser.add_argument(
        "--syst-frac",
        "--syst",
        dest="syst_frac",
        action="append",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "per-bin fractional systematic uncertainty (0.05 = 5%%); single "
            "value or one per dataset (default 0.10, formula F-CAL-1)"
        ),
    )
    add_channel_window(parser)
    add_output_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 calib is implemented in W3")
