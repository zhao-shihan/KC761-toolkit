"""``kc761 sim``: Geant4 simulation (W5).

Parses the frozen option surface and raises :class:`NotImplementedError`
until W5 implements the runner. Source keys mirror the pre-rewrite registry
(docs/plan.md Appendix A item 6); the matrix modes take the calibration
product as their argument, as before.
"""

from __future__ import annotations

import argparse

from kc761.cli._common import add_output_options, add_runtime_options
from kc761.sim import SOURCE_KEYS

DEFAULT_SEED = 908136382
"""Pre-rewrite base seed; W5 owns the final value (worker i uses seed + i + 1)."""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sim",
        help="run the Geant4 simulation (source mode or matrix modes)",
        description=(
            "Simulate either a radioactive source in its container (batch, or "
            "interactive when --events is omitted; the per-event ntuple was "
            "removed in the rewrite) or one of the matrix modes with a "
            "calibration product."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    for key in SOURCE_KEYS:
        source.add_argument(
            f"--{key}",
            dest="source_key",
            action="store_const",
            const=key,
            help=f"simulate the {key} source",
        )
    source.add_argument(
        "--plane-front-gamma",
        dest="plane_front_gamma",
        metavar="CALIB",
        help="matrix mode: plane source at the housing front surface",
    )
    source.add_argument(
        "--sphere-gamma",
        dest="sphere_gamma",
        metavar="CALIB",
        help="matrix mode: circumscribed-sphere source",
    )
    parser.add_argument(
        "-n",
        "--events",
        type=int,
        default=None,
        metavar="N",
        help="number of events; omit for an interactive session (source mode only)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="worker processes (default: memory-budgeted CPU count, D-38)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="SEED",
        help=f"base random seed (default {DEFAULT_SEED}); worker i uses SEED + i + 1",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase Geant4 verbosity (repeatable)",
    )
    add_output_options(parser, with_plot=False)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 sim is implemented in W5")
