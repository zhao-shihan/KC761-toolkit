"""``kc761 unfold``: regularized unfolding (W4).

Parses the frozen option surface and raises :class:`NotImplementedError` until
W4 implements the solve. ``--alpha`` is mandatory (D-45); the difference order
option uses ``--difference-order`` with the pre-rewrite ``--k`` alias. The
SNIP mask options are intentionally absent until the open contract point
(docs/plan.md Appendix A item 2) is decided.
"""

from __future__ import annotations

import argparse

from kc761.cli._common import (
    add_energy_window,
    add_output_options,
    add_runtime_options,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "unfold",
        help="unfold a channel spectrum with Tikhonov regularization",
        description=(
            "Compose the primary-to-channel response from a calibration "
            "product and a matrix-mode simulation product, then solve the "
            "non-negative Tikhonov problem and export the unfolded spectrum "
            "with strictly split uncertainty bands. --calib-only relabels the "
            "channel axis to energy without unfolding."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        metavar="FILE",
        help="data spectrum ROOT file (kc761_spectrum)",
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
        default=None,
        metavar="FILE",
        help="matrix-mode simulation product; required unless --calib-only",
    )
    parser.add_argument(
        "--calib-only",
        action="store_true",
        help="relabel the channel axis to energy without unfolding",
    )
    add_energy_window(parser)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        metavar="ALPHA",
        help="dimensionless Tikhonov strength (mandatory; formula F-SOLVE-1)",
    )
    parser.add_argument(
        "--difference-order",
        "--k",
        dest="difference_order",
        type=int,
        choices=(1, 2),
        default=2,
        metavar="K",
        help="difference order of the density penalty (default 2)",
    )
    parser.add_argument(
        "--pad-nsigma",
        type=float,
        default=5.0,
        metavar="N",
        help="working-window padding in local resolution widths (default 5)",
    )
    parser.add_argument(
        "--syst-frac",
        "--syst",
        dest="syst_frac",
        type=float,
        default=0.10,
        metavar="FRAC",
        help="data-side fractional systematic uncertainty (default 0.10)",
    )
    add_output_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    raise NotImplementedError("kc761 unfold is implemented in W4")
