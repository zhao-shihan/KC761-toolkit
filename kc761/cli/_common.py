"""Shared CLI option groups.

Naming policy (W0 brief, consistent with docs/plan.md D-66): long names are
primary, pre-rewrite short spellings stay available as aliases for the
options that had them. Window options use ``--energy-low/--elo``,
``--energy-high/--ehi``, ``--channel-low/--chlo`` and
``--channel-high/--chhi``.
"""

from __future__ import annotations

import argparse

from kc761.runtime import LOG_LEVELS


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--strict`` and ``--log-level``."""
    parser.add_argument(
        "--strict",
        action="store_true",
        help="enable all runtime certificate suites (or set KC761_STRICT=1)",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="logging verbosity (default: info)",
    )


def add_output_options(parser: argparse.ArgumentParser, *, with_plot: bool = True) -> None:
    """Add ``-o/--output``, ``-f/--force`` and (optionally) ``--no-plot``.

    ``--no-plot`` is provisional: the plot CLI surface is an open contract
    point (docs/plan.md Appendix A item 4).
    """
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="output product path (default: out/<command>/ under the repository root)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite an existing output file (default: refuse)",
    )
    if with_plot:
        parser.add_argument(
            "--no-plot",
            action="store_true",
            help="skip the plot report (provisional; see plan Appendix A item 4)",
        )


def add_energy_window(parser: argparse.ArgumentParser) -> None:
    """Add the required energy window ``--energy-low/--energy-high``."""
    parser.add_argument(
        "--energy-low",
        "--elo",
        dest="energy_low",
        type=float,
        required=True,
        metavar="KEV",
        help="lower energy bound of the working window in keV",
    )
    parser.add_argument(
        "--energy-high",
        "--ehi",
        dest="energy_high",
        type=float,
        required=True,
        metavar="KEV",
        help="upper energy bound of the working window in keV",
    )


def add_channel_window(parser: argparse.ArgumentParser) -> None:
    """Add the required channel window ``--channel-low/--channel-high``."""
    parser.add_argument(
        "--channel-low",
        "--chlo",
        dest="channel_low",
        type=int,
        required=True,
        metavar="CHANNEL",
        help="lower fit channel (0-based, inclusive)",
    )
    parser.add_argument(
        "--channel-high",
        "--chhi",
        dest="channel_high",
        type=int,
        required=True,
        metavar="CHANNEL",
        help="upper fit channel (0-based, inclusive)",
    )
