"""``kc761`` command-line entry point.

Six frozen subcommands (docs/plan.md D-66): ``calib``, ``unfold``, ``sim``,
``compose``, ``csv2root``, ``subbkg``. Logging, strict mode and the exit-code
policy (0 success / 1 runtime failure / 2 usage error) live here; the
subcommand modules only declare arguments and dispatch to their workstream.
"""

from __future__ import annotations

import argparse
import sys

from kc761 import __version__
from kc761.errors import Kc761Error, UsageError
from kc761.runtime import configure_logging, strict_enabled

SUBCOMMANDS: tuple[str, ...] = ("calib", "unfold", "sim", "compose", "csv2root", "subbkg")


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser (shared by both entry points)."""
    from kc761.cli import calib, compose, csv2root, sim, subbkg, unfold

    parser = argparse.ArgumentParser(
        prog="kc761",
        description=(
            "KC761 gamma-spectrometer toolkit: Geant4 simulation, energy/"
            "resolution calibration and spectrum unfolding."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for module in (calib, unfold, sim, compose, csv2root, subbkg):
        module.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        strict = strict_enabled(args.strict)
        logger = configure_logging(args.command, args.log_level)
    except UsageError as exc:
        print(f"[kc761.{args.command}] error: {exc}", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args, strict=strict))
    except NotImplementedError as exc:
        logger.error("not implemented yet: %s", exc)
        return 1
    except UsageError as exc:
        logger.error("%s", exc)
        return 2
    except Kc761Error as exc:
        logger.error("%s", exc)
        return 1
