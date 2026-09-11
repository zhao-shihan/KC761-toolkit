"""``kc761`` command-line entry point.

Six frozen subcommands (docs/plan.md D-66) in pipeline order: ``csv2root``,
``subbkg``, ``sim``, ``calib``, ``compose``, ``unfold``. Logging, strict mode
and the exit-code
policy (0 success / 1 runtime failure / 2 usage error) live here; the
subcommand modules only declare arguments and dispatch to their workstream.
"""

from __future__ import annotations

import argparse
import sys

from kc761 import __version__
from kc761.errors import Kc761Error, UsageError
from kc761.runtime import configure_logging, strict_enabled

SUBCOMMANDS: tuple[str, ...] = (
    "csv2root",
    "subbkg",
    "sim",
    "calib",
    "compose",
    "unfold",
)


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
    modules = {
        "calib": calib,
        "compose": compose,
        "csv2root": csv2root,
        "sim": sim,
        "subbkg": subbkg,
        "unfold": unfold,
    }
    for name in SUBCOMMANDS:
        modules[name].add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args.argv = raw_argv
    try:
        strict = strict_enabled(args.strict)
        logger = configure_logging(args.command, args.log_level)
    except UsageError as exc:
        print(f"[kc761.{args.command}] error: {exc}", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args, strict=strict))
    except UsageError as exc:
        logger.error("%s", exc)
        return 2
    except OSError as exc:
        # atomic_write and friends raise OSError for filesystem failures; map
        # them to a runtime failure with an actionable message.
        logger.error("filesystem error: %s", exc)
        return 1
    except Kc761Error as exc:
        logger.error("%s", exc)
        return 1
