"""Shared CLI option groups.

Naming policy: long names are primary; short spellings stay available as
aliases for the options that had them. Window options use ``--energy-low/--elo``,
``--energy-high/--ehi``, ``--channel-low/--chlo`` and
``--channel-high/--chhi``.

This module also owns the cross-command plumbing that must not be duplicated:
the repository-root output convention (D-19), full-argv provenance pairs
(D-18) and the config/run-option mutual-exclusion check (D-130).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from kc761.errors import UsageError
from kc761.runtime import LOG_LEVELS

#: Repository root derived from this file (``kc761/cli/_common.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Internal options that must not leak into the recorded provenance arguments.
_HIDDEN_ARGUMENTS = frozenset({"--provenance-input"})


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


def add_output_options(
    parser: argparse.ArgumentParser,
    *,
    with_plot: bool = True,
    default_hint: str | None = None,
) -> None:
    """Add ``-o/--output``, ``-f/--force`` and (optionally) ``--no-plot``.

    ``--no-plot`` is the frozen plot surface of D-142: plots are produced by
    default and suppressed with this single switch. ``default_hint`` overrides
    the help text for commands whose default output is not ``work/<command>/``
    (D-165: csv2root and subbkg write next to their input).
    """
    hint = default_hint or "work/<command>/ under the repository root"
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help=f"output product path (default: {hint})",
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
            help="skip the plot report (plots are produced by default)",
        )


def add_config_options(parser: argparse.ArgumentParser) -> None:
    """Add ``-c/--config`` and ``--dry-run`` (D-129/D-130)."""
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="FILE",
        help=(
            "run from a TOML configuration file; mutually exclusive with the "
            "run-selection options (only --strict/--log-level/--dry-run/--force "
            "may accompany it)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved invocation and exit without side effects",
    )


def default_output(command: str, filename: str) -> Path:
    """Return the D-19 default product path ``work/<command>/<filename>``."""
    if not filename or Path(filename).name != filename:
        raise UsageError(f"invalid default output name {filename!r} for {command}")
    return REPO_ROOT / "work" / command / filename


def default_output_beside(source: str | Path, filename: str) -> Path:
    """Return ``<source directory>/<filename>`` (D-165).

    Used by the commands whose default product lives next to the input file
    (``csv2root`` next to the CSV, ``subbkg`` next to the signal spectrum).
    """
    if not filename or Path(filename).name != filename:
        raise UsageError(f"invalid default output name {filename!r}")
    return Path(source).expanduser().parent / filename


def argv_arguments(argv: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Split a full argv into ordered ``(option, value)`` provenance pairs.

    Flag values that begin with ``-`` are recorded with an empty value, so a
    boolean flag never swallows the next option. Internal options are dropped.
    """
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _HIDDEN_ARGUMENTS:
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if token.startswith("-") and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            pairs.append((token, argv[index + 1]))
            index += 2
        else:
            pairs.append((token, ""))
            index += 1
    return tuple(pairs)


def reject_run_options(
    args: argparse.Namespace,
    run_defaults: Mapping[str, object],
    *,
    command: str,
) -> None:
    """Enforce D-130: config mode only accepts the global options."""
    offending = sorted(
        dest for dest, default in run_defaults.items() if getattr(args, dest, None) != default
    )
    if offending:
        rendered = ", ".join(f"--{dest.replace('_', '-')}" for dest in offending)
        raise UsageError(
            f"kc761 {command}: --config is mutually exclusive with the "
            f"run-selection options; remove: {rendered}"
        )


def add_energy_window(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the energy window ``--energy-low/--energy-high``.

    ``required`` is True for the calibration command; the unfold command makes
    the window conditional (``calib_only`` ignores it) and validates it in the
    handler (D-139).
    """
    parser.add_argument(
        "--energy-low",
        "--elo",
        dest="energy_low",
        type=float,
        required=required,
        default=None,
        metavar="KEV",
        help="lower energy bound of the working window in keV",
    )
    parser.add_argument(
        "--energy-high",
        "--ehi",
        dest="energy_high",
        type=float,
        required=required,
        default=None,
        metavar="KEV",
        help="upper energy bound of the working window in keV",
    )


def add_channel_window(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the channel window ``--channel-low/--channel-high``.

    ``required`` is True for the single-run calibration command; config mode
    may leave the window optional and the handler validates it (D-139).
    """
    parser.add_argument(
        "--channel-low",
        "--chlo",
        dest="channel_low",
        type=int,
        required=required,
        default=None,
        metavar="CHANNEL",
        help="lower fit channel (0-based, inclusive)",
    )
    parser.add_argument(
        "--channel-high",
        "--chhi",
        dest="channel_high",
        type=int,
        required=required,
        default=None,
        metavar="CHANNEL",
        help="upper fit channel (0-based, inclusive)",
    )
