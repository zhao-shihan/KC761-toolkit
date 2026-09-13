"""Shared CLI option groups.

Naming policy: long names are primary; short spellings stay available as
aliases for the options that had them. Window options use ``--energy-low/--elo``,
``--energy-high/--ehi``, ``--channel-low/--chlo`` and
``--channel-high/--chhi``.

This module also owns the cross-command plumbing that must not be duplicated:
the repository-root output convention (D-19), full-argv provenance pairs
(D-18) and the combined default output name of the two-operand spectrum
commands (D-185). The option declarations themselves, including the
config/run-option mutual exclusion of D-130, live in
:mod:`kc761tool.cli._registry` (D-190).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kc761tool.errors import UsageError

#: Repository root derived from this file (``kc761tool/cli/_common.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Internal options that must not leak into the recorded provenance arguments.
_HIDDEN_ARGUMENTS = frozenset({"--provenance-input"})


def default_output(command: str, filename: str) -> Path:
    """Return the D-19 default product path ``work/<command>/<filename>``."""
    if not filename or Path(filename).name != filename:
        raise UsageError(f"invalid default output name {filename!r} for {command}")
    return REPO_ROOT / "work" / command / filename


def default_output_beside(source: str | Path, filename: str) -> Path:
    """Return ``<source directory>/<filename>`` (D-165).

    Used by the commands whose default product lives next to the input file
    (``csv2root`` next to the CSV, ``specadd``/``specsub`` next to the first
    spectrum product).
    """
    if not filename or Path(filename).name != filename:
        raise UsageError(f"invalid default output name {filename!r}")
    return Path(source).expanduser().parent / filename


def combined_output_name(first: str | Path, second: str | Path, *, token: str) -> str:
    """Return ``<first-stem>-<token>-<second-stem>.root`` (D-185).

    The token is the operation name (``add``/``sub``) shared with
    :mod:`kc761tool.spectra`, so the connector of the combined product name and
    the operation name have a single definition.
    """
    first_stem = Path(first).stem
    second_stem = Path(second).stem
    if not first_stem or not second_stem:
        raise UsageError(f"invalid operand names {str(first)!r} and {str(second)!r}")
    return f"{first_stem}-{token}-{second_stem}.root"


def resolve_combination_paths(
    args: argparse.Namespace,
    *,
    token: str,
    output: object = None,
) -> tuple[Path, Path, Path]:
    """Resolve the two positional operands and the default product path (D-185).

    The default output lives next to the first operand and is named
    ``<first-stem>-<token>-<second-stem>.root``; the two-operand spectrum
    commands share this resolution. ``output`` is the resolved ``--output``
    value (D-190), ``None`` for the default name.
    """
    first = Path(args.first).expanduser()
    second = Path(args.second).expanduser()
    target = (
        Path(str(output)).expanduser()
        if output is not None
        else default_output_beside(first, combined_output_name(first, second, token=token))
    )
    return first, second, target


def add_spectrum_operands(
    parser: argparse.ArgumentParser,
    *,
    first_help: str,
    second_help: str,
) -> None:
    """Add the two positional spectrum operands of ``specadd``/``specsub`` (D-185)."""
    parser.add_argument("first", metavar="SPECTRUM_A", type=str, help=first_help)
    parser.add_argument("second", metavar="SPECTRUM_B", type=str, help=second_help)


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
