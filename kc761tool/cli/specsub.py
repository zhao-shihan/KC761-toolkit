"""``kc761tool specsub``: DAQ-time scaled background subtraction (F-SPEC-2).

The net spectrum is ``A - r B`` with ``r = t_A / t_B``, where the second
positional operand is the background spectrum to subtract; each input bin error
is floored at one count before subtraction and the net variance is
``err_A**2 + r**2 * err_B**2``. Both operands and the output are ``spectrum``
products; the output inherits the first DAQ time. The combination lives in
:mod:`kc761tool.spectra`; this module only declares the operands, resolves the
default output path and prints the result (D-185).
"""

from __future__ import annotations

import argparse

from kc761tool.cli._common import (
    add_spectrum_operands,
    argv_arguments,
    resolve_combination_paths,
)
from kc761tool.cli._registry import (
    RunPolicy,
    add_run_options,
    merge_options,
    output_options,
    resolve_run_options,
    runtime_options,
)
from kc761tool.runtime import configure_logging
from kc761tool.spectra import SUB, run_specsub

#: Declared option surface (D-190); the two operands stay argparse-native.
SPECSUB_POLICY = RunPolicy(
    command="specsub",
    spec=merge_options(
        output_options(
            with_plot=False,
            default_hint="next to the first spectrum product (SPECTRUM_A)",
        ),
        runtime_options(),
    ),
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "specsub",
        help="subtract a background spectrum scaled by DAQ time",
        description=(
            "Subtract the second spectrum product (background) from the first, "
            "scaled by their daq_time_s values, and write the net spectrum "
            "product (each input bin error is floored at one count)."
        ),
    )
    add_spectrum_operands(
        parser,
        first_help="spectrum product the background is subtracted from",
        second_help="background spectrum product, scaled by t_first / t_second",
    )
    add_run_options(parser, SPECSUB_POLICY)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    values = resolve_run_options(SPECSUB_POLICY, args)
    logger = configure_logging("specsub", str(values["log_level"]))
    first, second, output = resolve_combination_paths(args, token=SUB, output=values["output"])
    result = run_specsub(
        first,
        second,
        output=output,
        force=bool(values["force"]),
        strict=strict,
        arguments=argv_arguments(args.argv),
    )
    logger.info(
        "wrote %s (%d bins, scale r = %.6g)",
        result.product_path,
        result.combination.values.size,
        result.combination.scale,
    )
    return 0


__all__ = ["add_parser"]
