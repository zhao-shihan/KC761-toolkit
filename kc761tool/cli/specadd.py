"""``kc761tool specadd``: sum of two experimental spectra (F-SPEC-1).

The sum is bin by bin: ``values = A + B``, ``variances = err_A**2 + err_B**2``
with each input bin error floored at one count, and the two ``daq_time_s``
values add, so the summed product is equivalent to one longer acquisition of the
same source. Both operands and the output are ``spectrum`` products. The
combination itself lives in :mod:`kc761tool.spectra`; this module only declares
the arguments, resolves the default output name and reports the result (D-185).
"""

from __future__ import annotations

import argparse

from kc761tool.cli._common import (
    add_output_options,
    add_runtime_options,
    add_spectrum_operands,
    argv_arguments,
    resolve_combination_paths,
)
from kc761tool.runtime import configure_logging
from kc761tool.spectra import ADD, run_specadd


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "specadd",
        help="add two spectrum products and add their DAQ times",
        description=(
            "Add two spectrum products bin by bin, floor each input bin error "
            "at one count, add their daq_time_s values and write the summed "
            "spectrum product."
        ),
    )
    add_spectrum_operands(
        parser,
        first_help="first spectrum product",
        second_help="second spectrum product",
    )
    add_output_options(
        parser,
        with_plot=False,
        default_hint="next to the first spectrum product (SPECTRUM_A)",
    )
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("specadd", args.log_level)
    first, second, output = resolve_combination_paths(args, token=ADD)
    result = run_specadd(
        first,
        second,
        output=output,
        force=args.force,
        strict=strict,
        arguments=argv_arguments(args.argv),
    )
    logger.info(
        "wrote %s (%d bins, daq_time_s = %.6g)",
        result.product_path,
        result.combination.values.size,
        result.combination.daq_time_s,
    )
    return 0


__all__ = ["add_parser"]
