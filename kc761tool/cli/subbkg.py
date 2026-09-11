"""``kc761tool subbkg``: DAQ-time scaled background subtraction.

The net
spectrum is ``S - r B`` with ``r = t_sig / t_bkg``, each input bin error is
floored at one count before subtraction, and the net variance is
``sig_err**2 + r**2 * bkg_err**2``. Inputs and output are ``spectrum`` products;
the output inherits the signal DAQ time and names the signal as its source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kc761tool.cli._common import (
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output_beside,
)
from kc761tool.errors import Kc761toolError, SchemaError, ValidationError
from kc761tool.runtime import configure_logging


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "subbkg",
        help="subtract a background spectrum scaled by DAQ time",
        description=(
            "Subtract a background spectrum product from a signal product, "
            "scaled by their daq_time_s values, and write the net spectrum "
            "product (each input bin error is floored at one count)."
        ),
    )
    parser.add_argument(
        "--signal",
        "--sig",
        dest="signal",
        type=str,
        required=True,
        metavar="FILE",
        help="signal (data) spectrum product",
    )
    parser.add_argument(
        "--background",
        "--bkg",
        dest="background",
        type=str,
        required=True,
        metavar="FILE",
        help="background spectrum product with the same channel axis",
    )
    add_output_options(
        parser,
        with_plot=False,
        default_hint="next to the signal spectrum product",
    )
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _load_spectrum(path: Path, *, strict: bool):
    from kc761tool.schema.io import read_product
    from kc761tool.schema.products import SpectrumProduct

    product = read_product(path, strict=strict)
    if not isinstance(product, SpectrumProduct):
        raise SchemaError(f"{path}: expected a spectrum product")
    return product


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("subbkg", args.log_level)
    signal_path = Path(args.signal).expanduser()
    background_path = Path(args.background).expanduser()
    output = (
        Path(args.output).expanduser()
        if args.output is not None
        else default_output_beside(signal_path, signal_path.stem + "-subbkg.root")
    )
    from kc761tool.schema.io import validate_output_path

    validate_output_path(output, force=args.force)
    for label, path in (("signal", signal_path), ("background", background_path)):
        if not path.is_file():
            raise Kc761toolError(f"{label} file not found: {path}")

    signal = _load_spectrum(signal_path, strict=strict)
    background = _load_spectrum(background_path, strict=strict)
    if signal.spectrum.axis.unit != background.spectrum.axis.unit:
        raise ValidationError(
            "signal/background channel unit mismatch: "
            f"{signal.spectrum.axis.unit!r} vs {background.spectrum.axis.unit!r}"
        )
    if not np.array_equal(signal.spectrum.axis.edges, background.spectrum.axis.edges):
        raise ValidationError(
            "signal/background binning mismatch: the channel axes differ"
        )
    t_signal = float(signal.daq_time_s)
    t_background = float(background.daq_time_s)
    if t_signal <= 0.0 or t_background <= 0.0:
        raise ValidationError(
            "signal and background daq_time_s must be positive "
            f"(got {t_signal!r} and {t_background!r})"
        )
    scale = t_signal / t_background

    signal_values = np.asarray(signal.spectrum.values, dtype=np.float64)
    background_values = np.asarray(background.spectrum.values, dtype=np.float64)
    signal_error = np.sqrt(np.asarray(signal.spectrum.variances, dtype=np.float64))
    background_error = np.sqrt(
        np.asarray(background.spectrum.variances, dtype=np.float64)
    )
    signal_error = np.maximum(signal_error, 1.0)
    background_error = np.maximum(background_error, 1.0)
    values = signal_values - scale * background_values
    variances = signal_error**2 + scale**2 * background_error**2

    from kc761tool.schema.io import build_provenance, write_product
    from kc761tool.schema.products import SCHEMA_VERSION, Histogram1D, SpectrumProduct

    product = SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=signal.spectrum.axis, values=values, variances=variances
        ),
        daq_time_s=t_signal,
        source_file=str(signal_path),
        provenance=build_provenance(
            producer="kc761tool-subbkg",
            command="kc761tool subbkg",
            arguments=argv_arguments(args.argv),
            inputs=[signal_path, background_path],
        ),
    )
    path = write_product(product, output, force=args.force, strict=strict)
    logger.info(
        "wrote %s (%d bins, scale r = %.6g)", path, values.size, scale
    )
    return 0


__all__ = ["add_parser"]
