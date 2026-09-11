"""``kc761 calib``: energy/resolution calibration.

One or more data/MC dataset pairs are fitted with a shared calibration. The
measured spectrum is a ``spectrum`` product and ``--mc`` reads an
``mc_spectrum`` product (D-120/D-144); the library consumes only the histogram,
so the CLI owns product loading and axis checks. Config mode runs a single fit
from ``[[calib.datasets]]`` (D-139).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from kc761.cli._common import (
    add_channel_window,
    add_config_options,
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output,
    reject_run_options,
)
from kc761.cli.config import load_calib_config
from kc761.errors import SchemaError, UsageError
from kc761.runtime import configure_logging
from kc761.schema.io import validate_output_path

_RUN_ARG_DEFAULTS: dict[str, object] = {
    "data": None,
    "mc": None,
    "label": None,
    "syst_frac": None,
    "channel_low": None,
    "channel_high": None,
    "max_iter": None,
    "tolerance": None,
    "output": None,
    "no_plot": False,
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calib",
        help="calibrate energy and resolution from data/MC spectrum pairs",
        description=(
            "Fit the energy calibration and resolution model to one or more "
            "data/simulation dataset pairs, then export the calibration "
            "product. Repeat --data/--mc/--label once per dataset. With "
            "-c/--config one fit is described by [[calib.datasets]]."
        ),
    )
    parser.add_argument(
        "--data",
        dest="data",
        action="append",
        default=None,
        metavar="FILE",
        help="data spectrum product (kc761_spectrum); repeat once per dataset",
    )
    parser.add_argument(
        "--mc",
        dest="mc",
        action="append",
        default=None,
        metavar="FILE",
        help="source-mode simulation product (kc761_mc_spectrum); repeat per dataset",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="NAME",
        help="dataset label (plot titles, scale parameter names); repeat per dataset",
    )
    add_channel_window(parser, required=False)
    parser.add_argument(
        "--syst-frac",
        "--syst",
        dest="syst_frac",
        action="append",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "per-bin fractional systematic uncertainty (0.05 = 5%%); single "
            "value or one per dataset (default 0.05 = 5%%, formula F-CAL-1)"
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="maximum optimizer function evaluations (default: FitSettings, D-107)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        metavar="TOL",
        help="positive convergence tolerance for ftol/xtol/gtol (default: FitSettings)",
    )
    parser.set_defaults(progress_enabled=True)
    parser.add_argument(
        "--progress-every",
        dest="progress_every",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="fit progress line interval in seconds (default 1; 0 = every evaluation)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress_enabled",
        action="store_false",
        help="disable the fit summary and progress lines",
    )
    add_output_options(parser)
    add_config_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    if args.config is not None:
        reject_run_options(args, _RUN_ARG_DEFAULTS, command="calib")
        return _run_config(args, strict=strict)

    if args.data is None or args.mc is None or args.label is None:
        raise UsageError("--data, --mc and --label are required (repeat per dataset)")
    count = len(args.data)
    if len(args.mc) != count or len(args.label) != count:
        raise UsageError(
            "--data, --mc and --label must be repeated the same number of times; got "
            f"{count} data, {len(args.mc)} mc, {len(args.label)} label"
        )
    if args.channel_low is None or args.channel_high is None:
        raise UsageError("--channel-low and --channel-high are required")
    syst = _resolve_syst(args.syst_frac, count)
    settings = _settings(args.max_iter, args.tolerance)
    output = _output(args.output, list(args.label))
    if args.dry_run:
        _print_dry_run(args.data, args.mc, args.label, output)
        return 0
    validate_output_path(output, force=args.force)
    if not args.no_plot:
        validate_output_path(Path(output).with_suffix(".pdf"), force=args.force)
    hints = [(label, args.channel_low, args.channel_high) for label in args.label]
    progress = _progress_printer(hints, settings) if args.progress_enabled else None
    specs = tuple(
        _build_spec(
            data_path=args.data[index],
            mc_path=args.mc[index],
            label=args.label[index],
            channel_low=args.channel_low,
            channel_high=args.channel_high,
            syst_frac=syst[index],
            strict=strict,
        )
        for index in range(count)
    )
    return _execute(
        specs,
        output=output,
        force=args.force,
        no_plot=args.no_plot,
        settings=settings,
        progress=progress,
        progress_every_s=args.progress_every,
        strict=strict,
        logger=configure_logging("calib", args.log_level),
        arguments=argv_arguments(args.argv),
        config_inputs=(),
    )


def _run_config(args: argparse.Namespace, *, strict: bool) -> int:
    from kc761.calib.types import DEFAULT_SYST_FRAC

    config = load_calib_config(args.config, default_syst_frac=DEFAULT_SYST_FRAC)
    output = _output(config.output, [entry.label for entry in config.datasets])
    if args.dry_run or config.dry_run:
        _print_dry_run(
            [str(entry.data) for entry in config.datasets],
            [str(entry.mc) for entry in config.datasets],
            [entry.label for entry in config.datasets],
            output,
        )
        return 0
    force = bool(args.force or config.force)
    validate_output_path(output, force=force)
    if not config.no_plot:
        validate_output_path(Path(output).with_suffix(".pdf"), force=force)
    hints = [
        (entry.label, entry.channel_low, entry.channel_high) for entry in config.datasets
    ]
    progress = _progress_printer(hints, None) if args.progress_enabled else None
    specs = tuple(
        _build_spec(
            data_path=entry.data,
            mc_path=entry.mc,
            label=entry.label,
            channel_low=entry.channel_low,
            channel_high=entry.channel_high,
            syst_frac=entry.syst_frac,
            strict=strict,
        )
        for entry in config.datasets
    )
    return _execute(
        specs,
        output=output,
        force=force,
        no_plot=config.no_plot,
        settings=None,
        progress=progress,
        progress_every_s=args.progress_every,
        strict=strict,
        logger=configure_logging("calib", args.log_level),
        arguments=argv_arguments(args.argv),
        config_inputs=(config.config_path,),
    )


def _resolve_syst(values: list[float] | None, count: int) -> list[float]:
    from kc761.calib.types import DEFAULT_SYST_FRAC

    if values is None:
        return [DEFAULT_SYST_FRAC] * count
    for value in values:
        if not math.isfinite(value) or value < 0.0:
            raise UsageError(f"--syst-frac must be finite and >= 0, got {value!r}")
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return list(values)
    raise UsageError(
        f"--syst-frac must be given once or once per dataset ({count}), got {len(values)}"
    )


def _settings(max_iter: int | None, tolerance: float | None):  # noqa: ANN202
    if max_iter is None and tolerance is None:
        return None
    if max_iter is not None and max_iter < 1:
        raise UsageError(f"--max-iter must be >= 1, got {max_iter!r}")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0.0):
        raise UsageError(f"--tolerance must be positive and finite, got {tolerance!r}")
    from kc761.calib.types import FitSettings

    base = FitSettings()
    return FitSettings(
        maxiter=base.maxiter if max_iter is None else max_iter,
        ftol=base.ftol if tolerance is None else tolerance,
        xtol=base.xtol if tolerance is None else tolerance,
        gtol=base.gtol if tolerance is None else tolerance,
    )


def _build_spec(
    *,
    data_path: str | Path,
    mc_path: str | Path,
    label: str,
    channel_low: int | None,
    channel_high: int | None,
    syst_frac: float,
    strict: bool,
):
    from kc761.calib.types import DatasetSpec
    from kc761.schema.io import read_product
    from kc761.schema.products import McSpectrumProduct, SpectrumProduct

    data_product = read_product(data_path, strict=strict)
    if not isinstance(data_product, SpectrumProduct):
        raise SchemaError(f"{data_path}: expected a spectrum product")
    mc_product = read_product(mc_path, strict=strict)
    if not isinstance(mc_product, McSpectrumProduct):
        raise SchemaError(f"{mc_path}: expected an mc_spectrum product")
    n_bins = data_product.spectrum.axis.n_bins
    low = 0 if channel_low is None else channel_low
    high = n_bins - 1 if channel_high is None else channel_high
    if not 0 <= low <= high < n_bins:
        raise UsageError(
            f"dataset {label!r}: channel window [{low}, {high}] is outside "
            f"[0, {n_bins - 1}]"
        )
    return DatasetSpec(
        label=label,
        data=data_product.spectrum,
        mc=mc_product.spectrum,
        channel_low=low,
        channel_high=high,
        syst_frac=syst_frac,
        data_path=str(data_path),
        mc_path=str(mc_path),
    )


def _output(explicit: str | Path | None, labels: list[str]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    return default_output("calib", "calib-" + "-".join(labels) + ".root")


def _print_dry_run(
    data: list[str], mc: list[str], labels: list[str], output: Path
) -> None:
    print("kc761 calib (dry-run):")
    for label, data_path, mc_path in zip(labels, data, mc, strict=True):
        print(f"  dataset {label}: data={data_path} mc={mc_path}")
    print(f"  output={output}")


def _progress_printer(hints, settings):
    """Return a callback printing the pre-fit summary and periodic progress."""
    from kc761.calib.types import FitProgress, FitSettings

    resolved = settings if settings is not None else FitSettings()

    def printer(event: FitProgress) -> None:
        if event.nfev == 0:
            joined = ", ".join(
                f"{label} "
                f"[{lo if lo is not None else 'full'}-{hi if hi is not None else 'full'}]"
                for label, lo, hi in hints
            )
            print(f"[calib] fitting {event.n_datasets} dataset(s): {joined}", flush=True)
            print(
                f"[calib] free parameters {event.n_free}, fitted bins {event.n_bins}, "
                f"dof {event.dof}",
                flush=True,
            )
            print(
                f"[calib] optimizer maxiter={resolved.maxiter} "
                f"tol=({resolved.ftol:g},{resolved.xtol:g},{resolved.gtol:g})",
                flush=True,
            )
            print(
                f"[calib] initial chi2/dof = {event.chi2:.6g} / {event.dof} "
                f"= {event.reduced_chi2:.6g}",
                flush=True,
            )
            return
        print(
            f"[calib] fit iter {event.nfev}: chi2/dof = {event.chi2:.6g} / "
            f"{event.dof} = {event.reduced_chi2:.6g} "
            f"({event.ms_per_eval:.2f} ms/eval, {event.elapsed_s:.1f}s)",
            flush=True,
        )

    return printer


def _execute(
    specs,
    *,
    output: Path,
    force: bool,
    no_plot: bool,
    settings,
    progress,
    progress_every_s: float,
    strict: bool,
    logger,
    arguments,
    config_inputs,
) -> int:
    from kc761.calib import run_fit

    result = run_fit(
        specs,
        output=output,
        force=force,
        strict=strict,
        settings=settings,
        progress=progress,
        progress_every_s=progress_every_s,
        plot=not no_plot,
        plot_force=force,
        command="kc761 calib",
        arguments=arguments,
        extra_inputs=config_inputs,
    )
    if result.report:
        print(result.report)
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["add_parser"]
