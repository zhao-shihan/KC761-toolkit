"""``kc761 unfold``: regularized unfolding (W6).

Full mode composes the response from a calibration product and a matrix-mode
simulation product, solves the non-negative Tikhonov problem and exports the
unfolded spectrum with strictly split uncertainty bands. ``--calib-only``
relabels the channel axis to energy and needs neither ``--sim``, ``--alpha``
nor the energy window (D-139). Config mode runs one unfold (D-139).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from kc761.cli._common import (
    add_config_options,
    add_energy_window,
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output,
    reject_run_options,
)
from kc761.cli.config import UnfoldConfig, load_unfold_config
from kc761.core.solver import (
    DEFAULT_SNIP_FLOOR,
    DEFAULT_SNIP_MAX_ITERATIONS,
    DEFAULT_SNIP_PROTECT_SIGMA,
    DEFAULT_SNIP_THRESHOLD_SIGMA,
)
from kc761.core.uncertainty import DEFAULT_SYST_FRAC
from kc761.errors import UsageError
from kc761.runtime import configure_logging
from kc761.schema.io import validate_output_path

_RUN_ARG_DEFAULTS: dict[str, object] = {
    "data": None,
    "calib": None,
    "sim": None,
    "calib_only": False,
    "energy_low": None,
    "energy_high": None,
    "alpha": None,
    "difference_order": 2,
    "pad_nsigma": 5.0,
    "syst_frac": DEFAULT_SYST_FRAC,
    "output": None,
    "no_plot": False,
}

_CALIB_ONLY_BOUNDS = (0.0, 1.0)
"""Placeholder bounds for the calib-only path, which does not build a window."""


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
        default=None,
        metavar="FILE",
        help="data spectrum product (kc761_spectrum)",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
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
    add_energy_window(parser, required=False)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        metavar="ALPHA",
        help="dimensionless Tikhonov strength (required unless --calib-only)",
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
        default=DEFAULT_SYST_FRAC,
        metavar="FRAC",
        help=f"data-side fractional systematic uncertainty (default {DEFAULT_SYST_FRAC:g})",
    )
    parser.set_defaults(snip_enabled=True)
    parser.add_argument(
        "--snip",
        dest="snip_enabled",
        action="store_true",
        help="enable the SNIP peak mask (default)",
    )
    parser.add_argument(
        "--no-snip",
        dest="snip_enabled",
        action="store_false",
        help="disable the SNIP peak mask",
    )
    parser.add_argument(
        "--snip-threshold",
        dest="snip_threshold_sigma",
        type=float,
        default=DEFAULT_SNIP_THRESHOLD_SIGMA,
        metavar="SIGMA",
        help="peak significance threshold in sigma (default 5)",
    )
    parser.add_argument(
        "--snip-protect",
        dest="snip_protect_sigma",
        type=float,
        default=DEFAULT_SNIP_PROTECT_SIGMA,
        metavar="SIGMA",
        help="protective half-width around a peak in resolution sigmas (default 2)",
    )
    parser.add_argument(
        "--snip-floor",
        dest="snip_floor",
        type=float,
        default=DEFAULT_SNIP_FLOOR,
        metavar="W",
        help="penalty weight floor on protected peak bins (default 0.1)",
    )
    parser.add_argument(
        "--snip-iterations",
        dest="snip_iterations",
        type=int,
        default=None,
        metavar="M",
        help="SNIP iteration count override (default: resolution-derived)",
    )
    parser.add_argument(
        "--snip-max-iterations",
        dest="snip_max_iterations",
        type=int,
        default=DEFAULT_SNIP_MAX_ITERATIONS,
        metavar="M",
        help="cap for the resolution-derived SNIP iteration count (default 8)",
    )
    add_output_options(parser)
    add_config_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("unfold", args.log_level)
    if args.config is not None:
        reject_run_options(args, _RUN_ARG_DEFAULTS, command="unfold")
        config = load_unfold_config(args.config, default_syst_frac=DEFAULT_SYST_FRAC)
        output = _output(
            config.output, Path(config.data), config.sim, config.alpha, config.calib_only
        )
        if args.dry_run or config.dry_run:
            _print_dry_run(config, output)
            return 0
        _validate_outputs(
            output, bool(args.force or config.force), plot=not config.no_plot
        )
        return _execute(
            data=config.data,
            calib=config.calib,
            sim=config.sim,
            calib_only=config.calib_only,
            energy_low=config.energy_low_kev,
            energy_high=config.energy_high_kev,
            alpha=config.alpha,
            difference_order=config.difference_order,
            pad_nsigma=config.pad_nsigma,
            syst_frac=config.syst_frac,
            snip_enabled=config.snip_enabled,
            snip_threshold_sigma=config.snip_threshold_sigma,
            snip_protect_sigma=config.snip_protect_sigma,
            snip_floor=config.snip_floor,
            snip_iterations=config.snip_iterations,
            snip_max_iterations=config.snip_max_iterations,
            output=output,
            force=bool(args.force or config.force),
            no_plot=config.no_plot,
            strict=strict,
            logger=logger,
            arguments=argv_arguments(args.argv),
            config_inputs=(config.config_path,),
        )

    if args.data is None or args.calib is None:
        raise UsageError("--data and --calib are required")
    if args.calib_only:
        _reject_for_calib_only(args)
        sim = None
        energy_low, energy_high = _CALIB_ONLY_BOUNDS
        alpha = None
    else:
        if args.sim is None:
            raise UsageError("--sim is required unless --calib-only")
        if args.alpha is None:
            raise UsageError("--alpha is required for the full unfold")
        _validate_alpha(args.alpha)
        if args.energy_low is None or args.energy_high is None:
            raise UsageError("--energy-low and --energy-high are required")
        if not args.energy_low < args.energy_high:
            raise UsageError(
                f"--energy-low ({args.energy_low}) must be < --energy-high "
                f"({args.energy_high})"
            )
        sim = args.sim
        energy_low, energy_high = args.energy_low, args.energy_high
        alpha = args.alpha
    _validate_pad(args.pad_nsigma)
    _validate_syst(args.syst_frac)
    output = _output(
        args.output, Path(args.data), sim, alpha, args.calib_only
    )
    if args.dry_run:
        _print_args_dry_run(args, output)
        return 0
    _validate_outputs(output, args.force, plot=not args.no_plot)
    return _execute(
        data=args.data,
        calib=args.calib,
        sim=sim,
        calib_only=args.calib_only,
        energy_low=energy_low,
        energy_high=energy_high,
        alpha=alpha,
        difference_order=args.difference_order,
        pad_nsigma=args.pad_nsigma,
        syst_frac=args.syst_frac,
        snip_enabled=args.snip_enabled,
        snip_threshold_sigma=args.snip_threshold_sigma,
        snip_protect_sigma=args.snip_protect_sigma,
        snip_floor=args.snip_floor,
        snip_iterations=args.snip_iterations,
        snip_max_iterations=args.snip_max_iterations,
        output=output,
        force=args.force,
        no_plot=args.no_plot,
        strict=strict,
        logger=logger,
        arguments=argv_arguments(args.argv),
        config_inputs=(),
    )


def _reject_for_calib_only(args: argparse.Namespace) -> None:
    if args.sim is not None:
        raise UsageError("--sim is not used with --calib-only")
    if args.alpha is not None:
        raise UsageError("--alpha is not used with --calib-only")
    if args.energy_low is not None or args.energy_high is not None:
        raise UsageError("--energy-low/--energy-high are not used with --calib-only")


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise UsageError(f"--alpha must be positive and finite, got {alpha!r}")


def _validate_pad(pad_nsigma: float) -> None:
    if not math.isfinite(pad_nsigma) or pad_nsigma < 0.0:
        raise UsageError(f"--pad-nsigma must be finite and >= 0, got {pad_nsigma!r}")


def _validate_syst(syst_frac: float) -> None:
    if not math.isfinite(syst_frac) or syst_frac < 0.0:
        raise UsageError(f"--syst-frac must be finite and >= 0, got {syst_frac!r}")


def _output(
    explicit: str | Path | None,
    data: Path,
    sim: str | Path | None,
    alpha: float | None,
    calib_only: bool,
) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    if calib_only:
        return default_output("unfold", f"unfold-{data.stem}-calibonly.root")
    assert sim is not None and alpha is not None
    return default_output(
        "unfold", f"unfold-{data.stem}-{Path(sim).stem}-a{alpha:g}.root"
    )


def _print_dry_run(config: UnfoldConfig, output: Path) -> None:
    print("kc761 unfold (dry-run):")
    print(f"  mode={'calib_only' if config.calib_only else 'unfold'}")
    print(f"  data={config.data}")
    print(f"  calib={config.calib}")
    print(f"  sim={config.sim}")
    print(
        f"  alpha={config.alpha} difference_order={config.difference_order} "
        f"pad_nsigma={config.pad_nsigma} syst_frac={config.syst_frac}"
    )
    print(f"  output={output}")


def _print_args_dry_run(args: argparse.Namespace, output: Path) -> None:
    print("kc761 unfold (dry-run):")
    print(f"  mode={'calib_only' if args.calib_only else 'unfold'}")
    print(f"  data={args.data}")
    print(f"  calib={args.calib}")
    print(f"  sim={args.sim}")
    print(
        f"  alpha={args.alpha} difference_order={args.difference_order} "
        f"pad_nsigma={args.pad_nsigma} syst_frac={args.syst_frac}"
    )
    print(f"  output={output}")


def _validate_outputs(output: Path, force: bool, *, plot: bool) -> None:
    """Fail fast on an unusable product or figure target (D-171)."""
    validate_output_path(output, force=force)
    if plot:
        validate_output_path(Path(output).with_suffix(".pdf"), force=force)


def _execute(
    *,
    data,
    calib,
    sim,
    calib_only: bool,
    energy_low: float,
    energy_high: float,
    alpha: float | None,
    difference_order: int,
    pad_nsigma: float,
    syst_frac: float,
    snip_enabled: bool,
    snip_threshold_sigma: float,
    snip_protect_sigma: float,
    snip_floor: float,
    snip_iterations: int | None,
    snip_max_iterations: int,
    output: Path,
    force: bool,
    no_plot: bool,
    strict: bool,
    logger,
    arguments,
    config_inputs,
) -> int:
    from kc761.unfold import run_unfold

    result = run_unfold(
        data,
        calib,
        sim,
        energy_low_kev=energy_low,
        energy_high_kev=energy_high,
        alpha=alpha,
        difference_order=difference_order,
        pad_nsigma=pad_nsigma,
        syst_frac=syst_frac,
        snip_enabled=snip_enabled,
        snip_threshold_sigma=snip_threshold_sigma,
        snip_protect_sigma=snip_protect_sigma,
        snip_floor=snip_floor,
        snip_iterations=snip_iterations,
        snip_max_iterations=snip_max_iterations,
        calib_only=calib_only,
        output=output,
        force=force,
        strict=strict,
        command="kc761 unfold",
        arguments=arguments,
        plot=not no_plot,
        plot_force=force,
        extra_inputs=config_inputs,
    )
    if result.report:
        print(result.report)
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["add_parser"]
