"""``kc761tool unfold``: regularized unfolding.

Full mode composes the response from a calibration product and a matrix-mode
simulation product, solves the non-negative Tikhonov problem and exports the
unfolded spectrum with strictly split uncertainty bands. ``--calib-only``
relabels the channel axis to energy and needs neither ``--sim``, ``--alpha``
nor the energy window (D-139). Config mode runs one unfold (D-139).

The option surface is declared once in :data:`UNFOLD_POLICY` (D-190): the
command line and the ``[unfold]`` table share the flags, the TOML keys, the
defaults, the requirement rules and the validation, which is delegated to
:class:`kc761tool.unfold.types.UnfoldSettings` so the CLI and the library can
never disagree about what a valid invocation is.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from kc761tool.cli._common import argv_arguments, default_output
from kc761tool.cli._registry import (
    Kind,
    Requirement,
    RunOption,
    RunPolicy,
    add_run_options,
    config_options,
    energy_window_options,
    merge_options,
    output_options,
    resolve_run_options,
    runtime_options,
)
from kc761tool.cli.config import load_unfold_config
from kc761tool.core.solver import (
    DEFAULT_SNIP_FLOOR,
    DEFAULT_SNIP_MAX_ITERATIONS,
    DEFAULT_SNIP_PROTECT_BINS,
    DEFAULT_SNIP_THRESHOLD_SIGMA,
)
from kc761tool.core.uncertainty import DEFAULT_SYST_FRAC
from kc761tool.errors import UsageError, ValidationError
from kc761tool.runtime import configure_logging
from kc761tool.schema.io import validate_output_path
from kc761tool.unfold.types import UnfoldSettings

_CALIB_ONLY_BOUNDS = (0.0, 1.0)
"""Placeholder bounds for the calib-only path, which does not build a window."""

_CALIB_ONLY_UNUSED = ("sim", "alpha", "energy_low", "energy_high")
"""Options that ``--calib-only`` does not use (D-139)."""


def _validate_unfold(
    values: Mapping[str, object], present: frozenset[str], config_mode: bool
) -> None:
    """Cross-field rules, evaluated on the resolved mapping of either surface.

    Validating by constructing :class:`UnfoldSettings` means the CLI and the
    TOML table enforce exactly the bounds the library enforces (D-190).
    """
    calib_only = bool(values["calib_only"])
    if calib_only:
        banned = sorted(name for name in _CALIB_ONLY_UNUSED if name in present)
        if banned:
            rendered = ", ".join(
                f"'{name}'" if config_mode else f"--{name.replace('_', '-')}" for name in banned
            )
            raise UsageError(f"calib_only does not use {rendered}; remove it/them")
        energy_low, energy_high = _CALIB_ONLY_BOUNDS
        alpha = None
    else:
        energy_low = float(values["energy_low"])
        energy_high = float(values["energy_high"])
        alpha = values["alpha"]
    try:
        UnfoldSettings(
            energy_low_kev=energy_low,
            energy_high_kev=energy_high,
            alpha=alpha,
            difference_order=int(values["difference_order"]),
            syst_frac=float(values["syst_frac"]),
            snip_enabled=bool(values["snip_enabled"]),
            snip_threshold_sigma=float(values["snip_threshold_sigma"]),
            snip_protect_bins=int(values["snip_protect_bins"]),
            snip_floor=float(values["snip_floor"]),
            snip_iterations=values["snip_iterations"],
            snip_max_iterations=int(values["snip_max_iterations"]),
        )
    except ValidationError as exc:
        raise UsageError(f"invalid setting: {exc}") from exc


def _window_required(values: Mapping[str, object]) -> bool:
    """``--sim``/``--alpha``/the window are required unless ``--calib-only``."""
    return not bool(values["calib_only"])


#: Declared surface (D-190): flags, defaults, TOML keys and the shared rules.
UNFOLD_POLICY = RunPolicy(
    command="unfold",
    config=True,
    # Retired TOML keys get the same migration pointer as the retired command
    # line spelling (D-188/D-190).
    retired_keys=(
        (
            "pad_nsigma",
            "was removed in D-187: the unfold solve space is the reported window, "
            "so there is no padding to configure",
        ),
        (
            "snip_protect",
            "was renamed to 'snip_protect_bins' in D-188; the half-width unit "
            "changed from resolution sigmas to primary bins",
        ),
        (
            "snip_protect_sigma",
            "was renamed to 'snip_protect_bins' in D-188; the half-width unit "
            "changed from resolution sigmas to primary bins",
        ),
    ),
    validate=_validate_unfold,
    spec=merge_options(
        [
            RunOption(
                dest="data",
                flags=("--data",),
                kind=Kind.PATH,
                metavar="FILE",
                config_key="data",
                requirement=Requirement.ALWAYS,
                help="data spectrum product (kc761_spectrum)",
            ),
            RunOption(
                dest="calib",
                flags=("--calib",),
                kind=Kind.PATH,
                metavar="FILE",
                config_key="calib",
                requirement=Requirement.ALWAYS,
                help="calibration product with the deposition-to-channel matrix",
            ),
            RunOption(
                dest="sim",
                flags=("--sim",),
                kind=Kind.PATH,
                metavar="FILE",
                config_key="sim",
                requirement=Requirement.ALWAYS,
                required_if=_window_required,
                help="matrix-mode simulation product; required unless --calib-only",
            ),
            RunOption(
                dest="calib_only",
                flags=("--calib-only",),
                kind=Kind.BOOL,
                default=False,
                config_key="calib_only",
                help="relabel the channel axis to energy without unfolding",
            ),
            *energy_window_options(requirement=Requirement.ALWAYS, required_if=_window_required),
            RunOption(
                dest="alpha",
                flags=("--alpha",),
                kind=Kind.FLOAT,
                metavar="ALPHA",
                config_key="alpha",
                requirement=Requirement.ALWAYS,
                required_if=_window_required,
                help="dimensionless Tikhonov strength (required unless --calib-only)",
            ),
            RunOption(
                dest="difference_order",
                flags=("--difference-order", "--k"),
                kind=Kind.INT,
                default=2,
                metavar="K",
                config_key="difference_order",
                help="difference order of the density penalty (default 2)",
            ),
            RunOption(
                dest="syst_frac",
                flags=("--syst-frac", "--syst"),
                kind=Kind.FLOAT,
                default=DEFAULT_SYST_FRAC,
                metavar="FRAC",
                config_key="syst_frac",
                help=(
                    f"data-side fractional systematic uncertainty (default {DEFAULT_SYST_FRAC:g})"
                ),
            ),
            RunOption(
                dest="snip_enabled",
                flags=("--snip",),
                negative_flags=("--no-snip",),
                kind=Kind.BOOL_PAIR,
                default=True,
                config_key="snip_enabled",
                help="enable the SNIP peak mask (default)",
                negative_help="disable the SNIP peak mask",
            ),
            RunOption(
                dest="snip_threshold_sigma",
                flags=("--snip-threshold",),
                kind=Kind.FLOAT,
                default=DEFAULT_SNIP_THRESHOLD_SIGMA,
                metavar="SIGMA",
                config_key="snip_threshold",
                help="peak significance threshold in sigma (default 5)",
            ),
            RunOption(
                dest="snip_protect_bins",
                flags=("--snip-protect-bins",),
                kind=Kind.INT,
                default=DEFAULT_SNIP_PROTECT_BINS,
                metavar="BINS",
                config_key="snip_protect_bins",
                help="protective half-width around a peak candidate in primary bins (default 3)",
            ),
            RunOption(
                dest="snip_floor",
                flags=("--snip-floor",),
                kind=Kind.FLOAT,
                default=DEFAULT_SNIP_FLOOR,
                metavar="W",
                config_key="snip_floor",
                help="penalty weight floor on protected peak bins (default 0.1)",
            ),
            RunOption(
                dest="snip_iterations",
                flags=("--snip-iterations",),
                kind=Kind.INT,
                metavar="M",
                config_key="snip_iterations",
                help="SNIP iteration count override (default: resolution-derived)",
            ),
            RunOption(
                dest="snip_max_iterations",
                flags=("--snip-max-iterations",),
                kind=Kind.INT,
                default=DEFAULT_SNIP_MAX_ITERATIONS,
                metavar="M",
                config_key="snip_max_iterations",
                help="cap for the resolution-derived SNIP iteration count (default 8)",
            ),
            # The retired D-156 spelling would otherwise be accepted by argparse
            # prefix matching and silently read as --snip-protect-bins with a
            # different unit (resolution sigmas -> primary bins); D-188/D-190
            # reject it with a pointer instead.
            RunOption(
                dest="snip_protect_retired",
                flags=("--snip-protect",),
                kind=Kind.FLOAT,
                metavar="SIGMA",
                retired=(
                    "(resolution sigmas) was replaced by --snip-protect-bins "
                    "(primary bins, default 3) in D-188; the half-width unit changed"
                ),
            ),
        ],
        output_options(with_plot=True),
        config_options(),
        runtime_options(),
    ),
)


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
    add_run_options(parser, UNFOLD_POLICY)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("unfold", args.log_level)
    config = load_unfold_config(args.config) if args.config is not None else None
    values = resolve_run_options(
        UNFOLD_POLICY, args, config_values=config.values() if config is not None else None
    )
    data = Path(str(values["data"])).expanduser()
    sim = None if values["sim"] is None else Path(str(values["sim"])).expanduser()
    energy_low, energy_high = _bounds(values)
    output = _output(values["output"], data, sim, values["alpha"], bool(values["calib_only"]))
    if values["dry_run"]:
        _print_dry_run(values, data, sim, output)
        return 0
    force = bool(values["force"])
    _validate_outputs(output, force, plot=not values["no_plot"])
    return _execute(
        values,
        data=data,
        sim=sim,
        energy_low=energy_low,
        energy_high=energy_high,
        output=output,
        force=force,
        strict=strict,
        logger=logger,
        arguments=argv_arguments(args.argv),
        config_inputs=(config.config_path,) if config is not None else (),
    )


def _bounds(values: Mapping[str, object]) -> tuple[float, float]:
    if values["calib_only"]:
        return _CALIB_ONLY_BOUNDS
    return float(values["energy_low"]), float(values["energy_high"])


def _output(
    explicit: object,
    data: Path,
    sim: Path | None,
    alpha: float | None,
    calib_only: bool,
) -> Path:
    if explicit is not None:
        return Path(str(explicit)).expanduser()
    if calib_only:
        return default_output("unfold", f"unfold-{data.stem}-calibonly.root")
    assert sim is not None and alpha is not None
    return default_output("unfold", f"unfold-{data.stem}-{sim.stem}-a{alpha:g}.root")


def _print_dry_run(
    values: Mapping[str, object], data: Path, sim: Path | None, output: Path
) -> None:
    print("kc761tool unfold (dry-run):")
    print(f"  mode={'calib_only' if values['calib_only'] else 'unfold'}")
    print(f"  data={data}")
    print(f"  calib={values['calib']}")
    print(f"  sim={sim}")
    print(
        f"  alpha={values['alpha']} difference_order={values['difference_order']} "
        f"syst_frac={values['syst_frac']}"
    )
    print(f"  output={output}")


def _validate_outputs(output: Path, force: bool, *, plot: bool) -> None:
    """Fail fast on an unusable product or figure target (D-171)."""
    validate_output_path(output, force=force)
    if plot:
        validate_output_path(Path(output).with_suffix(".pdf"), force=force)


def _execute(
    values: Mapping[str, object],
    *,
    data: Path,
    sim: Path | None,
    energy_low: float,
    energy_high: float,
    output: Path,
    force: bool,
    strict: bool,
    logger,
    arguments,
    config_inputs,
) -> int:
    from kc761tool.unfold import run_unfold

    result = run_unfold(
        data,
        values["calib"],
        sim,
        energy_low_kev=energy_low,
        energy_high_kev=energy_high,
        alpha=values["alpha"],
        difference_order=int(values["difference_order"]),
        syst_frac=float(values["syst_frac"]),
        snip_enabled=bool(values["snip_enabled"]),
        snip_threshold_sigma=float(values["snip_threshold_sigma"]),
        snip_protect_bins=int(values["snip_protect_bins"]),
        snip_floor=float(values["snip_floor"]),
        snip_iterations=values["snip_iterations"],
        snip_max_iterations=int(values["snip_max_iterations"]),
        calib_only=bool(values["calib_only"]),
        output=output,
        force=force,
        strict=strict,
        command="kc761tool unfold",
        arguments=arguments,
        plot=not values["no_plot"],
        plot_force=force,
        extra_inputs=config_inputs,
    )
    if result.report:
        print(result.report)
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["UNFOLD_POLICY", "add_parser"]
