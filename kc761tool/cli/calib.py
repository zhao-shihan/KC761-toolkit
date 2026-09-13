"""``kc761tool calib``: energy/resolution calibration.

One or more data/MC dataset pairs are fitted with a shared calibration. The
measured spectrum is a ``spectrum`` product and ``--mc`` reads an
``mc_spectrum`` product (D-120/D-144); the library consumes only the histogram,
so the CLI owns product loading and axis checks. Config mode runs a single fit
from ``[[calib.datasets]]`` (D-139).

The option surface is declared once in :data:`CALIB_POLICY` (D-190). The
repeatable command-line options (one value per dataset) and the scalar
per-dataset TOML keys are separate declarations of the same concept, because
the two surfaces genuinely differ in shape; the batch twins use the ``dataset_``
dest prefix.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kc761tool.cli._common import argv_arguments, default_output
from kc761tool.cli._registry import (
    Kind,
    Requirement,
    RunOption,
    RunPolicy,
    Scope,
    add_run_options,
    channel_window_options,
    config_options,
    merge_options,
    output_options,
    resolve_run_options,
    runtime_options,
)
from kc761tool.cli.config import CalibConfig, load_calib_config
from kc761tool.core.uncertainty import DEFAULT_SYST_FRAC
from kc761tool.errors import SchemaError, UsageError
from kc761tool.runtime import configure_logging
from kc761tool.schema.io import validate_output_path


def _require_int_pair(
    values: Mapping[str, object],
    *,
    low_key: str,
    high_key: str,
    label: str,
) -> None:
    """Bounds for a channel window pair, shared by the CLI and the batch table."""
    low, high = values.get(low_key), values.get(high_key)
    if low is not None and low < 0:
        raise UsageError(f"{label}_low must be >= 0, got {low!r}")
    if low is not None and high is not None and high < low:
        raise UsageError(f"{label}_high ({high}) must be >= {label}_low ({low})")


def _validate_calib(
    values: Mapping[str, object], present: frozenset[str], config_mode: bool
) -> None:
    """Calibration cross-field rules, shared by the CLI and the batch table."""
    del present  # calib has no per-source rule beyond the requirements
    _require_int_pair(values, low_key="channel_low", high_key="channel_high", label="channel")
    iterations = values.get("max_iter")
    if iterations is not None and iterations < 1:
        raise UsageError(f"--max-iter must be >= 1, got {iterations!r}")
    tolerance = values.get("tolerance")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0.0):
        raise UsageError(f"--tolerance must be positive and finite, got {tolerance!r}")
    progress = values.get("progress_every")
    if progress is not None and (not math.isfinite(progress) or progress < 0.0):
        raise UsageError(f"--progress-every must be finite and >= 0, got {progress!r}")
    data, mc, label = values.get("data"), values.get("mc"), values.get("label")
    if data is None or mc is None or label is None:
        return
    count = len(data)
    if len(mc) != count or len(label) != count:
        raise UsageError(
            "--data, --mc and --label must be repeated the same number of times; got "
            f"{count} data, {len(mc)} mc, {len(label)} label"
        )


def _validate_dataset(
    values: Mapping[str, object], present: frozenset[str], config_mode: bool
) -> None:
    """The same window rules inside one ``[[calib.datasets]]`` entry."""
    del config_mode
    if ("channel_low" in present) != ("channel_high" in present):
        raise UsageError("channel_low and channel_high must be given together")
    _require_int_pair(values, low_key="channel_low", high_key="channel_high", label="channel")


def _non_negative_values(values: Mapping[str, object], key: str) -> str | None:
    """Finite non-negative check for one value or a repeatable list of them.

    The command line carries one value per dataset (``--syst-frac`` repeated) and
    a ``[[calib.datasets]]`` entry carries a scalar, so both spellings share this
    bound (D-190); a bound that only guards one surface is exactly the drift the
    declaration is meant to remove.
    """
    value = values.get(key)
    if value is None:
        return None
    seen = value if isinstance(value, (list, tuple)) else [value]
    for item in seen:
        if not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0.0:
            return f"must be finite and >= 0, got {item!r}"
    return None


def _syst_frac_check(values: Mapping[str, object]) -> str | None:
    return _non_negative_values(values, "syst_frac")


def _dataset_syst_frac_check(values: Mapping[str, object]) -> str | None:
    return _non_negative_values(values, "dataset_syst_frac")


#: Declared surface (D-190): flags, defaults, TOML keys and the shared rules.
CALIB_POLICY = RunPolicy(
    command="calib",
    config=True,
    validate=_validate_calib,
    validate_nested=_validate_dataset,
    spec=merge_options(
        [
            RunOption(
                dest="data",
                flags=("--data",),
                kind=Kind.PATH_LIST,
                metavar="FILE",
                requirement=Requirement.ARGS,
                help="data spectrum product (kc761_spectrum); repeat once per dataset",
            ),
            RunOption(
                dest="mc",
                flags=("--mc",),
                kind=Kind.PATH_LIST,
                metavar="FILE",
                requirement=Requirement.ARGS,
                help="source-mode simulation product (kc761_mc_spectrum); repeat per dataset",
            ),
            RunOption(
                dest="label",
                flags=("--label",),
                kind=Kind.STRING_LIST,
                metavar="NAME",
                requirement=Requirement.ARGS,
                help="dataset label (plot titles, scale parameter names); repeat per dataset",
            ),
            *channel_window_options(requirement=Requirement.ARGS),
            RunOption(
                dest="syst_frac",
                flags=("--syst-frac", "--syst"),
                kind=Kind.FLOAT_LIST,
                metavar="FRAC",
                check=_syst_frac_check,
                help=(
                    "per-bin fractional systematic uncertainty (0.05 = 5%%); single "
                    "value or one per dataset (default 0.05 = 5%%, formula F-CAL-1)"
                ),
            ),
            RunOption(
                dest="max_iter",
                flags=("--max-iter",),
                kind=Kind.INT,
                metavar="N",
                help="maximum optimizer function evaluations (default: FitSettings, D-107)",
            ),
            RunOption(
                dest="tolerance",
                flags=("--tolerance",),
                kind=Kind.FLOAT,
                metavar="TOL",
                help="positive convergence tolerance for ftol/xtol/gtol (default: FitSettings)",
            ),
            RunOption(
                dest="progress_every",
                flags=("--progress-every",),
                kind=Kind.FLOAT,
                default=1.0,
                scope=Scope.GLOBAL,
                metavar="SECONDS",
                help="fit progress line interval in seconds (default 1; 0 = every evaluation)",
            ),
            RunOption(
                dest="progress_enabled",
                flags=("--no-progress",),
                kind=Kind.BOOL,
                default=True,
                scope=Scope.GLOBAL,
                help="disable the fit summary and progress lines",
            ),
            # Batch twins: one scalar per [[calib.datasets]] entry (D-139/D-190).
            RunOption(
                dest="dataset_data",
                kind=Kind.PATH,
                nested_key="data",
                nested_requirement=Requirement.ALWAYS,
            ),
            RunOption(
                dest="dataset_mc",
                kind=Kind.PATH,
                nested_key="mc",
                nested_requirement=Requirement.ALWAYS,
            ),
            RunOption(
                dest="dataset_label",
                kind=Kind.STRING,
                nested_key="label",
                nested_requirement=Requirement.ALWAYS,
            ),
            RunOption(
                dest="dataset_syst_frac",
                kind=Kind.FLOAT,
                default=DEFAULT_SYST_FRAC,
                nested_key="syst_frac",
                check=_dataset_syst_frac_check,
            ),
        ],
        output_options(with_plot=True),
        config_options(),
        runtime_options(),
    ),
)


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
    add_run_options(parser, CALIB_POLICY)
    parser.set_defaults(handler=_run)


@dataclass(frozen=True)
class _Dataset:
    """One resolved dataset, from the command line or a ``[[calib.datasets]]``."""

    data: str | Path
    mc: str | Path
    label: str
    channel_low: int | None
    channel_high: int | None
    syst_frac: float


def _datasets(values: Mapping[str, object], config: CalibConfig | None) -> tuple[_Dataset, ...]:
    """Assemble the dataset list of either surface into one shape (D-190)."""
    if config is not None:
        return tuple(
            _Dataset(
                data=entry.data,
                mc=entry.mc,
                label=entry.label,
                channel_low=entry.channel_low,
                channel_high=entry.channel_high,
                syst_frac=entry.syst_frac,
            )
            for entry in config.datasets
        )
    syst = _resolve_syst(values["syst_frac"], len(values["data"]))
    return tuple(
        _Dataset(
            data=data,
            mc=mc,
            label=label,
            channel_low=values["channel_low"],
            channel_high=values["channel_high"],
            syst_frac=syst[index],
        )
        for index, (data, mc, label) in enumerate(
            zip(values["data"], values["mc"], values["label"], strict=True)
        )
    )


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("calib", args.log_level)
    config = load_calib_config(args.config) if args.config is not None else None
    values = resolve_run_options(
        CALIB_POLICY, args, config_values=config.values() if config is not None else None
    )
    datasets = _datasets(values, config)
    hints = [(entry.label, entry.channel_low, entry.channel_high) for entry in datasets]
    output = _output(values["output"], [entry.label for entry in datasets])
    if values["dry_run"]:
        _print_dry_run(
            [str(entry.data) for entry in datasets],
            [str(entry.mc) for entry in datasets],
            [entry.label for entry in datasets],
            output,
        )
        return 0
    force = bool(values["force"])
    validate_output_path(output, force=force)
    if not values["no_plot"]:
        validate_output_path(Path(output).with_suffix(".pdf"), force=force)
    # The optimizer overrides are command-line-only (the batch table uses
    # FitSettings defaults); the progress controls are global (D-130).
    settings = None if config is not None else _settings(values["max_iter"], values["tolerance"])
    progress = _progress_printer(hints, settings) if values["progress_enabled"] else None
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
        for entry in datasets
    )
    return _execute(
        specs,
        output=output,
        force=force,
        no_plot=bool(values["no_plot"]),
        settings=settings,
        progress=progress,
        progress_every_s=float(values["progress_every"]),
        strict=strict,
        logger=logger,
        arguments=argv_arguments(args.argv),
        config_inputs=(config.config_path,) if config is not None else (),
    )


def _resolve_syst(values: list[float] | None, count: int) -> list[float]:
    from kc761tool.calib.types import DEFAULT_SYST_FRAC

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
    from kc761tool.calib.types import FitSettings

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
    from kc761tool.calib.types import DatasetSpec
    from kc761tool.schema.io import read_product
    from kc761tool.schema.products import McSpectrumProduct, SpectrumProduct

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
            f"dataset {label!r}: channel window [{low}, {high}] is outside [0, {n_bins - 1}]"
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


def _print_dry_run(data: list[str], mc: list[str], labels: list[str], output: Path) -> None:
    print("kc761tool calib (dry-run):")
    for label, data_path, mc_path in zip(labels, data, mc, strict=True):
        print(f"  dataset {label}: data={data_path} mc={mc_path}")
    print(f"  output={output}")


def _progress_printer(hints, settings):
    """Return a callback printing the pre-fit summary and periodic progress."""
    from kc761tool.calib.types import FitProgress, FitSettings

    resolved = settings if settings is not None else FitSettings()

    def printer(event: FitProgress) -> None:
        if event.nfev == 0:
            joined = ", ".join(
                f"{label} [{lo if lo is not None else 'full'}-{hi if hi is not None else 'full'}]"
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
    from kc761tool.calib import run_fit

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
        command="kc761tool calib",
        arguments=arguments,
        extra_inputs=config_inputs,
    )
    if result.report:
        print(result.report)
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["add_parser"]
