"""Strict TOML configuration parsing for the four config-capable subcommands.

Decision D-129..D-142. One file may hold the top-level tables ``[sim]``,
``[calib]``, ``[compose]`` and ``[unfold]``; each subcommand reads only its own
table. Every file declares ``config_version = 1``, unknown keys fail loud, and
relative paths resolve against the current working directory (D-164).

This module is deliberately dependency-light: standard library only, no
Geant4 and no numerics. The valid source-key set and default numeric constants
are injected by the caller (:mod:`kc761tool.cli.sim` and the library it wires) so
this module never imports ``kc761tool.sim`` or ``kc761tool.calib``.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kc761tool.core.solver import (
    DEFAULT_SNIP_FLOOR,
    DEFAULT_SNIP_MAX_ITERATIONS,
    DEFAULT_SNIP_PROTECT_SIGMA,
    DEFAULT_SNIP_THRESHOLD_SIGMA,
)
from kc761tool.errors import UsageError

CONFIG_VERSION = 1
"""The only accepted ``config_version`` value (D-131)."""

CONFIG_TABLES = ("sim", "calib", "compose", "unfold")
"""Top-level tables a configuration file may declare (D-129)."""

# The accepted matrix-mode tokens are injected by the caller (D-143/D-2) so this
# module stays free of any ``kc761tool.sim`` import.


@dataclass(frozen=True)
class SimRunSpec:
    """One expanded single-run simulation ready for argv expansion (D-134)."""

    source_key: str | None
    matrix_mode: str | None
    calib: Path | None
    events: int
    threads: int | None
    seed: int
    verbose: int
    output: Path | None


@dataclass(frozen=True)
class SimConfig:
    """Validated ``[sim]`` table: batch options plus serial runs (D-134)."""

    config_path: Path
    resume: bool
    force: bool
    dry_run: bool
    runs: tuple[SimRunSpec, ...]


@dataclass(frozen=True)
class CalibDatasetConfig:
    """One ``[[calib.datasets]]`` entry (D-139/D-144)."""

    data: Path
    mc: Path
    label: str
    channel_low: int | None
    channel_high: int | None
    syst_frac: float


@dataclass(frozen=True)
class CalibConfig:
    """Validated ``[calib]`` table (D-139)."""

    config_path: Path
    datasets: tuple[CalibDatasetConfig, ...]
    output: Path | None
    no_plot: bool
    force: bool
    dry_run: bool


@dataclass(frozen=True)
class ComposeConfig:
    """Validated ``[compose]`` table (D-139)."""

    config_path: Path
    calib: Path
    sim: Path
    output: Path | None
    force: bool
    dry_run: bool


@dataclass(frozen=True)
class UnfoldConfig:
    """Validated ``[unfold]`` table (D-139)."""

    config_path: Path
    data: Path
    calib: Path
    sim: Path | None
    calib_only: bool
    energy_low_kev: float | None
    energy_high_kev: float | None
    alpha: float | None
    difference_order: int
    pad_nsigma: float
    syst_frac: float
    snip_enabled: bool
    snip_threshold_sigma: float
    snip_protect_sigma: float
    snip_floor: float
    snip_iterations: int | None
    snip_max_iterations: int
    output: Path | None
    no_plot: bool
    force: bool
    dry_run: bool


# --------------------------------------------------------------------------
# Loading and key/type validation
# --------------------------------------------------------------------------
def _load(path: str | Path) -> tuple[dict[str, object], Path]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise UsageError(f"config file not found: {config_path}")
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise UsageError(f"{config_path}: invalid TOML: {exc}") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise UsageError(f"{config_path}: invalid TOML text encoding: {exc}") from exc
    except OSError as exc:
        raise UsageError(f"{config_path}: cannot read config file: {exc}") from exc
    return data, config_path.resolve()


def _config_table(
    path: str | Path, name: str
) -> tuple[dict[str, object], Path, Path]:
    """Return ``(table, base_dir, resolved_config_path)`` for one subcommand.

    ``base_dir`` is the current working directory (D-163): relative paths in a
    configuration file resolve against the process CWD, not the file location.
    """
    data, config_path = _load(path)
    unknown = sorted(set(data) - {"config_version", *CONFIG_TABLES})
    if unknown:
        raise UsageError(
            f"{config_path}: unknown top-level key(s): {', '.join(unknown)}; "
            f"expected config_version and any of {', '.join(CONFIG_TABLES)}"
        )
    if "config_version" not in data:
        raise UsageError(f"{config_path}: config_version = {CONFIG_VERSION} is required")
    version = data["config_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise UsageError(
            f"{config_path}: config_version must be the integer {CONFIG_VERSION}, "
            f"got {version!r}"
        )
    if version != CONFIG_VERSION:
        raise UsageError(
            f"{config_path}: config_version must be {CONFIG_VERSION}, got {version!r}"
        )
    if name not in data:
        raise UsageError(f"{config_path}: missing [{name}] section")
    table = data[name]
    if not isinstance(table, dict):
        raise UsageError(f"{config_path}: [{name}] must be a table")
    return table, Path.cwd(), config_path


def _check_keys(
    table: dict[str, object],
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
    context: str,
) -> None:
    allowed = set(required) | set(optional)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise UsageError(
            f"{context}: unknown key(s): {', '.join(unknown)}; "
            f"expected {', '.join(sorted(allowed))}"
        )
    missing = [key for key in required if key not in table]
    if missing:
        raise UsageError(f"{context}: missing required key(s): {', '.join(missing)}")


def _as_str(table: dict[str, object], key: str, context: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value:
        raise UsageError(f"{context}: {key!r} must be a non-empty string")
    return value


def _as_int(table: dict[str, object], key: str, context: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsageError(f"{context}: {key!r} must be an integer")
    return value


def _as_float(table: dict[str, object], key: str, context: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError(f"{context}: {key!r} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise UsageError(f"{context}: {key!r} must be finite")
    return result


def _as_bool(table: dict[str, object], key: str, context: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise UsageError(f"{context}: {key!r} must be a boolean")
    return value


def _optional_bool(
    table: dict[str, object], key: str, default: bool, context: str
) -> bool:
    return _as_bool(table, key, context) if key in table else default


def _optional_int(table: dict[str, object], key: str, context: str) -> int | None:
    return _as_int(table, key, context) if key in table else None


def _optional_float(
    table: dict[str, object], key: str, default: float, context: str
) -> float:
    return _as_float(table, key, context) if key in table else default


def _resolve(base: Path, value: str, context: str, key: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def _positive(value: float, context: str, key: str) -> float:
    if value <= 0.0:
        raise UsageError(f"{context}: {key!r} must be positive, got {value!r}")
    return value


def _non_negative(value: float, context: str, key: str) -> float:
    if value < 0.0:
        raise UsageError(f"{context}: {key!r} must be non-negative, got {value!r}")
    return value


def _require_table(
    raw: object, context: str, *, required: Sequence[str], optional: Sequence[str]
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise UsageError(f"{context}: must be a table")
    _check_keys(raw, required=required, optional=optional, context=context)
    return raw


# --------------------------------------------------------------------------
# Per-command parsers
# --------------------------------------------------------------------------
def load_sim_config(
    path: str | Path,
    *,
    source_keys: Sequence[str],
    default_seed: int,
    matrix_modes: Sequence[str],
) -> SimConfig:
    """Parse and validate the ``[sim]`` batch table (D-134..D-137, D-143)."""
    table, base, config_path = _config_table(path, "sim")
    _check_keys(
        table,
        optional=("resume", "force", "dry_run", "runs"),
        context="[sim]",
    )
    raw_runs = table.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise UsageError("[sim]: runs must be a non-empty array of tables ([[sim.runs]])")
    runs: list[SimRunSpec] = []
    for index, raw in enumerate(raw_runs):
        context = f"[sim.runs][{index}]"
        run = _require_table(
            raw,
            context,
            required=("events",),
            optional=("source", "mode", "calib", "threads", "seed", "verbose", "output"),
        )
        has_source = "source" in run
        has_mode = "mode" in run
        if has_source == has_mode:
            raise UsageError(f"{context}: specify exactly one of 'source' or 'mode'")
        events = _as_int(run, "events", context)
        if events < 1:
            raise UsageError(f"{context}: 'events' must be >= 1, got {events!r}")
        threads = _optional_int(run, "threads", context)
        if threads is not None and threads < 1:
            raise UsageError(f"{context}: 'threads' must be >= 1, got {threads!r}")
        seed = _as_int(run, "seed", context) if "seed" in run else default_seed
        verbose = _optional_int(run, "verbose", context) or 0
        if verbose < 0:
            raise UsageError(f"{context}: 'verbose' must be >= 0, got {verbose!r}")
        output = _resolve(base, _as_str(run, "output", context), context, "output") \
            if "output" in run else None
        if has_source:
            source_key = _as_str(run, "source", context)
            if source_key not in source_keys:
                raise UsageError(
                    f"{context}: unknown source key {source_key!r}; "
                    f"expected one of {tuple(source_keys)}"
                )
            if "calib" in run:
                raise UsageError(f"{context}: a source run must not set 'calib'")
            runs.append(
                SimRunSpec(
                    source_key=source_key,
                    matrix_mode=None,
                    calib=None,
                    events=events,
                    threads=threads,
                    seed=seed,
                    verbose=verbose,
                    output=output,
                )
            )
            continue
        matrix_mode = _as_str(run, "mode", context)
        if matrix_mode not in matrix_modes:
            raise UsageError(
                f"{context}: 'mode' must be one of {tuple(matrix_modes)}, got {matrix_mode!r}"
            )
        if "calib" not in run:
            raise UsageError(f"{context}: a matrix run requires 'calib'")
        runs.append(
            SimRunSpec(
                source_key=None,
                matrix_mode=matrix_mode,
                calib=_resolve(base, _as_str(run, "calib", context), context, "calib"),
                events=events,
                threads=threads,
                seed=seed,
                verbose=verbose,
                output=output,
            )
        )
    return SimConfig(
        config_path=config_path,
        resume=_optional_bool(table, "resume", True, "[sim]"),
        force=_optional_bool(table, "force", False, "[sim]"),
        dry_run=_optional_bool(table, "dry_run", False, "[sim]"),
        runs=tuple(runs),
    )


def load_calib_config(path: str | Path, *, default_syst_frac: float) -> CalibConfig:
    """Parse and validate the ``[calib]`` single-fit table (D-139)."""
    table, base, config_path = _config_table(path, "calib")
    _check_keys(
        table,
        required=("datasets",),
        optional=("output", "no_plot", "force", "dry_run"),
        context="[calib]",
    )
    raw_datasets = table["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise UsageError(
            "[calib]: datasets must be a non-empty array of tables ([[calib.datasets]])"
        )
    datasets: list[CalibDatasetConfig] = []
    for index, raw in enumerate(raw_datasets):
        context = f"[calib.datasets][{index}]"
        entry = _require_table(
            raw,
            context,
            required=("data", "mc", "label"),
            optional=("channel_low", "channel_high", "syst_frac"),
        )
        label = _as_str(entry, "label", context)
        if ("channel_low" in entry) != ("channel_high" in entry):
            raise UsageError(
                f"{context}: channel_low and channel_high must be given together"
            )
        channel_low: int | None = None
        channel_high: int | None = None
        if "channel_low" in entry:
            channel_low = _as_int(entry, "channel_low", context)
            channel_high = _as_int(entry, "channel_high", context)
            if channel_low < 0:
                raise UsageError(f"{context}: 'channel_low' must be >= 0")
            if channel_high < channel_low:
                raise UsageError(
                    f"{context}: channel_high ({channel_high}) must be >= "
                    f"channel_low ({channel_low})"
                )
        syst_frac = _non_negative(
            _optional_float(entry, "syst_frac", default_syst_frac, context),
            context,
            "syst_frac",
        )
        datasets.append(
            CalibDatasetConfig(
                data=_resolve(base, _as_str(entry, "data", context), context, "data"),
                mc=_resolve(base, _as_str(entry, "mc", context), context, "mc"),
                label=label,
                channel_low=channel_low,
                channel_high=channel_high,
                syst_frac=syst_frac,
            )
        )
    output = _resolve(base, _as_str(table, "output", "[calib]"), "[calib]", "output") \
        if "output" in table else None
    return CalibConfig(
        config_path=config_path,
        datasets=tuple(datasets),
        output=output,
        no_plot=_optional_bool(table, "no_plot", False, "[calib]"),
        force=_optional_bool(table, "force", False, "[calib]"),
        dry_run=_optional_bool(table, "dry_run", False, "[calib]"),
    )


def load_compose_config(path: str | Path) -> ComposeConfig:
    """Parse and validate the ``[compose]`` table (D-139)."""
    table, base, config_path = _config_table(path, "compose")
    _check_keys(
        table,
        required=("calib", "sim"),
        optional=("output", "force", "dry_run"),
        context="[compose]",
    )
    output = _resolve(base, _as_str(table, "output", "[compose]"), "[compose]", "output") \
        if "output" in table else None
    return ComposeConfig(
        config_path=config_path,
        calib=_resolve(base, _as_str(table, "calib", "[compose]"), "[compose]", "calib"),
        sim=_resolve(base, _as_str(table, "sim", "[compose]"), "[compose]", "sim"),
        output=output,
        force=_optional_bool(table, "force", False, "[compose]"),
        dry_run=_optional_bool(table, "dry_run", False, "[compose]"),
    )


def load_unfold_config(path: str | Path, *, default_syst_frac: float) -> UnfoldConfig:
    """Parse and validate the ``[unfold]`` table (D-139)."""
    table, base, config_path = _config_table(path, "unfold")
    _check_keys(
        table,
        required=("data", "calib"),
        optional=(
            "sim",
            "calib_only",
            "energy_low",
            "energy_high",
            "alpha",
            "difference_order",
            "pad_nsigma",
            "syst_frac",
            "snip_enabled",
            "snip_threshold",
            "snip_protect",
            "snip_floor",
            "snip_iterations",
            "snip_max_iterations",
            "output",
            "no_plot",
            "force",
            "dry_run",
        ),
        context="[unfold]",
    )
    calib_only = _optional_bool(table, "calib_only", False, "[unfold]")
    sim: Path | None = None
    energy_low: float | None = None
    energy_high: float | None = None
    alpha: float | None = None
    if calib_only:
        banned = [key for key in ("sim", "energy_low", "energy_high", "alpha") if key in table]
        if banned:
            raise UsageError(
                "[unfold]: calib_only does not use "
                f"{', '.join(banned)}; remove it/them"
            )
    else:
        if "sim" not in table:
            raise UsageError("[unfold]: 'sim' is required unless calib_only = true")
        if "energy_low" not in table or "energy_high" not in table:
            raise UsageError(
                "[unfold]: energy_low and energy_high are required unless calib_only = true"
            )
        if "alpha" not in table:
            raise UsageError("[unfold]: 'alpha' is required unless calib_only = true")
        sim = _resolve(base, _as_str(table, "sim", "[unfold]"), "[unfold]", "sim")
        energy_low = _as_float(table, "energy_low", "[unfold]")
        energy_high = _as_float(table, "energy_high", "[unfold]")
        if not energy_low < energy_high:
            raise UsageError(
                f"[unfold]: energy_low ({energy_low}) must be < energy_high ({energy_high})"
            )
        alpha = _positive(_as_float(table, "alpha", "[unfold]"), "[unfold]", "alpha")
    parsed_order = _optional_int(table, "difference_order", "[unfold]")
    difference_order = 2 if parsed_order is None else parsed_order
    if difference_order not in (1, 2):
        raise UsageError(
            f"[unfold]: 'difference_order' must be 1 or 2, got {difference_order!r}"
        )
    pad_nsigma = _non_negative(
        _optional_float(table, "pad_nsigma", 5.0, "[unfold]"), "[unfold]", "pad_nsigma"
    )
    syst_frac = _non_negative(
        _optional_float(table, "syst_frac", default_syst_frac, "[unfold]"),
        "[unfold]",
        "syst_frac",
    )
    output = _resolve(base, _as_str(table, "output", "[unfold]"), "[unfold]", "output") \
        if "output" in table else None
    snip_threshold = _positive(
        _optional_float(table, "snip_threshold", DEFAULT_SNIP_THRESHOLD_SIGMA, "[unfold]"),
        "[unfold]",
        "snip_threshold",
    )
    snip_protect = _non_negative(
        _optional_float(table, "snip_protect", DEFAULT_SNIP_PROTECT_SIGMA, "[unfold]"),
        "[unfold]",
        "snip_protect",
    )
    snip_floor = _optional_float(table, "snip_floor", DEFAULT_SNIP_FLOOR, "[unfold]")
    if not 0.0 <= snip_floor <= 1.0:
        raise UsageError(f"[unfold]: 'snip_floor' must lie in [0, 1], got {snip_floor!r}")
    snip_iterations = _optional_int(table, "snip_iterations", "[unfold]")
    if snip_iterations is not None and snip_iterations < 1:
        raise UsageError(
            f"[unfold]: 'snip_iterations' must be >= 1, got {snip_iterations!r}"
        )
    snip_max_iterations = _optional_int(table, "snip_max_iterations", "[unfold]")
    if snip_max_iterations is None:
        snip_max_iterations = DEFAULT_SNIP_MAX_ITERATIONS
    if snip_max_iterations < 1:
        raise UsageError(
            f"[unfold]: 'snip_max_iterations' must be >= 1, got {snip_max_iterations!r}"
        )
    return UnfoldConfig(
        config_path=config_path,
        data=_resolve(base, _as_str(table, "data", "[unfold]"), "[unfold]", "data"),
        calib=_resolve(base, _as_str(table, "calib", "[unfold]"), "[unfold]", "calib"),
        sim=sim,
        calib_only=calib_only,
        energy_low_kev=energy_low,
        energy_high_kev=energy_high,
        alpha=alpha,
        difference_order=difference_order,
        pad_nsigma=pad_nsigma,
        syst_frac=syst_frac,
        snip_enabled=_optional_bool(table, "snip_enabled", True, "[unfold]"),
        snip_threshold_sigma=snip_threshold,
        snip_protect_sigma=snip_protect,
        snip_floor=snip_floor,
        snip_iterations=snip_iterations,
        snip_max_iterations=snip_max_iterations,
        output=output,
        no_plot=_optional_bool(table, "no_plot", False, "[unfold]"),
        force=_optional_bool(table, "force", False, "[unfold]"),
        dry_run=_optional_bool(table, "dry_run", False, "[unfold]"),
    )


__all__ = [
    "CONFIG_TABLES",
    "CONFIG_VERSION",
    "CalibConfig",
    "CalibDatasetConfig",
    "ComposeConfig",
    "SimConfig",
    "SimRunSpec",
    "UnfoldConfig",
    "load_calib_config",
    "load_compose_config",
    "load_sim_config",
    "load_unfold_config",
]
