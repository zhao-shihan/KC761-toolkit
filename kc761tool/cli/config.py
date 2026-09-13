"""Strict TOML configuration parsing for the four config-capable subcommands.

Decision D-129..D-142. One file may hold the top-level tables ``[sim]``,
``[calib]``, ``[compose]`` and ``[unfold]``; each subcommand reads only its own
table. Every file declares ``config_version = 1``, unknown keys fail loud, and
relative paths resolve against the current working directory (D-164).

This module is deliberately dependency-light: standard library only, no
Geant4 and no numerics. The accepted keys, their types and their defaults come
from the owning command's :class:`kc761tool.cli._registry.RunPolicy` (D-190),
which each loader imports lazily, so this module never imports
``kc761tool.sim`` or ``kc761tool.calib`` at import time.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kc761tool.cli._registry import (
    Kind,
    RunOption,
    check_policy,
    config_keys,
    nested_config_keys,
    nested_policy,
    options_by_dest,
    read_config_value,
    retired_toml_keys,
)
from kc761tool.errors import UsageError

CONFIG_VERSION = 1
"""The only accepted ``config_version`` value (D-131)."""

CONFIG_TABLES = ("sim", "calib", "compose", "unfold")
"""Top-level tables a configuration file may declare (D-129)."""

# The source keys and matrix-mode tokens are declared by SIM_POLICY (D-190), so
# this module stays free of any ``kc761tool.sim`` import.


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

    def values(self) -> dict[str, object]:
        """Registry mapping: dest -> value for the batch-level options (D-190)."""
        return {"resume": self.resume, "force": self.force, "dry_run": self.dry_run}


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

    def values(self) -> dict[str, object]:
        """Registry mapping: dest -> value for the flat options (D-190)."""
        return {
            "output": self.output,
            "no_plot": self.no_plot,
            "force": self.force,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class ComposeConfig:
    """Validated ``[compose]`` table (D-139)."""

    config_path: Path
    calib: Path
    sim: Path
    output: Path | None
    force: bool
    dry_run: bool

    def values(self) -> dict[str, object]:
        """Registry mapping: dest -> value for every config-backed option (D-190)."""
        return {
            "calib": self.calib,
            "sim": self.sim,
            "output": self.output,
            "force": self.force,
            "dry_run": self.dry_run,
        }


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
    syst_frac: float
    snip_enabled: bool
    snip_threshold_sigma: float
    snip_protect_bins: int
    snip_floor: float
    snip_iterations: int | None
    snip_max_iterations: int
    output: Path | None
    no_plot: bool
    force: bool
    dry_run: bool

    def values(self) -> dict[str, object]:
        """Registry mapping: dest -> value for every flat option (D-190)."""
        return {
            "data": self.data,
            "calib": self.calib,
            "sim": self.sim,
            "calib_only": self.calib_only,
            "energy_low": self.energy_low_kev,
            "energy_high": self.energy_high_kev,
            "alpha": self.alpha,
            "difference_order": self.difference_order,
            "syst_frac": self.syst_frac,
            "snip_enabled": self.snip_enabled,
            "snip_threshold_sigma": self.snip_threshold_sigma,
            "snip_protect_bins": self.snip_protect_bins,
            "snip_floor": self.snip_floor,
            "snip_iterations": self.snip_iterations,
            "snip_max_iterations": self.snip_max_iterations,
            "output": self.output,
            "no_plot": self.no_plot,
            "force": self.force,
            "dry_run": self.dry_run,
        }


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


def _config_table(path: str | Path, name: str) -> tuple[dict[str, object], Path, Path]:
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
            f"{config_path}: config_version must be the integer {CONFIG_VERSION}, got {version!r}"
        )
    if version != CONFIG_VERSION:
        raise UsageError(f"{config_path}: config_version must be {CONFIG_VERSION}, got {version!r}")
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
    retired: Mapping[str, str] | None = None,
    context: str,
) -> None:
    allowed = set(required) | set(optional)
    hints = retired or {}
    unknown = sorted(set(table) - allowed)
    stale = [key for key in unknown if key in hints]
    if stale:
        raise UsageError(
            f"{context}: {', '.join(repr(key) for key in stale)} "
            + "; ".join(hints[key] for key in stale)
        )
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


def _optional_bool(table: dict[str, object], key: str, default: bool, context: str) -> bool:
    return _as_bool(table, key, context) if key in table else default


def _optional_int(table: dict[str, object], key: str, context: str) -> int | None:
    return _as_int(table, key, context) if key in table else None


def _optional_float(table: dict[str, object], key: str, default: float, context: str) -> float:
    return _as_float(table, key, context) if key in table else default


def read_toml_value(table: Mapping[str, object], key: str, tag: str, context: str) -> object:
    """Read ``table[key]`` as the registry's declared TOML type ``tag`` (D-190).

    The tags mirror :class:`kc761tool.cli._registry.Kind`, so an option's config
    type and its command-line type come from one declaration.
    """
    if tag == "str":
        return _as_str(table, key, context)
    if tag == "int":
        return _as_int(table, key, context)
    if tag == "float":
        return _as_float(table, key, context)
    if tag == "bool":
        return _as_bool(table, key, context)
    if tag in ("str_list", "float_list"):
        raw = table[key]
        if not isinstance(raw, list) or not raw:
            raise UsageError(f"{context}: {key!r} must be a non-empty array")
        reader = _as_float if tag == "float_list" else _as_str
        return [reader({key: item}, key, context) for item in raw]
    raise AssertionError(f"unknown TOML type tag {tag!r}")


def _table_values(
    table: Mapping[str, object],
    base: Path,
    options: Mapping[str, RunOption],
    *,
    context: str,
) -> dict[str, object]:
    """Read every declared key of ``table`` into a dest-keyed mapping (D-190).

    Path-valued options are resolved against the config file's directory; absent
    keys take the option's declared default. Required-ness and bounds are not
    checked here: :func:`check_policy` applies the same rules the command line
    uses.
    """
    values: dict[str, object] = {}
    for key, option in options.items():
        value = read_config_value(table, option, context=context)
        if option.kind is Kind.PATH and value is not None:
            value = _resolve(base, value, context, key)
        values[option.dest] = value
    return values


def _resolve(base: Path, value: str, context: str, key: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


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
def load_sim_config(path: str | Path) -> SimConfig:
    """Parse the ``[sim]`` batch table (D-134..D-137, D-143, D-190).

    Per-run keys, their types and their defaults come from ``SIM_POLICY`` (the
    declaration that also builds the parser); only the batch-level ``runs``
    array is local to this loader. Source keys and matrix modes are validated by
    that declaration, so nothing is listed twice.
    """
    from kc761tool.cli.sim import SIM_POLICY as policy

    table, base, config_path = _config_table(path, "sim")
    flat = config_keys(policy)
    _check_keys(table, required=(), optional=("runs", *flat), context="[sim]")
    values = _table_values(table, base, flat, context="[sim]")
    check_policy(policy, values, config_mode=True, context="[sim]")

    nested = nested_policy(policy)
    nested_keys = nested_config_keys(policy)
    raw_runs = table.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise UsageError("[sim]: runs must be a non-empty array of tables ([[sim.runs]])")
    runs: list[SimRunSpec] = []
    for index, raw in enumerate(raw_runs):
        context = f"[sim.runs][{index}]"
        entry = _require_table(raw, context, required=(), optional=tuple(nested_keys))
        entry_values = _table_values(entry, base, nested_keys, context=context)
        check_policy(
            nested,
            entry_values,
            config_mode=True,
            context=context,
            present=frozenset(
                option.dest
                for option in options_by_dest(nested).values()
                if option.toml_key() in entry
            ),
        )
        has_mode = entry_values["mode"] is not None
        if has_mode:
            runs.append(
                SimRunSpec(
                    source_key=None,
                    matrix_mode=str(entry_values["mode"]),
                    calib=entry_values["matrix_calib"],
                    events=int(entry_values["events"]),
                    threads=entry_values["threads"],
                    seed=int(entry_values["seed"]),
                    verbose=int(entry_values["verbose"]),
                    output=entry_values["output"],
                )
            )
        else:
            runs.append(
                SimRunSpec(
                    source_key=str(entry_values["source"]),
                    matrix_mode=None,
                    calib=None,
                    events=int(entry_values["events"]),
                    threads=entry_values["threads"],
                    seed=int(entry_values["seed"]),
                    verbose=int(entry_values["verbose"]),
                    output=entry_values["output"],
                )
            )
    return SimConfig(
        config_path=config_path,
        resume=bool(values["resume"]),
        force=bool(values["force"]),
        dry_run=bool(values["dry_run"]),
        runs=tuple(runs),
    )


def load_calib_config(path: str | Path) -> CalibConfig:
    """Parse the ``[calib]`` single-fit table (D-139/D-190) against its policy."""
    from kc761tool.cli.calib import CALIB_POLICY as policy

    table, base, config_path = _config_table(path, "calib")
    flat = config_keys(policy)
    _check_keys(table, required=("datasets",), optional=tuple(flat), context="[calib]")
    values = _table_values(table, base, flat, context="[calib]")
    check_policy(policy, values, config_mode=True, context="[calib]")

    nested = nested_policy(policy)
    nested_keys = nested_config_keys(policy)
    raw_datasets = table["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise UsageError(
            "[calib]: datasets must be a non-empty array of tables ([[calib.datasets]])"
        )
    datasets: list[CalibDatasetConfig] = []
    for index, raw in enumerate(raw_datasets):
        context = f"[calib.datasets][{index}]"
        entry = _require_table(raw, context, required=(), optional=tuple(nested_keys))
        entry_values = _table_values(entry, base, nested_keys, context=context)
        check_policy(
            nested,
            entry_values,
            config_mode=True,
            context=context,
            present=frozenset(
                option.dest
                for option in options_by_dest(nested).values()
                if option.toml_key() in entry
            ),
        )
        datasets.append(
            CalibDatasetConfig(
                data=entry_values["dataset_data"],
                mc=entry_values["dataset_mc"],
                label=str(entry_values["dataset_label"]),
                channel_low=entry_values["channel_low"],
                channel_high=entry_values["channel_high"],
                syst_frac=float(entry_values["dataset_syst_frac"]),
            )
        )
    return CalibConfig(
        config_path=config_path,
        datasets=tuple(datasets),
        output=values["output"],
        no_plot=bool(values["no_plot"]),
        force=bool(values["force"]),
        dry_run=bool(values["dry_run"]),
    )


def load_compose_config(path: str | Path) -> ComposeConfig:
    """Parse and validate the ``[compose]`` table (D-139/D-190).

    The key set, types and defaults come from ``COMPOSE_POLICY`` (the same
    declaration that builds the parser), and the assembled mapping goes through
    :func:`check_policy`, so the config path enforces exactly the rules the
    command line does.
    """
    from kc761tool.cli.compose import COMPOSE_POLICY as policy

    table, base, config_path = _config_table(path, "compose")
    keys = config_keys(policy)
    _check_keys(table, required=(), optional=tuple(keys), context="[compose]")
    values = _table_values(table, base, keys, context="[compose]")
    check_policy(policy, values, config_mode=True, context="[compose]")
    return ComposeConfig(
        config_path=config_path,
        calib=values["calib"],
        sim=values["sim"],
        output=values["output"],
        force=bool(values["force"]),
        dry_run=bool(values["dry_run"]),
    )


def load_unfold_config(path: str | Path) -> UnfoldConfig:
    """Parse the ``[unfold]`` table (D-139/D-190) against its policy."""
    from kc761tool.cli.unfold import UNFOLD_POLICY as policy

    table, base, config_path = _config_table(path, "unfold")
    keys = config_keys(policy)
    _check_keys(
        table,
        required=(),
        optional=tuple(keys),
        retired=retired_toml_keys(policy),
        context="[unfold]",
    )
    values = _table_values(table, base, keys, context="[unfold]")
    check_policy(
        policy,
        values,
        config_mode=True,
        context="[unfold]",
        present=frozenset(option.dest for option in keys.values() if option.toml_key() in table),
    )
    return UnfoldConfig(
        config_path=config_path,
        data=values["data"],
        calib=values["calib"],
        sim=values["sim"],
        calib_only=bool(values["calib_only"]),
        energy_low_kev=values["energy_low"],
        energy_high_kev=values["energy_high"],
        alpha=values["alpha"],
        difference_order=int(values["difference_order"]),
        syst_frac=float(values["syst_frac"]),
        snip_enabled=bool(values["snip_enabled"]),
        snip_threshold_sigma=float(values["snip_threshold_sigma"]),
        snip_protect_bins=int(values["snip_protect_bins"]),
        snip_floor=float(values["snip_floor"]),
        snip_iterations=values["snip_iterations"],
        snip_max_iterations=int(values["snip_max_iterations"]),
        output=values["output"],
        no_plot=bool(values["no_plot"]),
        force=bool(values["force"]),
        dry_run=bool(values["dry_run"]),
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
