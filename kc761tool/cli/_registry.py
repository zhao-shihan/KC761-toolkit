"""Declarative run-option registry: the single source of the CLI option surface.

Why this module exists (D-190). The config-exclusivity rule of D-130 used to be
enforced by comparing each parsed value with a mirror table of defaults
(``getattr(args, dest) != default``). argparse materializes defaults for options
the user never passed, so "explicitly passed the default value" and "not passed
at all" were indistinguishable, and the mirror table could drift from the parser
(it did: the whole SNIP family was missing, so every ``--snip-*`` value was
silently ignored in config mode). This registry removes the mirror table and the
value inference alike:

* every option is declared **once**, with its flag spellings, type, default,
  scope, TOML key and requirement;
* :func:`add_run_options` builds the parser from that declaration and gives
  run-scoped options ``default=argparse.SUPPRESS``, so "the user provided this"
  becomes a structural fact (attribute presence) instead of a value comparison;
* :func:`resolve_run_options` is the **only** place where defaults are applied,
  requirements are checked, retired spellings are rejected, the config conflict
  is detected and validation runs — so ``--help``, ``--dry-run``, a real run and
  a TOML config cannot disagree about what an option means or when it is valid.

Scopes (D-130, clarified by D-190):

``RUN``
    Run-selection option. Mutually exclusive with ``--config``; its value comes
    from the command line or from the config file, never from both.
``GLOBAL``
    May accompany ``--config`` (``--strict``, ``--log-level``, ``--dry-run``,
    ``--force`` and the calibration progress controls). These keep ordinary
    argparse defaults because the entry point reads them before the handler.
``INTERNAL``
    Hidden plumbing (``--provenance-input``): outside the run-option policy,
    never recorded in provenance, never rejected in config mode.

Requirements are evaluated on the *resolved* mapping (defaults plus whatever the
command line or the config file supplied), which is what lets one rule serve
both paths: ``NEVER`` (optional), ``ALWAYS`` (required from either source) and
``ARGS`` (required unless a config file is in play — used where the CLI demands
an explicit choice but the config format allows omitting it, e.g. the
calibration channel window, whose library default is the full range).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from kc761tool.errors import UsageError

__all__ = [
    "Kind",
    "Requirement",
    "RunOption",
    "RunPolicy",
    "Scope",
    "add_run_options",
    "channel_window_options",
    "check_policy",
    "config_keys",
    "config_options",
    "energy_window_options",
    "nested_config_keys",
    "options_by_dest",
    "output_options",
    "merge_options",
    "nested_policy",
    "provided_options",
    "read_config_value",
    "resolve_run_options",
    "retired_toml_keys",
    "runtime_options",
]


class Scope(StrEnum):
    """Where an option's value may come from (see the module docstring)."""

    RUN = "run"
    GLOBAL = "global"
    INTERNAL = "internal"


class Requirement(StrEnum):
    """When an option must be present in the resolved mapping."""

    NEVER = "never"
    ALWAYS = "always"
    ARGS = "args"


class Merge(StrEnum):
    """How a config value combines with the command line (D-130/D-190)."""

    CLI = "cli"
    """Config values never override this option (nothing to merge)."""
    CONFIG = "config"
    """A supplied config value wins; run options forbid the CLI spelling anyway."""
    OR = "or"
    """Boolean OR of the two sources; used by the global ``--force``/``--dry-run``."""


class Kind(StrEnum):
    """Value shape of an option; drives argparse and the TOML reader alike."""

    PATH = "path"
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    """``store_true`` when the default is False, ``store_false`` when it is True."""
    BOOL_PAIR = "bool_pair"
    """One boolean with a positive and a negative spelling (``--snip``/``--no-snip``)."""
    COUNT = "count"
    """``action="count"``; the value is the number of repetitions."""
    CONST = "const"
    """``action="store_const"``; alternative flags may share one dest."""
    PATH_LIST = "path_list"
    """``action="append"`` of paths."""
    FLOAT_LIST = "float_list"
    """``action="append"`` of floats."""
    STRING_LIST = "string_list"
    """``action="append"`` of strings."""


#: Kind -> argparse ``type`` (``None`` for the value-less actions).
_KIND_ARG_TYPES: dict[Kind, Callable[[str], Any] | None] = {
    Kind.PATH: str,
    Kind.STRING: str,
    Kind.INT: int,
    Kind.FLOAT: float,
    Kind.BOOL: None,
    Kind.BOOL_PAIR: None,
    Kind.COUNT: None,
    Kind.CONST: None,
    Kind.PATH_LIST: str,
    Kind.FLOAT_LIST: float,
    Kind.STRING_LIST: str,
}

#: Kind -> the TOML reader tag used by :func:`read_config_value`. Repeatable
#: kinds never reach it (a scalar TOML key cannot carry them; see
#: :func:`validate_policy`), so only the scalar tags appear here.
_KIND_TOML_TYPES: dict[Kind, str] = {
    Kind.PATH: "str",
    Kind.STRING: "str",
    Kind.INT: "int",
    Kind.FLOAT: "float",
    Kind.BOOL: "bool",
    Kind.BOOL_PAIR: "bool",
    Kind.COUNT: "int",
    Kind.CONST: "str",
    Kind.PATH_LIST: "str_list",
    Kind.FLOAT_LIST: "float_list",
    Kind.STRING_LIST: "str_list",
}


@dataclass(frozen=True)
class RunOption:
    """One CLI option (and, when declared, its TOML counterpart)."""

    dest: str
    flags: tuple[str, ...] = ()
    kind: Kind = Kind.STRING
    default: object = None
    scope: Scope = Scope.RUN
    help: str = ""
    metavar: str | None = None
    choices: tuple[str, ...] | None = None
    config_key: str | None = None
    nested_key: str | None = None
    requirement: Requirement = Requirement.NEVER
    nested_requirement: Requirement | None = None
    """Requirement inside a per-dataset/per-run table, when it differs."""
    required_if: Callable[[Mapping[str, object]], bool] | None = None
    merge: Merge | None = None
    """``None`` derives it from the scope: RUN -> CONFIG, otherwise CLI."""
    group: str | None = None
    const: object = None
    negative_flags: tuple[str, ...] = ()
    negative_help: str | None = None
    """Help of the negative spelling (``--no-x``); defaults to :attr:`help`."""
    retired: str | None = None
    check: Callable[[Mapping[str, object]], str | None] | None = None
    """Whole-mapping check; returns a message suffix or ``None``. Cross-field
    rules that belong to one option (``--energy-high`` vs ``--energy-low``) use
    this, so nothing has to be re-checked in a second place."""

    @property
    def merge_mode(self) -> Merge:
        if self.merge is not None:
            return self.merge
        return Merge.CONFIG if self.scope is Scope.RUN else Merge.CLI

    @property
    def configurable(self) -> bool:
        """True when a TOML key (top-level or nested) carries this value.

        This drives the loaders' key sets and the spelling used in messages; the
        config conflict itself is decided by :attr:`scope`, because a run
        option is exclusive with ``--config`` even when the batch table has no
        counterpart for it.
        """
        return self.config_key is not None or self.nested_key is not None

    @property
    def spelling(self) -> str:
        """Primary command-line spelling for messages."""
        for flag in self.flags:
            if flag.startswith("--"):
                return flag
        if self.flags:
            return self.flags[0]
        return f"'{self.toml_key()}'"

    def toml_key(self) -> str:
        """The TOML key that carries this option (top-level or nested)."""
        key = self.config_key or self.nested_key
        if key is None:
            raise AssertionError(f"{self.dest!r} has no TOML key")
        return key

    def spelling_for(self, config_mode: bool) -> str:
        """Message spelling: the TOML key inside a config file, else the flag."""
        if config_mode and self.configurable:
            return f"'{self.toml_key()}'"
        return self.spelling

    @property
    def is_hidden(self) -> bool:
        return self.scope is Scope.INTERNAL or self.retired is not None

    def toml_type(self) -> str:
        return _KIND_TOML_TYPES[self.kind]

    def argparse_kwargs(self) -> dict[str, Any]:
        """Translate the declaration into ``add_argument`` keyword arguments."""
        kwargs: dict[str, Any] = {"dest": self.dest}
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.kind is Kind.BOOL:
            kwargs["action"] = "store_false" if self.default is True else "store_true"
        elif self.kind is Kind.BOOL_PAIR:
            kwargs["action"] = "store_true"
        elif self.kind is Kind.COUNT:
            kwargs["action"] = "count"
        elif self.kind is Kind.CONST:
            kwargs["action"] = "store_const"
            kwargs["const"] = self.const
        elif self.kind in (Kind.PATH_LIST, Kind.FLOAT_LIST, Kind.STRING_LIST):
            kwargs["action"] = "append"
            kwargs["type"] = _KIND_ARG_TYPES[self.kind]
        else:
            kwargs["type"] = _KIND_ARG_TYPES[self.kind]
        kwargs["help"] = argparse.SUPPRESS if self.is_hidden else self.help
        # Run-scoped options must not be materialized by argparse: presence is
        # the load-bearing information (D-190). Global/internal options keep a
        # real default because the entry point reads them before the handler.
        kwargs["default"] = argparse.SUPPRESS if self.scope is Scope.RUN else self.default
        return kwargs

    def value_check(self, values: Mapping[str, object], *, config_mode: bool = False) -> None:
        """Run this option's check, if any, as a classified usage error."""
        if self.check is None:
            return
        message = self.check(values)
        if message is not None:
            raise UsageError(f"{self.spelling_for(config_mode)} {message}")


@dataclass(frozen=True)
class RunPolicy:
    """The option surface of one subcommand."""

    command: str
    spec: tuple[RunOption, ...]
    config: bool = False
    retired_keys: tuple[tuple[str, str], ...] = ()
    """``(old TOML key, why it is gone)`` pairs, reported with a pointer.

    The command-line spelling of a renamed option is caught by its declared
    ``retired`` entry; this is the TOML counterpart, so both surfaces fail with
    a migration hint instead of a generic unknown-key message (D-188/D-190).
    """
    validate: Callable[[Mapping[str, object], frozenset[str], bool], None] | None = None
    """Cross-field validation ``(resolved, present, config_mode)``; raises
    ``UsageError``. ``present`` holds the dests the source actually supplied, so
    a rule such as "calib_only does not use --sim" can be expressed once."""
    validate_nested: Callable[[Mapping[str, object], frozenset[str], bool], None] | None = None
    """Cross-field validation of one per-dataset/per-run table (config mode)."""


# --------------------------------------------------------------------------
# Shared option fragments
# --------------------------------------------------------------------------
def output_options(
    *,
    with_plot: bool,
    default_hint: str | None = None,
    nested_key: str | None = None,
    config_key: str | None = "output",
) -> tuple[RunOption, ...]:
    """``-o/--output``, ``-f/--force`` and, when plotted, ``--no-plot`` (D-142).

    ``nested_key``/``config_key`` select where the TOML table carries the value:
    the batch commands keep ``output`` inside their per-run/per-dataset table
    (``config_key=None``), while ``compose``/``calib``/``unfold`` carry it at the
    top level (D-134/D-139).
    """
    hint = default_hint or "work/<command>/ under the repository root"
    options = [
        RunOption(
            dest="output",
            flags=("-o", "--output"),
            kind=Kind.PATH,
            metavar="FILE",
            config_key=config_key,
            nested_key=nested_key,
            help=f"output product path (default: {hint})",
        ),
        # --force may accompany --config (D-130), so it is GLOBAL and OR-ed with
        # the config value by the handler.
        RunOption(
            dest="force",
            flags=("-f", "--force"),
            kind=Kind.BOOL,
            default=False,
            scope=Scope.GLOBAL,
            config_key="force",
            merge=Merge.OR,
            help="overwrite an existing output file (default: refuse)",
        ),
    ]
    if with_plot:
        options.append(
            RunOption(
                dest="no_plot",
                flags=("--no-plot",),
                kind=Kind.BOOL,
                default=False,
                config_key="no_plot",
                help="skip the plot report (plots are produced by default)",
            )
        )
    return tuple(options)


def runtime_options() -> tuple[RunOption, ...]:
    """``--strict`` and ``--log-level`` (D-67); both may accompany ``--config``."""
    from kc761tool.runtime import LOG_LEVELS

    return (
        RunOption(
            dest="strict",
            flags=("--strict",),
            kind=Kind.BOOL,
            default=False,
            scope=Scope.GLOBAL,
            help="enable all runtime certificate suites (or set KC761TOOL_STRICT=1)",
        ),
        RunOption(
            dest="log_level",
            flags=("--log-level",),
            kind=Kind.STRING,
            default="info",
            scope=Scope.GLOBAL,
            choices=tuple(LOG_LEVELS),
            help="logging verbosity (default: info)",
        ),
    )


def config_options() -> tuple[RunOption, ...]:
    """``-c/--config`` and ``--dry-run`` (D-129/D-130)."""
    return (
        RunOption(
            dest="config",
            flags=("-c", "--config"),
            kind=Kind.PATH,
            metavar="FILE",
            scope=Scope.GLOBAL,
            help=(
                "run from a TOML configuration file; mutually exclusive with the "
                "run-selection options (only {global} may accompany it)"
            ),
        ),
        RunOption(
            dest="dry_run",
            flags=("--dry-run",),
            kind=Kind.BOOL,
            default=False,
            scope=Scope.GLOBAL,
            config_key="dry_run",
            merge=Merge.OR,
            help="print the resolved invocation and exit without side effects",
        ),
    )


def energy_window_options(
    *,
    requirement: Requirement,
    required_if: Callable[[Mapping[str, object]], bool] | None = None,
) -> tuple[RunOption, ...]:
    """``--energy-low/--elo`` and ``--energy-high/--ehi`` (D-187 wording)."""
    return (
        RunOption(
            dest="energy_low",
            flags=("--energy-low", "--elo"),
            kind=Kind.FLOAT,
            metavar="KEV",
            help=(
                "lower energy bound of the energy window in keV "
                "(solve space and report range, D-187)"
            ),
            config_key="energy_low",
            requirement=requirement,
            required_if=required_if,
        ),
        RunOption(
            dest="energy_high",
            flags=("--energy-high", "--ehi"),
            kind=Kind.FLOAT,
            metavar="KEV",
            help=(
                "upper energy bound of the energy window in keV "
                "(solve space and report range, D-187)"
            ),
            config_key="energy_high",
            requirement=requirement,
            required_if=required_if,
        ),
    )


def channel_window_options(*, requirement: Requirement) -> tuple[RunOption, ...]:
    """``--channel-low/--chlo`` and ``--channel-high/--chhi``."""
    return (
        RunOption(
            dest="channel_low",
            flags=("--channel-low", "--chlo"),
            kind=Kind.INT,
            metavar="CHANNEL",
            help="lower fit channel (0-based, inclusive)",
            nested_key="channel_low",
            requirement=requirement,
        ),
        RunOption(
            dest="channel_high",
            flags=("--channel-high", "--chhi"),
            kind=Kind.INT,
            metavar="CHANNEL",
            help="upper fit channel (0-based, inclusive)",
            nested_key="channel_high",
            requirement=requirement,
        ),
    )


# --------------------------------------------------------------------------
# Parser construction
# --------------------------------------------------------------------------
def validate_policy(policy: RunPolicy) -> None:
    """Structural invariants of a declaration (checked when the parser is built).

    ``CONST`` entries may share one dest (the sim source keys are alternative
    spellings of a single selection); every other dest must be unique, and flag
    spellings are unique across the whole surface.
    """
    by_dest: dict[str, list[RunOption]] = {}
    seen_flag: set[str] = set()
    seen_config: set[str] = set()
    for option in policy.spec:
        by_dest.setdefault(option.dest, []).append(option)
        for flag in (*option.flags, *option.negative_flags):
            if flag in seen_flag:
                raise AssertionError(f"{policy.command}: duplicate flag {flag!r}")
            seen_flag.add(flag)
        if option.config_key is not None:
            if option.config_key in seen_config:
                raise AssertionError(
                    f"{policy.command}: duplicate config key {option.config_key!r}"
                )
            seen_config.add(option.config_key)
        if option.kind is Kind.CONST and option.const is None:
            raise AssertionError(f"{policy.command}: CONST {option.dest!r} needs a const")
        if option.kind is Kind.BOOL_PAIR and not option.negative_flags:
            raise AssertionError(
                f"{policy.command}: BOOL_PAIR {option.dest!r} needs a negative flag"
            )
        if (
            option.kind is not Kind.CONST
            and not option.flags
            and option.config_key is None
            and option.nested_key is None
        ):
            raise AssertionError(f"{policy.command}: {option.dest!r} has no flag and no TOML key")
        if option.config_key and option.nested_key:
            raise AssertionError(
                f"{policy.command}: {option.dest!r} declares both config_key and nested_key"
            )
        if option.configurable and option.kind in (
            Kind.PATH_LIST,
            Kind.FLOAT_LIST,
            Kind.STRING_LIST,
        ):
            raise AssertionError(
                f"{policy.command}: {option.dest!r} is repeatable on the command line and "
                "cannot be carried by a scalar TOML key; declare a batch twin instead"
            )
    for dest, options in by_dest.items():
        if len(options) == 1:
            continue
        if any(option.kind is not Kind.CONST for option in options):
            raise AssertionError(f"{policy.command}: duplicate dest {dest!r} is not a CONST group")
        consts = [option.const for option in options]
        if len(set(consts)) != len(consts):
            raise AssertionError(f"{policy.command}: duplicate const in {dest!r}")
        if len({option.default for option in options}) != 1:
            raise AssertionError(f"{policy.command}: inconsistent defaults for {dest!r}")


def _global_spellings(policy: RunPolicy) -> str:
    """The GLOBAL options of a command, for the ``--config`` help text (D-130)."""
    names = sorted(
        option.spelling
        for option in policy.spec
        if option.scope is Scope.GLOBAL and option.dest != "config"
    )
    return ", ".join(names)


def add_run_options(parser: argparse.ArgumentParser, policy: RunPolicy) -> None:
    """Register every option of ``policy`` on ``parser``.

    The ``{global}`` placeholder in an option's help text is replaced by the
    GLOBAL options of the same declaration, so the ``--config`` help can never
    list a different set than the one the policy actually allows.
    """
    validate_policy(policy)
    globals_text = _global_spellings(policy)
    groups: dict[str, Any] = {}
    for option in policy.spec:
        if not option.flags:
            continue  # config-only entry (no command-line spelling)
        kwargs = option.argparse_kwargs()
        if isinstance(kwargs.get("help"), str):
            kwargs["help"] = kwargs["help"].replace("{global}", globals_text)
        target: Any = parser
        if option.group is not None:
            if option.group not in groups:
                groups[option.group] = parser.add_mutually_exclusive_group(required=False)
            target = groups[option.group]
        target.add_argument(*option.flags, **kwargs)
        if option.kind is Kind.BOOL_PAIR:
            negative = dict(kwargs)
            negative["action"] = "store_false"
            if option.negative_help is not None:
                negative["help"] = option.negative_help
            target.add_argument(*option.negative_flags, **negative)


def provided_options(policy: RunPolicy, args: argparse.Namespace) -> dict[str, RunOption]:
    """The options the user actually supplied, by dest (never value-based).

    For a ``CONST`` group the entry whose const matches the parsed value is
    returned, so messages name the flag the user typed.
    """
    provided: dict[str, RunOption] = {}
    for option in policy.spec:
        if option.scope is not Scope.RUN or not hasattr(args, option.dest):
            continue
        value = getattr(args, option.dest)
        if option.kind is Kind.CONST and value != option.const:
            continue
        provided[option.dest] = option
    return provided


def _missing_requirements(
    policy: RunPolicy,
    values: Mapping[str, object],
    *,
    config_mode: bool,
    provided: Mapping[str, RunOption],
) -> list[str]:
    missing: list[str] = []
    for option in policy.spec:
        if option.scope is Scope.INTERNAL or option.requirement is Requirement.NEVER:
            continue
        required = option.requirement is Requirement.ALWAYS or (
            option.requirement is Requirement.ARGS and not config_mode
        )
        if required and option.required_if is not None and not option.required_if(values):
            required = False
        if not required or option.dest in provided:
            continue
        if values.get(option.dest) is None:
            missing.append(option.spelling_for(config_mode))
    return missing


def check_policy(
    policy: RunPolicy,
    values: Mapping[str, object],
    *,
    config_mode: bool,
    provided: Mapping[str, RunOption] | None = None,
    present: frozenset[str] | None = None,
    context: str | None = None,
) -> None:
    """Reject retired spellings, config conflicts, missing and invalid values.

    Shared by :func:`resolve_run_options` (command line) and the TOML loaders, so
    a value can never be valid on one path and invalid on the other.

    ``provided`` carries the *command-line* options (used for the retired-spelling
    and config-conflict rules, which are only meaningful for what the user typed);
    ``present`` carries the dests the source actually supplied, whichever surface
    that is, and feeds the cross-field validators. Inside a config file the
    conflict rule must not fire, so ``provided`` stays empty there.
    """
    label = context if context is not None else f"kc761tool {policy.command}"
    supplied = dict(provided or {})
    retired = [option for option in supplied.values() if option.retired]
    if retired:
        option = retired[0]
        raise UsageError(f"{label}: {option.flags[0]} {option.retired}")
    if config_mode:
        conflicting = sorted(
            option.spelling for option in supplied.values() if option.scope is Scope.RUN
        )
        if conflicting:
            raise UsageError(
                f"{label}: --config is mutually exclusive with the "
                f"run-selection options; remove: {', '.join(conflicting)}"
            )
    supplied_dests = frozenset(present) if present is not None else frozenset(supplied)
    missing = _missing_requirements(policy, values, config_mode=config_mode, provided=supplied)
    if missing:
        raise UsageError(f"{label}: required option(s) missing: {', '.join(missing)}")
    for option in policy.spec:
        if option.check is not None:
            option.value_check(values, config_mode=config_mode)
    if policy.validate is not None:
        policy.validate(values, supplied_dests, config_mode)


def resolve_run_options(
    policy: RunPolicy,
    args: argparse.Namespace,
    *,
    config_values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve defaults, provided values and (optionally) config values.

    ``config_values`` is the mapping a TOML loader produced; when it is given the
    run options may not also appear on the command line (D-130).
    """
    supplied = provided_options(policy, args)
    resolved: dict[str, object] = {}
    for option in policy.spec:
        if option.scope is Scope.RUN:
            resolved[option.dest] = (
                getattr(args, option.dest) if option.dest in supplied else option.default
            )
        else:
            resolved[option.dest] = getattr(args, option.dest, option.default)
    config_mode = config_values is not None
    if config_values is not None:
        for option in policy.spec:
            key = option.config_key
            if key is None or key not in config_values:
                continue
            value = config_values[key]
            if option.merge_mode is Merge.OR:
                resolved[option.dest] = bool(resolved[option.dest] or value)
            elif option.merge_mode is Merge.CONFIG:
                resolved[option.dest] = value
    check_policy(policy, resolved, config_mode=config_mode, provided=supplied)
    return resolved


# --------------------------------------------------------------------------
# Config-schema accessors (the TOML side derives from the same declaration)
# --------------------------------------------------------------------------
def retired_toml_keys(policy: RunPolicy) -> dict[str, str]:
    """Old TOML key -> migration message, for the loaders' key check."""
    return dict(policy.retired_keys)


def config_keys(policy: RunPolicy) -> dict[str, RunOption]:
    """Top-level TOML key -> option for this command."""
    return {option.config_key: option for option in policy.spec if option.config_key is not None}


def nested_config_keys(policy: RunPolicy) -> dict[str, RunOption]:
    """Per-dataset/per-run TOML key -> option for this command."""
    return {option.nested_key: option for option in policy.spec if option.nested_key is not None}


def read_config_value(
    table: Mapping[str, object],
    option: RunOption,
    *,
    context: str,
    key: str | None = None,
) -> object:
    """Read a key of ``table`` using the option's declared type and default."""
    from kc761tool.cli.config import read_toml_value  # local import: no cycle

    name = key or option.config_key or option.nested_key
    if name is None:
        raise AssertionError(f"{option.dest!r} has no TOML key")
    if name not in table:
        return option.default
    return read_toml_value(table, name, option.toml_type(), context)


def options_by_dest(policy: RunPolicy) -> dict[str, RunOption]:
    """Map dest -> option (the first entry of a CONST group)."""
    by_dest: dict[str, RunOption] = {}
    for option in policy.spec:
        by_dest.setdefault(option.dest, option)
    return by_dest


def nested_policy(policy: RunPolicy) -> RunPolicy:
    """The per-dataset/per-run subset of a declaration, as its own policy.

    Inside a ``[[command.datasets]]``/``[[sim.runs]]`` table the same
    requirement, bound and cross-field rules apply as on the command line, so
    the loader drives the same :func:`check_policy` on this subset.
    """
    nested: list[RunOption] = []
    for option in policy.spec:
        if option.nested_key is None:
            continue
        if option.nested_requirement is not None:
            option = replace(option, requirement=option.nested_requirement)
        nested.append(option)
    return RunPolicy(
        command=policy.command,
        spec=tuple(nested),
        config=True,
        validate=policy.validate_nested,
    )


def merge_options(*groups: Sequence[RunOption]) -> tuple[RunOption, ...]:
    """Concatenate shared fragments and command-specific options into one spec."""
    return tuple(option for group in groups for option in group)
