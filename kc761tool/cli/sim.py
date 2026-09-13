"""``kc761tool sim``: Geant4 simulation.

Single-run mode keeps the frozen flag surface: one source key XOR one matrix
mode (``--plane-front-gamma``/``--sphere-gamma`` with the calibration product),
``-n`` events (omit for an interactive source-mode session), ``-t`` workers,
``-s`` seed, ``-v`` verbosity, ``-o``/``-f`` and the runtime options.

Config mode (D-134..D-137) expands ``[[sim.runs]]`` into one child process per
run, ``sys.executable -m kc761tool sim ...``, because a ``G4RunManager`` can be
initialized only once per process. The parent never imports Geant4; it only
validates, expands argv, manages resume and aggregates the exit code.

The option surface is declared once in :data:`SIM_POLICY` (D-190). The source
keys are ``store_const`` entries sharing one dest, and the per-run table keys
(``source``/``mode``/``calib``/``events``/``threads``/``seed``/``verbose``/
``output``) are declared next to the flags they mirror, so the two surfaces
cannot drift.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from kc761tool.cli._common import REPO_ROOT, argv_arguments, default_output
from kc761tool.cli._registry import (
    Kind,
    Requirement,
    RunOption,
    RunPolicy,
    Scope,
    add_run_options,
    config_options,
    merge_options,
    output_options,
    resolve_run_options,
    runtime_options,
)
from kc761tool.cli.config import SimRunSpec, load_sim_config
from kc761tool.errors import Kc761toolError, UsageError
from kc761tool.runtime import configure_logging
from kc761tool.schema.io import validate_output_path, verify_product
from kc761tool.sim import MATRIX_MODE_NAMES, SOURCE_KEYS
from kc761tool.sim.config import DEFAULT_SEED

_SOURCE_FLAG_ORDER: tuple[str, ...] = tuple(sorted(SOURCE_KEYS))


def _matrix_flag(mode: str) -> str:
    """Matrix-mode token -> CLI flag (D-143/D-2); the token is the dest."""
    return f"--{mode}"


def _validate_sim(values: Mapping[str, object], present: frozenset[str], config_mode: bool) -> None:
    """Selection rules of a single simulation (D-2/D-134), shared with the batch.

    In config mode the per-run tables carry the selection, so this only checks
    the batch-level mapping (``force``/``dry_run``/``resume``).
    """
    if config_mode:
        return
    selected = [
        name
        for name in ("source_key", "plane_front_gamma", "sphere_gamma")
        if values.get(name) is not None
    ]
    if len(selected) != 1:
        sources = ", ".join(f"--{key}" for key in _SOURCE_FLAG_ORDER)
        raise UsageError(
            f"select exactly one source ({sources}) or a matrix mode "
            "(--plane-front-gamma/--sphere-gamma)"
        )
    del present
    matrix = values.get("plane_front_gamma") or values.get("sphere_gamma")
    events = values.get("events")
    if matrix is not None and events is None:
        raise UsageError("matrix modes require --events (only the source mode is interactive)")
    if matrix is None and events is None and values.get("output") is not None:
        raise UsageError(
            "the interactive source session produces no product; omit --output "
            "or pass --events for a batch run"
        )


def _validate_run(values: Mapping[str, object], present: frozenset[str], config_mode: bool) -> None:
    """Rules of one ``[[sim.runs]]`` entry (D-134/D-137)."""
    del present, config_mode
    has_source = values.get("source") is not None
    has_mode = values.get("mode") is not None
    if has_source == has_mode:
        raise UsageError("specify exactly one of 'source' or 'mode'")
    if has_source and values.get("matrix_calib") is not None:
        raise UsageError("a source run must not set 'calib'")
    if has_mode and values.get("matrix_calib") is None:
        raise UsageError("a matrix run requires 'calib'")


def _source_modes() -> list[RunOption]:
    """The source keys, declared as ``store_const`` flags sharing one dest."""
    return [
        RunOption(
            dest="source_key",
            flags=(f"--{key}",),
            kind=Kind.CONST,
            const=key,
            group="selection",
            help=f"simulate the {key} source",
        )
        for key in _SOURCE_FLAG_ORDER
    ]


#: Cosmetic help text per matrix mode (the tokens are frozen by sim D-143).
_MATRIX_HELP: dict[str, str] = {
    "plane-front-gamma": "matrix mode: plane source at the housing front surface",
    "sphere-gamma": "matrix mode: circumscribed-sphere source",
}


def _matrix_modes() -> list[RunOption]:
    return [
        RunOption(
            dest=mode.replace("-", "_"),
            flags=(_matrix_flag(mode),),
            kind=Kind.PATH,
            metavar="CALIB",
            group="selection",
            help=_MATRIX_HELP[mode],
        )
        for mode in MATRIX_MODE_NAMES
    ]


#: Declared surface (D-190): flags, per-run keys, defaults and the shared rules.
SIM_POLICY = RunPolicy(
    command="sim",
    config=True,
    validate=_validate_sim,
    validate_nested=_validate_run,
    spec=merge_options(
        _source_modes(),
        _matrix_modes(),
        [
            RunOption(
                dest="events",
                flags=("-n", "--events"),
                kind=Kind.INT,
                metavar="N",
                nested_key="events",
                nested_requirement=Requirement.ALWAYS,
                check=lambda v: (
                    None
                    if v.get("events") is None or v["events"] >= 1
                    else f"must be >= 1, got {v['events']!r}"
                ),
                help="number of events; omit for an interactive session (source mode only)",
            ),
            RunOption(
                dest="threads",
                flags=("-t", "--threads"),
                kind=Kind.INT,
                metavar="N",
                nested_key="threads",
                check=lambda v: (
                    None
                    if v.get("threads") is None or v["threads"] >= 1
                    else f"must be >= 1, got {v['threads']!r}"
                ),
                help="worker processes (default: memory-budgeted CPU count, D-38)",
            ),
            RunOption(
                dest="seed",
                flags=("-s", "--seed"),
                kind=Kind.INT,
                default=DEFAULT_SEED,
                metavar="SEED",
                nested_key="seed",
                help=(
                    f"base random seed (default {DEFAULT_SEED}); per-column/per-block "
                    "streams (F-SIM-7)"
                ),
            ),
            RunOption(
                dest="verbose",
                flags=("-v", "--verbose"),
                kind=Kind.COUNT,
                default=0,
                nested_key="verbose",
                check=lambda v: (
                    None
                    if v.get("verbose") is None or v["verbose"] >= 0
                    else f"must be >= 0, got {v['verbose']!r}"
                ),
                help="increase Geant4 verbosity (repeatable)",
            ),
            RunOption(
                dest="provenance_input",
                flags=("--provenance-input",),
                kind=Kind.STRING_LIST,
                scope=Scope.INTERNAL,
                metavar="FILE",
            ),
            # Config-only keys: batch-level resume, and the per-run selection
            # that the command line expresses as mutually exclusive flags.
            RunOption(dest="resume", kind=Kind.BOOL, default=True, config_key="resume"),
            RunOption(
                dest="source",
                kind=Kind.STRING,
                nested_key="source",
                check=lambda v: (
                    None
                    if v.get("source") is None or v["source"] in SOURCE_KEYS
                    else f"unknown source key {v['source']!r}; expected one of {tuple(SOURCE_KEYS)}"
                ),
            ),
            RunOption(
                dest="mode",
                kind=Kind.STRING,
                nested_key="mode",
                check=lambda v: (
                    None
                    if v.get("mode") is None or v["mode"] in MATRIX_MODE_NAMES
                    else f"must be one of {tuple(MATRIX_MODE_NAMES)}, got {v['mode']!r}"
                ),
            ),
            RunOption(dest="matrix_calib", kind=Kind.PATH, nested_key="calib"),
        ],
        output_options(with_plot=False, nested_key="output", config_key=None),
        config_options(),
        runtime_options(),
    ),
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sim",
        help="run the Geant4 simulation (source mode or matrix modes)",
        description=(
            "Simulate either a radioactive source in its container (batch, or "
            "interactive when --events is omitted) or one of the matrix modes "
            "with a calibration product. With -c/--config a TOML file drives a "
            "serial batch of runs, each in its own child process."
        ),
    )
    add_run_options(parser, SIM_POLICY)
    parser.set_defaults(handler=_run)


def _matrix_selection(values: Mapping[str, object]) -> tuple[str, Path] | None:
    """The selected matrix mode and calibration product, if any (D-143)."""
    for mode in MATRIX_MODE_NAMES:
        calib = values.get(mode.replace("-", "_"))
        if calib is not None:
            return mode, Path(str(calib)).expanduser()
    return None


def _resolve_output(explicit: object, filename: str) -> Path:
    if explicit is not None:
        return Path(str(explicit)).expanduser()
    return default_output("sim", filename)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("sim", args.log_level)
    config = load_sim_config(args.config) if args.config is not None else None
    values = resolve_run_options(
        SIM_POLICY, args, config_values=config.values() if config is not None else None
    )
    if config is not None:
        return _run_batch(config, values, strict=strict, logger=logger)

    events = values["events"]
    seed = int(values["seed"])
    matrix = _matrix_selection(values)
    if matrix is not None:
        mode, calib = matrix
        output = _resolve_output(values["output"], f"{calib.stem}-{mode}-n{events}-s{seed}.root")
        if values["dry_run"]:
            _print_dry_run(values, mode, calib, output)
            return 0
        force = bool(values["force"])
        validate_output_path(output, force=force)
        from kc761tool.sim.runner import run_matrix

        run_matrix(
            mode,
            calib,
            output,
            n_events=int(events),
            seed=seed,
            threads=values["threads"],
            force=force,
            strict=strict,
            verbose=int(values["verbose"]),
            command="kc761tool sim",
            arguments=argv_arguments(args.argv),
            extra_inputs=values["provenance_input"] or (),
        )
        logger.info("wrote %s", output)
        return 0

    if events is None:
        if values["dry_run"]:
            print(f"kc761tool sim (dry-run): interactive {values['source_key']} session")
            return 0
        _run_interactive(values)
        return 0

    source_key = str(values["source_key"])
    output = _resolve_output(values["output"], f"{source_key}-n{events}-s{seed}.root")
    if values["dry_run"]:
        _print_dry_run(values, source_key, None, output)
        return 0
    force = bool(values["force"])
    validate_output_path(output, force=force)
    from kc761tool.sim.runner import run_source

    run_source(
        source_key,
        output,
        n_events=int(events),
        seed=seed,
        threads=values["threads"],
        force=force,
        strict=strict,
        verbose=int(values["verbose"]),
        command="kc761tool sim",
        arguments=argv_arguments(args.argv),
        extra_inputs=values["provenance_input"] or (),
    )
    logger.info("wrote %s", output)
    return 0


def _print_dry_run(
    values: Mapping[str, object],
    selection: str,
    calib: Path | None,
    output: Path,
) -> None:
    target = calib if calib is not None else "-"
    print(
        "kc761tool sim (dry-run): "
        f"selection={selection} calib={target} events={values['events']} "
        f"threads={values['threads']} seed={values['seed']} verbose={values['verbose']} "
        f"strict={'--strict' if values['strict'] else 'no'} force={values['force']} "
        f"output={output}"
    )


def _run_interactive(values: Mapping[str, object]) -> None:
    from geant4_pybind import G4UIExecutive, G4UImanager, G4VisExecutive

    from kc761tool.sim.runner import macro_path, prepare_interactive

    prepare_interactive(
        str(values["source_key"]), seed=int(values["seed"]), verbose=int(values["verbose"])
    )
    vis_manager = G4VisExecutive("quiet")
    vis_manager.Initialize()
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(f"/control/execute {macro_path('init_vis.mac')}")
    ui.ApplyCommand(f"/control/execute {macro_path('vis.mac')}")
    session = G4UIExecutive(len(sys.argv), sys.argv)
    if session.IsGUI():
        ui.ApplyCommand(f"/control/execute {macro_path('gui.mac')}")
    session.SessionStart()


# --------------------------------------------------------------------------
# Config-mode batch driver (D-134..D-137)
# --------------------------------------------------------------------------
def _run_batch(config, values: Mapping[str, object], *, strict: bool, logger) -> int:
    """Expand every ``[[sim.runs]]`` entry into one child process (D-134..D-137)."""
    effective_force = bool(values["force"])
    dry_run = bool(values["dry_run"])
    failures = 0
    for index, run in enumerate(config.runs):
        target = _run_output(run)
        command = _expand_run_argv(
            run,
            target,
            strict=strict,
            force=effective_force,
            log_level=str(values["log_level"]),
            config_path=config.config_path,
        )
        if dry_run:
            print(f"kc761tool sim (dry-run) [run {index}]: {shlex.join(command)}")
            continue
        if values["resume"] and target.exists():
            try:
                verify_product(target, strict=False)
            except Kc761toolError as exc:
                logger.error("[run %d] existing output %s is invalid: %s", index, target, exc)
                failures += 1
                continue
            logger.info("[run %d] resume: %s is valid; skipping", index, target)
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            logger.error("[run %d] failed with exit code %d", index, completed.returncode)
            failures += 1
        else:
            logger.info("[run %d] wrote %s", index, target)
    return 1 if failures else 0


def _run_output(run: SimRunSpec) -> Path:
    if run.output is not None:
        return run.output
    if run.source_key is not None:
        return default_output("sim", f"{run.source_key}-n{run.events}-s{run.seed}.root")
    assert run.calib is not None
    assert run.matrix_mode is not None
    return default_output(
        "sim",
        f"{run.calib.stem}-{run.matrix_mode}-n{run.events}-s{run.seed}.root",
    )


def _expand_run_argv(
    run: SimRunSpec,
    target: Path,
    *,
    strict: bool,
    force: bool,
    log_level: str,
    config_path: Path,
) -> list[str]:
    command = [sys.executable, "-m", "kc761tool", "sim"]
    if run.source_key is not None:
        command.append(f"--{run.source_key}")
    else:
        assert run.matrix_mode is not None
        assert run.calib is not None
        command.append(_matrix_flag(run.matrix_mode))
        command.append(str(run.calib))
    command += ["-n", str(run.events)]
    if run.threads is not None:
        command += ["-t", str(run.threads)]
    command += ["-s", str(run.seed)]
    if run.verbose:
        command += ["-v"] * run.verbose
    command += ["-o", str(target)]
    if force:
        command.append("--force")
    if strict:
        command.append("--strict")
    command += ["--log-level", log_level, "--provenance-input", str(config_path)]
    return command
