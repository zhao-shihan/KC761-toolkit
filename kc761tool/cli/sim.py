"""``kc761tool sim``: Geant4 simulation.

Single-run mode keeps the frozen flag surface: one source key XOR one matrix
mode (``--plane-front-gamma``/``--sphere-gamma`` with the calibration product),
``-n`` events (omit for an interactive source-mode session), ``-t`` workers,
``-s`` seed, ``-v`` verbosity, ``-o``/``-f`` and the runtime options.

Config mode (D-134..D-137) expands ``[[sim.runs]]`` into one child process per
run, ``sys.executable -m kc761tool sim ...``, because a ``G4RunManager`` can be
initialized only once per process. The parent never imports Geant4; it only
validates, expands argv, manages resume and aggregates the exit code.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from kc761tool.cli._common import (
    REPO_ROOT,
    add_config_options,
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output,
    reject_run_options,
)
from kc761tool.cli.config import SimRunSpec, load_sim_config
from kc761tool.errors import Kc761toolError, UsageError
from kc761tool.runtime import configure_logging
from kc761tool.schema.io import validate_output_path, verify_product
from kc761tool.sim import MATRIX_MODE_NAMES, SOURCE_KEYS
from kc761tool.sim.config import DEFAULT_SEED

#: Display order for the source flags and help text: alphabetical, which keeps
#: each ``*-unshielded`` variant next to its base key. The registry order in
#: ``kc761tool.sim.sources`` is unchanged.
_SOURCE_FLAG_ORDER: tuple[str, ...] = tuple(sorted(SOURCE_KEYS))

_MATRIX_FLAGS: dict[str, str] = {
    "plane-front-gamma": "--plane-front-gamma",
    "sphere-gamma": "--sphere-gamma",
}
"""Canonical matrix-mode token -> CLI flag (D-143/D-2)."""

_RUN_ARG_DEFAULTS: dict[str, object] = {
    "source_key": None,
    "plane_front_gamma": None,
    "sphere_gamma": None,
    "events": None,
    "threads": None,
    "seed": DEFAULT_SEED,
    "verbose": 0,
    "output": None,
}


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
    source = parser.add_mutually_exclusive_group(required=False)
    for key in _SOURCE_FLAG_ORDER:
        source.add_argument(
            f"--{key}",
            dest="source_key",
            action="store_const",
            const=key,
            help=f"simulate the {key} source",
        )
    source.add_argument(
        "--plane-front-gamma",
        dest="plane_front_gamma",
        metavar="CALIB",
        help="matrix mode: plane source at the housing front surface",
    )
    source.add_argument(
        "--sphere-gamma",
        dest="sphere_gamma",
        metavar="CALIB",
        help="matrix mode: circumscribed-sphere source",
    )
    parser.add_argument(
        "-n",
        "--events",
        type=int,
        default=None,
        metavar="N",
        help="number of events; omit for an interactive session (source mode only)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="worker processes (default: memory-budgeted CPU count, D-38)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="SEED",
        help=f"base random seed (default {DEFAULT_SEED}); per-column/per-block streams (F-SIM-7)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase Geant4 verbosity (repeatable)",
    )
    parser.add_argument(
        "--provenance-input",
        action="append",
        default=None,
        metavar="FILE",
        help=argparse.SUPPRESS,
    )
    add_output_options(parser, with_plot=False)
    add_config_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _matrix_selection(args: argparse.Namespace) -> tuple[str, str] | None:
    if args.plane_front_gamma is not None:
        return "plane-front-gamma", args.plane_front_gamma
    if args.sphere_gamma is not None:
        return "sphere-gamma", args.sphere_gamma
    return None


def _resolve_output(args: argparse.Namespace, filename: str) -> Path:
    if args.output is not None:
        return Path(args.output).expanduser()
    return default_output("sim", filename)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("sim", args.log_level)
    if args.config is not None:
        reject_run_options(args, _RUN_ARG_DEFAULTS, command="sim")
        return _run_batch(args, strict=strict, logger=logger)
    if args.threads is not None and args.threads < 1:
        raise UsageError(f"--threads must be >= 1, got {args.threads!r}")
    if args.events is not None and args.events < 1:
        raise UsageError(f"--events must be >= 1, got {args.events!r}")

    matrix = _matrix_selection(args)
    source_selected = args.source_key is not None
    if source_selected and matrix is not None:
        raise UsageError("select exactly one source key or matrix mode")
    if not source_selected and matrix is None:
        raise UsageError(
            "select a source (" + ", ".join(f"--{key}" for key in _SOURCE_FLAG_ORDER) + ") "
            "or a matrix mode (--plane-front-gamma/--sphere-gamma)"
        )

    if matrix is not None:
        if args.events is None:
            raise UsageError(
                "matrix modes require --events (only the source mode is interactive)"
            )
        mode, calib = matrix
        output = _resolve_output(
            args, f"{Path(calib).stem}-{mode}-n{args.events}-s{args.seed}.root"
        )
        if args.dry_run:
            _print_dry_run(mode, calib, output, args)
            return 0
        validate_output_path(output, force=args.force)
        from kc761tool.sim.runner import run_matrix

        run_matrix(
            mode,
            calib,
            output,
            n_events=args.events,
            seed=args.seed,
            threads=args.threads,
            force=args.force,
            strict=strict,
            verbose=args.verbose,
            command="kc761tool sim",
            arguments=argv_arguments(args.argv),
            extra_inputs=args.provenance_input or (),
        )
        logger.info("wrote %s", output)
        return 0

    if args.events is None:
        if args.output is not None:
            raise UsageError(
                "the interactive source session produces no product; omit --output "
                "or pass --events for a batch run"
            )
        if args.dry_run:
            print(f"kc761tool sim (dry-run): interactive {args.source_key} session")
            return 0
        _run_interactive(args)
        return 0

    output = _resolve_output(
        args, f"{args.source_key}-n{args.events}-s{args.seed}.root"
    )
    if args.dry_run:
        _print_dry_run(args.source_key, None, output, args)
        return 0
    validate_output_path(output, force=args.force)
    from kc761tool.sim.runner import run_source

    run_source(
        args.source_key,
        output,
        n_events=args.events,
        seed=args.seed,
        threads=args.threads,
        force=args.force,
        strict=strict,
        verbose=args.verbose,
        command="kc761tool sim",
        arguments=argv_arguments(args.argv),
        extra_inputs=args.provenance_input or (),
    )
    logger.info("wrote %s", output)
    return 0


def _print_dry_run(
    selection: str, calib: str | None, output: Path, args: argparse.Namespace
) -> None:
    target = calib if calib is not None else "-"
    print(
        "kc761tool sim (dry-run): "
        f"selection={selection} calib={target} events={args.events} "
        f"threads={args.threads} seed={args.seed} verbose={args.verbose} "
        f"strict={'--strict' if args.strict else 'no'} force={args.force} "
        f"output={output}"
    )


def _run_interactive(args: argparse.Namespace) -> None:
    from geant4_pybind import G4UIExecutive, G4UImanager, G4VisExecutive

    from kc761tool.sim.runner import macro_path, prepare_interactive

    prepare_interactive(args.source_key, seed=args.seed, verbose=args.verbose)
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
def _run_batch(args: argparse.Namespace, *, strict: bool, logger) -> int:
    config = load_sim_config(
        args.config,
        source_keys=SOURCE_KEYS,
        default_seed=DEFAULT_SEED,
        matrix_modes=MATRIX_MODE_NAMES,
    )
    effective_force = bool(args.force or config.force)
    dry_run = bool(args.dry_run or config.dry_run)
    failures = 0
    for index, run in enumerate(config.runs):
        target = _run_output(run)
        command = _expand_run_argv(
            run,
            target,
            strict=strict,
            force=effective_force,
            log_level=args.log_level,
            config_path=config.config_path,
        )
        if dry_run:
            print(f"kc761tool sim (dry-run) [run {index}]: {shlex.join(command)}")
            continue
        if config.resume and target.exists():
            try:
                verify_product(target, strict=False)
            except Kc761toolError as exc:
                logger.error(
                    "[run %d] existing output %s is invalid: %s", index, target, exc
                )
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
        return default_output(
            "sim", f"{run.source_key}-n{run.events}-s{run.seed}.root"
        )
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
        command.append(_MATRIX_FLAGS[run.matrix_mode])
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
