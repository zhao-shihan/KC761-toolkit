"""``kc761tool compose``: build the primary-to-channel response.

Writes the inspection artifact ``R = C . p_tilde . diag(eta)`` from a
calibration product and a matrix-mode simulation product (D-43). Composition
itself is the ``core`` function F-RESP-2; this command only resolves inputs,
the default output name and provenance. Config mode runs one compose (D-139).

The option surface is declared once in :data:`COMPOSE_POLICY` (D-190), so the
command line and the TOML table cannot drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kc761tool.cli._common import argv_arguments, default_output_beside
from kc761tool.cli._registry import (
    Kind,
    Requirement,
    RunOption,
    RunPolicy,
    add_run_options,
    config_options,
    merge_options,
    output_options,
    resolve_run_options,
    runtime_options,
)
from kc761tool.cli.config import load_compose_config
from kc761tool.runtime import configure_logging
from kc761tool.schema.io import validate_output_path

#: Declared surface: flag -> dest -> TOML key, with one default each (D-190).
COMPOSE_POLICY = RunPolicy(
    command="compose",
    config=True,
    spec=merge_options(
        [
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
                help="matrix-mode simulation product with the primary-to-deposition matrix",
            ),
        ],
        output_options(
            with_plot=False,
            default_hint="next to the matrix simulation product (--sim)",
        ),
        config_options(),
        runtime_options(),
    ),
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compose",
        help="compose the primary-to-channel response matrix R",
        description=(
            "Compose R = C . p_tilde . diag(eta) over the full primary axis "
            "and write the inspection product (R plus the C/G inputs and the "
            "derived efficiency)."
        ),
    )
    add_run_options(parser, COMPOSE_POLICY)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("compose", args.log_level)
    if args.config is not None:
        config = load_compose_config(args.config)
        values = resolve_run_options(COMPOSE_POLICY, args, config_values=config.values())
        config_inputs = (config.config_path,)
    else:
        values = resolve_run_options(COMPOSE_POLICY, args)
        config_inputs = ()

    calib = Path(str(values["calib"])).expanduser()
    sim = Path(str(values["sim"])).expanduser()
    output = _output(values["output"], calib, sim)
    if values["dry_run"]:
        _print_dry_run(calib, sim, output)
        return 0
    force = bool(values["force"])
    validate_output_path(output, force=force)
    return _execute(
        calib,
        sim,
        output=output,
        force=force,
        strict=strict,
        logger=logger,
        arguments=argv_arguments(args.argv),
        config_inputs=config_inputs,
    )


def _output(explicit: object, calib: Path, sim: Path) -> Path:
    if explicit is not None:
        return Path(str(explicit)).expanduser()
    return default_output_beside(sim, f"compose-{calib.stem}-{sim.stem}.root")


def _print_dry_run(calib: Path, sim: Path, output: Path) -> None:
    print("kc761tool compose (dry-run):")
    print(f"  calib={calib}")
    print(f"  sim={sim}")
    print(f"  output={output}")


def _execute(
    calib: Path | str,
    sim: Path | str,
    *,
    output: Path,
    force: bool,
    strict: bool,
    logger,
    arguments,
    config_inputs,
) -> int:
    from kc761tool.unfold import run_compose

    result = run_compose(
        calib,
        sim,
        output=output,
        force=force,
        strict=strict,
        command="kc761tool compose",
        arguments=arguments,
        extra_inputs=config_inputs,
    )
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["COMPOSE_POLICY", "add_parser"]
