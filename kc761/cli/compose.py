"""``kc761 compose``: build the primary-to-channel response (W6).

Writes the inspection artifact ``R = C . p_tilde . diag(eta)`` from a
calibration product and a matrix-mode simulation product (D-43). Composition
itself is the ``core`` function F-RESP-2; this command only resolves inputs,
the default output name and provenance. Config mode runs one compose (D-139).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kc761.cli._common import (
    add_config_options,
    add_output_options,
    add_runtime_options,
    argv_arguments,
    default_output_beside,
    reject_run_options,
)
from kc761.cli.config import load_compose_config
from kc761.errors import UsageError
from kc761.runtime import configure_logging
from kc761.schema.io import validate_output_path

_RUN_ARG_DEFAULTS: dict[str, object] = {
    "calib": None,
    "sim": None,
    "output": None,
}


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
        help="matrix-mode simulation product with the primary-to-deposition matrix",
    )
    add_output_options(
        parser,
        with_plot=False,
        default_hint="next to the matrix simulation product (--sim)",
    )
    add_config_options(parser)
    add_runtime_options(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace, *, strict: bool) -> int:
    logger = configure_logging("compose", args.log_level)
    if args.config is not None:
        reject_run_options(args, _RUN_ARG_DEFAULTS, command="compose")
        config = load_compose_config(args.config)
        output = _output(config.output, config.calib, config.sim)
        if args.dry_run or config.dry_run:
            _print_dry_run(config.calib, config.sim, output)
            return 0
        validate_output_path(output, force=bool(args.force or config.force))
        with_config = (config.config_path,)
        return _execute(
            config.calib,
            config.sim,
            output=output,
            force=bool(args.force or config.force),
            strict=strict,
            logger=logger,
            arguments=argv_arguments(args.argv),
            config_inputs=with_config,
        )

    if args.calib is None or args.sim is None:
        raise UsageError("--calib and --sim are required")
    output = _output(args.output, Path(args.calib), Path(args.sim))
    if args.dry_run:
        _print_dry_run(Path(args.calib), Path(args.sim), output)
        return 0
    validate_output_path(output, force=args.force)
    return _execute(
        Path(args.calib),
        Path(args.sim),
        output=output,
        force=args.force,
        strict=strict,
        logger=logger,
        arguments=argv_arguments(args.argv),
        config_inputs=(),
    )


def _output(explicit: str | Path | None, calib: Path, sim: Path) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    return default_output_beside(sim, f"compose-{calib.stem}-{sim.stem}.root")


def _print_dry_run(calib: Path, sim: Path, output: Path) -> None:
    print("kc761 compose (dry-run):")
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
    from kc761.unfold import run_compose

    result = run_compose(
        calib,
        sim,
        output=output,
        force=force,
        strict=strict,
        command="kc761 compose",
        arguments=arguments,
        extra_inputs=config_inputs,
    )
    logger.info("wrote %s", result.product_path)
    return 0


__all__ = ["add_parser"]
