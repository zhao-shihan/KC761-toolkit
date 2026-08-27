#!/usr/bin/env python3
"""Geant4 gamma-spectrometry Monte Carlo of a CsI(Tl) probe with fixed sources."""

from __future__ import annotations

import argparse
import os
import sys

from geant4_pybind import G4UIExecutive, G4UImanager, G4VisExecutive
from kc761sim import config, runner
from kc761sim.paths import final_output_path, output_stem

_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kc761sim", "script"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sim.py",
        description=(
            "Geant4 gamma-spectrometry Monte Carlo simulation: CsI(Tl) probe "
            "with fixed radioactive sources; per-event energy deposition is "
            "saved to a ROOT ntuple."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    for key, spec in config.SOURCES.items():
        sources.add_argument(f"--{key}", action="store_true", help=spec.name)

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="output ROOT file name (default: sim_output.root in batch mode, "
        "sim_vis_output.root in interactive mode; a missing .root suffix is "
        "appended)",
    )
    parser.add_argument(
        "-n",
        "--events",
        type=int,
        metavar="N",
        help="number of events to simulate; if omitted, an interactive Geant4 "
        "(visualization) session is started instead",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        metavar="N",
        help="number of worker processes for batch mode "
        "(default: number of CPUs)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=runner.DEFAULT_SEED,
        metavar="SEED",
        help="base random seed; worker i uses SEED + i + 1",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        metavar="N",
        help="Geant4 verbosity level (default: 0 in batch mode, 1 in "
        "interactive mode)",
    )
    return parser.parse_args(argv)


def _selected_source(args: argparse.Namespace) -> str:
    for key in config.SOURCES:
        if getattr(args, key):
            return key
    raise SystemExit("error: no source selected")


def interactive_mode(source_key: str, seed: int, verbose: int, output_stem_: str) -> None:
    """Start an initialized Geant4 UI/visualization session."""
    run_manager = runner.prepare_run_manager(
        config.SOURCES[source_key], output_stem_, seed=seed, verbose=verbose
    )

    vis_manager = G4VisExecutive("quiet")
    vis_manager.Initialize()

    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(
        f"/control/execute {os.path.join(_SCRIPT_DIR, 'init_vis.mac')}"
    )
    ui.ApplyCommand(f"/control/execute {os.path.join(_SCRIPT_DIR, 'vis.mac')}")

    ui_session = G4UIExecutive(len(sys.argv), sys.argv)
    if ui_session.IsGUI():
        ui.ApplyCommand(
            f"/control/execute {os.path.join(_SCRIPT_DIR, 'gui.mac')}")
    ui_session.SessionStart()


def batch_mode(args: argparse.Namespace, source_key: str, verbose: int) -> None:
    threads = args.threads if args.threads and args.threads > 0 else max(
        1, os.cpu_count() or 1)
    runner.run_batch(source_key, args.output, args.events,
                     threads, args.seed, verbose)
    print(f"Simulation finished: {final_output_path(args.output)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_key = _selected_source(args)
    if args.events is None:
        if args.output is None:
            args.output = "sim_vis_output.root"
        verbose = args.verbose if args.verbose is not None else 1
        interactive_mode(source_key, args.seed, verbose,
                         output_stem(args.output))
    else:
        if args.events <= 0:
            raise SystemExit("error: --events must be a positive integer")
        if args.output is None:
            args.output = "sim_output.root"
        verbose = args.verbose if args.verbose is not None else 0
        batch_mode(args, source_key, verbose)


if __name__ == "__main__":
    main()
