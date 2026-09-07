#!/usr/bin/env python3
"""Geant4 gamma-spectrometry Monte Carlo of a CsI(Tl) probe.

Two simulation families share this entry point:

* radioactive sources (mutually exclusive --<key> flags): the source
  nuclide decays inside its geometry, and per-event energy deposition is
  saved to a ROOT ntuple;
* matrix modes (--plane-front-gamma/--sphere-gamma): one gamma
  per event is launched from a sampling surface, and the (primary energy,
  crystal deposition) pairs accumulate into the primary-to-deposition
  matrix G,
  which is composed with a kc761calib export's deposition-to-channel matrix
  C into the primary-to-channel matrix R = C @ G.  These modes are
  batch-only and produce no
  ntuple; the output ROOT file contains the three matrices
  (``primary_to_channel`` = R, ``deposition_to_channel`` = C copy,
  ``primary_to_deposition`` = G) plus the inherited calibration and
  new mode parameters, and is directly readable by kc761unfold.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from geant4_pybind import G4UIExecutive, G4UImanager, G4VisExecutive

from _bootstrap import REPO_ROOT
from kc761sim import compose, config, detector, runner
from kc761sim.paths import count_label, final_output_path, output_stem
from kc761util.calibfile import load_calib_file
from kc761util.hadd import add_hadd_option
from kc761util.rootcxxfrontend import add_root_option

_OUT_DIR = os.path.join(REPO_ROOT, "out")
_SIM_OUT_DIR = os.path.join(_OUT_DIR, "sim")


# The Geant4 UI macros live inside the kc761sim package, not next to this
# script, so resolve them from the package location.
_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(config.__file__)), "script"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sim.py",
        description=(
            "Geant4 gamma-spectrometry Monte Carlo simulation: CsI(Tl) probe "
            "with fixed radioactive sources (per-event energy deposition is "
            "saved to a ROOT ntuple), or matrix modes launching one "
            "gamma per event from a sampling surface and composing the true "
            "matrix R = C @ G from a kc761calib export (no ntuple; the "
            "output is directly readable by kc761unfold)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    for key, spec in config.SOURCES.items():
        sources.add_argument(
            f"--{key}", dest=key, action="store_true", help=spec.name
        )
    sources.add_argument(
        "--plane-front-gamma",
        dest="plane_front_gamma_calib",
        metavar="CALIB",
        default=None,
        help="matrix mode 1: square plane gamma source on the detector "
        "front surface, Lambertian toward the crystal; CALIB is the "
        "kc761calib export ROOT file providing the deposition-to-channel matrix "
        "(batch-only)",
    )
    sources.add_argument(
        "--sphere-gamma",
        dest="sphere_gamma_calib",
        metavar="CALIB",
        default=None,
        help="matrix mode 2: circumscribed-sphere isotropic gamma source "
        "wrapping the housing, inward-Lambertian; CALIB is the kc761calib "
        "export ROOT file providing the deposition-to-channel matrix (batch-only)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="output ROOT file name (default: out/sim_output.root in batch "
        "mode, out/sim_vis_output.root in interactive mode; matrix modes: "
        "out/sim/<calib>-<mode>-matrix-<N>.root; a missing .root suffix "
        "is appended)",
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
    parser.add_argument(
        "--compose-from",
        dest="compose_from",
        metavar="G_FILE",
        default=None,
        help="matrix modes only: skip the simulation and compose the "
        "composite matrix from an existing merged-G intermediate "
        "(the *-g.root kept after a failed export)",
    )
    add_hadd_option(parser)
    add_root_option(parser)
    return parser.parse_args(argv)


def _selected_source(args: argparse.Namespace) -> str:
    for key in config.SOURCES:
        if getattr(args, key):
            return key
    raise SystemExit("error: no source selected")


def _selected_matrix_mode(args: argparse.Namespace) -> tuple[str, str] | None:
    """Matrix-mode label and calibration path, or None for a radioactive run."""
    if args.plane_front_gamma_calib:
        return "plane-front-gamma", args.plane_front_gamma_calib
    if args.sphere_gamma_calib:
        return "sphere-gamma", args.sphere_gamma_calib
    return None


def interactive_mode(source_key: str, seed: int, verbose: int, output_stem_: str) -> None:
    """Start an initialized Geant4 UI/visualization session."""
    runner.prepare_run_manager(
        config.SOURCES[source_key], output_stem_, seed=seed, verbose=verbose)

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
                     threads, args.seed, verbose, hadd_exe=args.hadd)
    print(f"Simulation finished: {final_output_path(args.output)}")


def batch_matrix_mode(args: argparse.Namespace, mode: str, calib_path: str,
                      verbose: int) -> None:
    """Simulate G, compose R = C @ G and write the composite ROOT file."""
    threads = args.threads if args.threads and args.threads > 0 else max(
        1, os.cpu_count() or 1)

    # Fail fast on the calibration input before launching any worker, so
    # a deterministically invalid input cannot waste the whole batch.
    calib_file = Path(calib_path).expanduser().resolve()
    if not calib_file.is_file():
        raise SystemExit(f"error: calibration file not found: {calib_file}")
    calib = load_calib_file(calib_file)
    if calib.matrix_errors is None:
        raise SystemExit(
            "error: the calibration file stores no per-element errors "
            "(no fSumw2); the composite output needs them for the "
            "deposition response copy and the error propagation")
    if mode == "plane-front-gamma":
        source = detector.build_plane_gamma_source(calib.energy_edges)
    else:
        source = detector.build_sphere_gamma_source(calib.energy_edges)

    if args.output is None:
        os.makedirs(_SIM_OUT_DIR, exist_ok=True)
        args.output = os.path.join(
            _SIM_OUT_DIR,
            f"{calib_file.stem}-{mode}-matrix-{count_label(args.events)}.root")

    # The merged G histogram is an intermediate next to the output; its
    # stem must stay dot-free (the Geant4 analysis manager appends ".root"
    # only to dot-free names).
    g_path = output_stem(args.output) + "-g.root"
    if args.compose_from is not None:
        # Retry path: reuse a kept intermediate instead of re-simulating.
        g_path = str(Path(args.compose_from).expanduser().resolve())
        if not Path(g_path).is_file():
            raise SystemExit(
                f"error: merged-G intermediate not found: {g_path}")
    else:
        runner.run_batch_matrix(source, g_path, args.events, threads,
                                args.seed, verbose, hadd_exe=args.hadd)
    rc = compose.compose_matrix_output(
        calib_path=str(calib_file), merged_g_path=g_path,
        output_path=args.output, source=source,
        n_events=args.events, seed=args.seed, root_exe=args.root,
        calib=calib)
    if rc != 0:
        retry = ("" if args.compose_from is not None
                 else f"; retry the composition alone with "
                      f"--compose-from {g_path}")
        raise SystemExit(
            f"error: composite export failed (exit code {rc}){retry}")
    # G lives on in the composite output, so drop the intermediate unless
    # the user supplied it via --compose-from.
    if args.compose_from is None:
        os.remove(g_path)
    print(f"Composite response written: {final_output_path(args.output)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix_sel = _selected_matrix_mode(args)
    if matrix_sel is not None:
        mode, calib_path = matrix_sel
        if args.events is None:
            raise SystemExit(
                f"error: --{mode} is batch-only; pass --events N")
        if args.events <= 0:
            raise SystemExit("error: --events must be a positive integer")
        verbose = args.verbose if args.verbose is not None else 0
        batch_matrix_mode(args, mode, calib_path, verbose)
        return

    source_key = _selected_source(args)
    if args.events is None:
        if args.output is None:
            os.makedirs(_OUT_DIR, exist_ok=True)
            args.output = os.path.join(_OUT_DIR, "sim_vis_output.root")
        verbose = args.verbose if args.verbose is not None else 1
        interactive_mode(source_key, args.seed, verbose,
                         output_stem(args.output))
    else:
        if args.events <= 0:
            raise SystemExit("error: --events must be a positive integer")
        if args.output is None:
            os.makedirs(_OUT_DIR, exist_ok=True)
            args.output = os.path.join(_OUT_DIR, "sim_output.root")
        verbose = args.verbose if args.verbose is not None else 0
        batch_mode(args, source_key, verbose)


if __name__ == "__main__":
    main()
