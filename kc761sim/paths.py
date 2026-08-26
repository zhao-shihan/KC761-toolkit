"""Shared output-file naming conventions for the kc761 simulation.

The Geant4 application (:mod:`kc761sim.actions`), the run orchestrator
(:mod:`kc761sim.runner`) and the batch helper scripts (``sim.py`` /
``run-sim.py``) must agree on the ntuple / histogram names and on how output
file paths are derived.  Those constants and helpers live here, free of the
Geant4 bindings, so the plain helper scripts can import them without pulling
in geant4_pybind.
"""

from __future__ import annotations

import os

#: ROOT ntuple / spectrum-histogram names written by the simulation.
NTUPLE_NAME = "kc761_data"
SPECTRUM_HIST_NAME = "kc761_spectrum"


def output_stem(path: str) -> str:
    """Strip a trailing ``.root`` (case-insensitive) for G4Analysis, which
    appends the file-type extension itself."""
    lower = path.lower()
    if lower.endswith(".root"):
        return path[:-5]
    return path


def final_output_path(path: str) -> str:
    """Output path with a ``.root`` suffix guaranteed."""
    return output_stem(path) + ".root"


def temp_work_dir(stem: str) -> str:
    """Hidden scratch directory that holds the per-worker ROOT files.

    The directory is derived from the output file name, e.g.
    ``sim_output.root`` -> ``.sim_output-kc761sim_wip``.
    """
    dirname, basename = os.path.split(stem)
    return os.path.join(dirname, f".{basename}-kc761sim_wip")
