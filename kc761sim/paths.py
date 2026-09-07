"""Output file layout and ROOT output conventions shared by all sim tools."""

from __future__ import annotations

import os

import numpy as np

from kc761util.spectrum import SPECTRUM_HIST_NAME  # noqa: F401 (re-export)

NTUPLE_NAME = "kc761_data"

#: Worker/merged histogram name of the matrix modes' gamma transport
#: matrix G (x = energy deposition, y = primary energy).
MATRIX_G_HIST_NAME = "primary_deposition_matrix"

#: Worker/merged histogram name of the matrix modes' per-primary-column
#: zero-deposition counts.
MATRIX_ZERO_HIST_NAME = "zero_deposition_counts"

#: Ntuple column order and dtypes; written by actions.RunAction and
#: re-created by runner.merge_root_files when combining worker files.
NTUPLE_COLUMNS: dict[str, np.dtype] = {
    "event_id": np.dtype(np.int32),
    "edep": np.dtype(np.float32),
    "time": np.dtype(np.float64),
}


def ntuple_title(source_name: str | None = None) -> str:
    """Ntuple title for one source, or for a merged multi-worker file."""
    label = f"{NTUPLE_NAME} - {source_name}" if source_name else NTUPLE_NAME
    return f"{label} - per-pulse energy deposition"


def count_label(n: int) -> str:
    """Compact event-count label, e.g. 3000000 -> '3e6'."""
    mantissa, exponent = f"{n:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def output_stem(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".root"):
        return path[:-5]
    return path


def final_output_path(path: str) -> str:
    return output_stem(path) + ".root"


def temp_work_dir(stem: str) -> str:
    dirname, basename = os.path.split(stem)
    return os.path.join(dirname, f".{basename}-kc761sim_wip")
