"""Shared output-file naming conventions for the KC761 simulation."""

from __future__ import annotations

import os

NTUPLE_NAME = "kc761_data"
SPECTRUM_HIST_NAME = "kc761_spectrum"


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
