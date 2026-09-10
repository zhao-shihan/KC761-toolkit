"""Geant4 simulation workstream package (W5).

Owns the frozen source registry (D-122), the detector geometry/material data
(D-34), the two matrix sampling modes (D-31) and the batch runner with
memory-budgeted workers and deterministic per-column/per-block randomness
(D-123/D-124).

Geant4 is imported lazily inside functions only, so this package stays
importable without Geant4; the registry, geometry and sampling surfaces are
plain Python.

Entry points for W6 live in :mod:`kc761.sim.runner`
(``run_source``/``run_matrix``/``prepare_interactive``).
"""

from __future__ import annotations

from kc761.sim.config import DEFAULT_SEED
from kc761.sim.sources import MATRIX_MODE_NAMES, SOURCE_KEYS, SOURCES, get_source

__all__ = [
    "DEFAULT_SEED",
    "MATRIX_MODE_NAMES",
    "SOURCES",
    "SOURCE_KEYS",
    "get_source",
]
