"""Run-level constants for the Geant4 simulation workstream (W5).

These values are the frozen pre-rewrite run defaults (docs/plan.md D-30..D-38,
D-120..D-128). Geometry values live in :mod:`kc761.sim.geometry`, the source
registry in :mod:`kc761.sim.sources` and the material data in
:mod:`kc761.sim.materials`; this module only holds run orchestration constants.
No Geant4 import happens here so the module stays importable without Geant4.
"""

from __future__ import annotations

from typing import Final

from kc761.schema.products import (
    OBJ_PRIMARY_TO_DEPOSITION,
    OBJ_SPECTRUM,
)

DEFAULT_SEED: Final = 908136382
"""Legacy base seed (``cli/sim.py`` used the same value; D-123).

Worker ``i`` no longer derives its stream from ``seed + i + 1``; matrix mode
derives a stream per primary column and source mode per event block (F-SIM-7),
so the result does not depend on the worker partition.
"""

SOURCE_MODE_NAME: Final = "source_decay"
"""``mode_name`` recorded by the source-mode ``mc_spectrum`` product (D-120)."""

SOURCE_MODE_EVENT_BLOCK: Final = 1024
"""Events per source-mode RNG block (D-123). Blocks are the reseed granularity
and are never split across workers, so each block's stream is fixed."""

# Raw worker-object names. The matrix names reuse the frozen schema object
# names; the zero-deposition counter and the source spectrum are sim-local.
MATRIX_HIST_NAME: Final = OBJ_PRIMARY_TO_DEPOSITION
ZERO_DEPOSITION_HIST_NAME: Final = "zero_deposition_counts"
SPECTRUM_HIST_NAME: Final = OBJ_SPECTRUM

# Memory budget (D-124). ``G4_WORKER_BASELINE_BYTES`` is an assumed Geant4
# process footprint; the histogram term is computed from the run geometry.
MEMORY_SAFETY_FRACTION: Final = 0.8
G4_WORKER_BASELINE_BYTES: Final = 192 * 1024 * 1024
BYTES_PER_FLOAT64: Final = 8
