"""Run-level constants for the Geant4 simulation package.

These values are the frozen run defaults (docs/plan.md D-30..D-38,
D-120..D-128). Geometry values live in :mod:`kc761.sim.geometry`, the source
registry in :mod:`kc761.sim.sources` and the material data in
:mod:`kc761.sim.materials`; this module only holds run orchestration constants.
No Geant4 import happens here so the module stays importable without Geant4.
"""

from __future__ import annotations

from typing import Final

from kc761.schema.products import OBJ_PRIMARY_TO_DEPOSITION

DEFAULT_SEED: Final = 908136382
"""Legacy base seed (``cli/sim.py`` used the same value; D-123).

Worker ``i`` no longer derives its stream from ``seed + i + 1``; matrix mode
derives a stream per primary column and source mode per event block (F-SIM-7),
so a fixed seed and worker partition reproduce bit-for-bit.
"""

SOURCE_MODE_NAME: Final = "source_decay"
"""``mode_name`` recorded by the source-mode ``mc_spectrum`` product (D-120)."""

SOURCE_MODE_EVENT_BLOCK: Final = 1024
"""Events per source-mode RNG block (D-123). Blocks are the reseed granularity
and are never split across workers, so each block's stream is fixed."""

# Raw worker-object names (D-125). Matrix names reuse the frozen schema object
# names; the zero-deposition counter and the source-mode raw spectrum are
# sim-local. ``SPECTRUM_HIST_NAME`` is the *worker scratch* name, deliberately
# distinct from ``OBJ_MC_SPECTRUM`` (``kc761_mc_spectrum``), which is only used
# by the merged final product (to remove the name ambiguity).
MATRIX_HIST_NAME: Final = OBJ_PRIMARY_TO_DEPOSITION
ZERO_DEPOSITION_HIST_NAME: Final = "zero_deposition_counts"
SPECTRUM_HIST_NAME: Final = "source_spectrum_counts"

# Memory budget (D-124). ``G4_WORKER_BASELINE_BYTES`` is an **uncalibrated
# assumption** for the resident Geant4 process footprint (not measured on
# this machine); it only biases the automatic worker-count estimate, never a
# physics or product result, and an explicit ``threads`` overrides it.
MEMORY_SAFETY_FRACTION: Final = 0.8
G4_WORKER_BASELINE_BYTES: Final = 192 * 1024 * 1024
BYTES_PER_FLOAT64: Final = 8
