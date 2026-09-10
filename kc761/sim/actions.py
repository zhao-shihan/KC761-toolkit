"""User actions for both simulation scoring paths (Geant4, imported lazily).

* Source mode declares only the pulse spectrum (the ntuple is deleted, D-30),
  merges crystal deposits within the 10 us coincidence window (F-SIM-5) and
  reseeds the global engine at each source-mode event block (F-SIM-7).
* Matrix mode declares the primary-to-deposition histogram ``G`` and the
  zero-deposition counter, accumulates the per-event crystal deposit and fills
  ``G`` at ``(deposition, primary)`` in keV (no pulse merging: each event is one
  gamma).

All Geant4 classes are defined inside the builder functions so importing this
module without Geant4 stays possible.
"""

from __future__ import annotations

from kc761.core.binning import (
    SOURCE_MODE_DEPOSITION_BINS,
    SOURCE_MODE_DEPOSITION_MAX_KEV,
)
from kc761.sim.config import (
    MATRIX_HIST_NAME,
    SOURCE_MODE_EVENT_BLOCK,
    SPECTRUM_HIST_NAME,
    ZERO_DEPOSITION_HIST_NAME,
)
from kc761.sim.generator import GammaEventState, block_seed, make_gamma_generator
from kc761.sim.sources import MatrixSource, PrimaryAxis, SourceSpec

COINCIDENCE_RESOLVING_TIME_US = 10.0
"""Pulse-merging window in microseconds (D-33/F-SIM-5)."""


def _analysis_manager():  # noqa: ANN202
    from geant4_pybind import G4RootAnalysisManager

    return G4RootAnalysisManager.Instance()


def build_source_run_action(output_stem: str, verbose: int = 0):  # noqa: ANN201
    """Run action declaring the source-mode pulse spectrum (no ntuple)."""
    from geant4_pybind import G4RootAnalysisManager, G4UserRunAction, keV

    class _RunAction(G4UserRunAction):
        def __init__(self) -> None:
            super().__init__()
            manager = G4RootAnalysisManager.Instance()
            manager.SetVerboseLevel(verbose)
            manager.CreateH1(
                SPECTRUM_HIST_NAME,
                "Energy deposited in CsI crystal per pulse",
                SOURCE_MODE_DEPOSITION_BINS,
                0.0,
                SOURCE_MODE_DEPOSITION_MAX_KEV * keV,
                "keV",
            )

        def BeginOfRunAction(self, run) -> None:  # noqa: ANN001, N802
            G4RootAnalysisManager.Instance().OpenFile(output_stem)

        def EndOfRunAction(self, run) -> None:  # noqa: ANN001, N802
            manager = G4RootAnalysisManager.Instance()
            manager.Write()
            manager.CloseFile()

    return _RunAction()


def build_matrix_run_action(  # noqa: ANN201
    output_stem: str,
    deposition_edges_kev,
    primary_edges_kev,
    verbose: int = 0,
):
    """Run action declaring ``G`` and the zero-deposition counter (D-31/D-121).

    Both histogram axes carry the keV edge values verbatim (no unit
    conversion), so the merged axes match ``C.y`` bit-for-bit; the fill values
    are converted from the Geant4 internal MeV to keV.
    """
    from geant4_pybind import G4doubleVector, G4RootAnalysisManager, G4UserRunAction

    deposition_edges = G4doubleVector([float(edge) for edge in deposition_edges_kev])
    primary_edges = G4doubleVector([float(edge) for edge in primary_edges_kev])

    class _RunAction(G4UserRunAction):
        def __init__(self) -> None:
            super().__init__()
            manager = G4RootAnalysisManager.Instance()
            manager.SetVerboseLevel(verbose)
            manager.CreateH2(
                MATRIX_HIST_NAME,
                "Primary-to-deposition matrix (counts)",
                deposition_edges,
                primary_edges,
            )
            manager.CreateH1(
                ZERO_DEPOSITION_HIST_NAME,
                "Zero-deposition events per primary-energy bin",
                primary_edges,
            )

        def BeginOfRunAction(self, run) -> None:  # noqa: ANN001, N802
            G4RootAnalysisManager.Instance().OpenFile(output_stem)

        def EndOfRunAction(self, run) -> None:  # noqa: ANN001, N802
            manager = G4RootAnalysisManager.Instance()
            manager.Write()
            manager.CloseFile()

    return _RunAction()


def build_source_event_action(event_offset: int, base_seed: int):  # noqa: ANN201
    """Source-mode event action: pulse merging and per-block reseeding."""
    from geant4_pybind import G4Random, G4UserEventAction, us

    window = COINCIDENCE_RESOLVING_TIME_US * us
    set_the_seed = G4Random.setTheSeed

    class _EventAction(G4UserEventAction):
        def __init__(self) -> None:
            super().__init__()
            self._deposits: list[tuple[float, float]] = []

        def BeginOfEventAction(self, event) -> None:  # noqa: ANN001, N802
            self._deposits = []
            global_start = event_offset + int(event.GetEventID())
            if global_start % SOURCE_MODE_EVENT_BLOCK == 0:
                set_the_seed(block_seed(base_seed, global_start // SOURCE_MODE_EVENT_BLOCK))

        def AddDeposit(self, global_time: float, edep: float) -> None:
            self._deposits.append((global_time, edep))

        def _merge_pulses(self) -> list[float]:
            deposits = sorted(self._deposits, key=lambda item: item[0])
            pulses: list[float] = []
            index = 0
            count = len(deposits)
            while index < count:
                cut = deposits[index][0] + window
                energy = 0.0
                while index < count and deposits[index][0] <= cut:
                    energy += deposits[index][1]
                    index += 1
                pulses.append(energy)
            return pulses

        def EndOfEventAction(self, event) -> None:  # noqa: ANN001, N802
            if not self._deposits:
                return
            manager = _analysis_manager()
            for pulse in self._merge_pulses():
                manager.FillH1(0, pulse)

    return _EventAction()


def build_matrix_event_action(state: GammaEventState):  # noqa: ANN201
    """Matrix event action: total per-event deposit, no pulse merging."""
    from geant4_pybind import G4UserEventAction, keV

    class _EventAction(G4UserEventAction):
        def __init__(self) -> None:
            super().__init__()
            self._total = 0.0

        def BeginOfEventAction(self, event) -> None:  # noqa: ANN001, N802
            self._total = 0.0

        def AddDeposit(self, global_time: float, edep: float) -> None:
            self._total += edep

        def EndOfEventAction(self, event) -> None:  # noqa: ANN001, N802
            manager = _analysis_manager()
            if self._total > 0.0:
                manager.FillH2(0, self._total / keV, state.e_gamma / keV)
            else:
                manager.FillH1(0, state.e_gamma / keV)

    return _EventAction()


def build_stepping_action(detector, event_action):  # noqa: ANN001, ANN201
    """Collect crystal deposits and hand them to the event action."""
    from geant4_pybind import G4UserSteppingAction

    class _SteppingAction(G4UserSteppingAction):
        def __init__(self) -> None:
            super().__init__()
            self._crystal_lv = detector.crystal_lv

        def UserSteppingAction(self, step) -> None:  # noqa: ANN001, N802
            volume = step.GetPreStepPoint().GetTouchable().GetVolume()
            if volume is None:
                return
            if volume.GetLogicalVolume() != self._crystal_lv:
                return
            deposit = step.GetTotalEnergyDeposit()
            if deposit > 0.0:
                event_action.AddDeposit(step.GetPreStepPoint().GetGlobalTime(), deposit)

    return _SteppingAction()


def build_source_action_initialization(  # noqa: ANN201
    source: SourceSpec,
    detector,
    output_stem: str,
    event_offset: int,
    base_seed: int,
    verbose: int = 0,
):
    """Action initialization for the radioactive-source mode (GPS primaries).

    Geant4 requires the physics list to be assigned before any
    ``G4UserRunAction``/``G4UserEventAction`` is constructed, so all user action
    objects are created inside :meth:`Build`, which the run manager calls during
    ``Initialize``.
    """
    from geant4_pybind import (
        G4GeneralParticleSource,
        G4VUserActionInitialization,
        G4VUserPrimaryGeneratorAction,
    )

    class _PrimaryGenerator(G4VUserPrimaryGeneratorAction):
        def __init__(self) -> None:
            super().__init__()
            self._gps = G4GeneralParticleSource()

        def GeneratePrimaries(self, event) -> None:  # noqa: ANN001, N802
            self._gps.GeneratePrimaryVertex(event)

    class _ActionInitialization(G4VUserActionInitialization):
        def BuildForMaster(self) -> None:  # noqa: N802
            self.SetUserAction(build_source_run_action(output_stem, verbose))

        def Build(self) -> None:  # noqa: N802
            event_action = build_source_event_action(event_offset, base_seed)
            self.SetUserAction(_PrimaryGenerator())
            self.SetUserAction(build_source_run_action(output_stem, verbose))
            self.SetUserAction(event_action)
            self.SetUserAction(build_stepping_action(detector, event_action))

    return _ActionInitialization()


def build_matrix_action_initialization(  # noqa: ANN201
    column_slice,  # noqa: ANN001
    axis: PrimaryAxis,
    source: MatrixSource,
    detector,
    output_stem: str,
    deposition_edges_kev,
    base_seed: int,
    verbose: int = 0,
):
    """Action initialization for the matrix modes (surface gamma primaries).

    The user actions are constructed inside :meth:`Build` (see
    :func:`build_source_action_initialization` for the Geant4 ordering rule).
    """
    from geant4_pybind import G4VUserActionInitialization

    class _ActionInitialization(G4VUserActionInitialization):
        def BuildForMaster(self) -> None:  # noqa: N802
            self.SetUserAction(
                build_matrix_run_action(
                    output_stem,
                    deposition_edges_kev=deposition_edges_kev,
                    primary_edges_kev=axis.edges_kev,
                    verbose=verbose,
                )
            )

        def Build(self) -> None:  # noqa: N802
            state = GammaEventState()
            event_action = build_matrix_event_action(state)
            self.SetUserAction(
                make_gamma_generator(column_slice, axis, source, base_seed, state)
            )
            self.SetUserAction(
                build_matrix_run_action(
                    output_stem,
                    deposition_edges_kev=deposition_edges_kev,
                    primary_edges_kev=axis.edges_kev,
                    verbose=verbose,
                )
            )
            self.SetUserAction(event_action)
            self.SetUserAction(build_stepping_action(detector, event_action))

    return _ActionInitialization()


__all__ = [
    "COINCIDENCE_RESOLVING_TIME_US",
    "build_matrix_action_initialization",
    "build_matrix_event_action",
    "build_matrix_run_action",
    "build_source_action_initialization",
    "build_source_event_action",
    "build_source_run_action",
    "build_stepping_action",
]
