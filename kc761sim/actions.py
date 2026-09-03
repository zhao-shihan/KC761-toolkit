"""User actions: primary generation, run/event/stepping hooks and scoring."""

from __future__ import annotations

from geant4_pybind import (
    G4RootAnalysisManager,
    G4UserEventAction,
    G4UserRunAction,
    G4UserSteppingAction,
    G4VUserActionInitialization,
    keV,
    s,
    us,
)
from .config import SourceSpec
from .paths import (
    NTUPLE_COLUMNS,
    NTUPLE_NAME,
    SPECTRUM_HIST_NAME,
    ntuple_title,
)
from .source import PrimarySource

G4AnalysisManager = G4RootAnalysisManager

#: Geant4 root-manager column method, chosen by the numpy dtype *name* in
#: ``paths.NTUPLE_COLUMNS`` (int32/float32/float64).  Note ``dtype.kind`` is
#: not usable here: both float32 and float64 report kind ``'f'``.
_NTUPLE_COLUMN_CREATORS = {
    "int32": "CreateNtupleIColumn",
    "float32": "CreateNtupleFColumn",
    "float64": "CreateNtupleDColumn",
}

_EDEP_HISTOGRAM_BINS = 4096
_EDEP_HISTOGRAM_MAX_KEV = 4096.0
RESOLUTION_TIME = 10 * us


class RunAction(G4UserRunAction):
    """Opens the output file and declares the ntuple/spectrum schema."""

    def __init__(self, output_stem: str, source_name: str, verbose: int = 0):
        super().__init__()
        self.output_stem = output_stem
        am = G4AnalysisManager.Instance()
        am.SetVerboseLevel(verbose)
        am.CreateNtuple(NTUPLE_NAME, ntuple_title(source_name))
        # Column set is driven by the canonical NTUPLE_COLUMNS spec so the
        # worker ntuple and the merged tree can never drift apart.
        for name, dtype in NTUPLE_COLUMNS.items():
            getattr(am, _NTUPLE_COLUMN_CREATORS[dtype.name])(name)
        am.FinishNtuple()
        am.CreateH1(
            SPECTRUM_HIST_NAME,
            "Energy deposited in CsI crystal per pulse",
            _EDEP_HISTOGRAM_BINS,
            0.0,
            _EDEP_HISTOGRAM_MAX_KEV * keV,
            "keV",
        )

    def BeginOfRunAction(self, run) -> None:
        am = G4AnalysisManager.Instance()
        am.OpenFile(self.output_stem)

    def EndOfRunAction(self, run) -> None:
        am = G4AnalysisManager.Instance()
        am.Write()
        am.CloseFile()


class EventAction(G4UserEventAction):
    """Accumulates per-step crystal deposits and merges them into pulses."""

    def __init__(self, event_offset: int = 0):
        super().__init__()
        self.event_offset = event_offset
        self.deposits: list[tuple[float, float]] = []

    def BeginOfEventAction(self, event) -> None:
        self.deposits = []

    def AddDeposit(self, global_time: float, edep: float) -> None:
        self.deposits.append((global_time, edep))

    def _merge_pulses(self) -> list[tuple[float, float]]:
        """Merge deposits within one scintillator resolution time."""
        deposits = sorted(self.deposits, key=lambda d: d[0])
        pulses: list[tuple[float, float]] = []
        i, n = 0, len(deposits)
        while i < n:
            t0 = deposits[i][0]
            t_cut = t0 + RESOLUTION_TIME
            edep = 0.0
            while i < n and deposits[i][0] <= t_cut:
                edep += deposits[i][1]
                i += 1
            pulses.append((t0, edep))
        return pulses

    def EndOfEventAction(self, event) -> None:
        if not self.deposits:
            return
        am = G4AnalysisManager.Instance()
        event_id = self.event_offset + event.GetEventID()
        for t0, edep in self._merge_pulses():
            am.FillNtupleIColumn(0, event_id)
            am.FillNtupleFColumn(1, float(edep / keV))
            am.FillNtupleDColumn(2, float(t0 / s))
            am.AddNtupleRow()
            am.FillH1(0, edep)


class SteppingAction(G4UserSteppingAction):
    def __init__(self, detector, event_action: EventAction):
        super().__init__()
        self.detector = detector
        self.event_action = event_action
        self._crystal_lv = None

    def UserSteppingAction(self, step) -> None:
        if self._crystal_lv is None:
            self._crystal_lv = self.detector.crystal_lv
        volume = step.GetPreStepPoint().GetTouchable().GetVolume()
        if volume is None:
            return
        if volume.GetLogicalVolume() != self._crystal_lv:
            return
        edep_step = step.GetTotalEnergyDeposit()
        if edep_step > 0.0:
            self.event_action.AddDeposit(
                step.GetPreStepPoint().GetGlobalTime(), edep_step
            )


class ActionInitialization(G4VUserActionInitialization):
    def __init__(
        self,
        source: SourceSpec,
        detector,
        output_stem: str,
        event_offset: int = 0,
        verbose: int = 0,
    ):
        super().__init__()
        self.source = source
        self.detector = detector
        self.output_stem = output_stem
        self.event_offset = event_offset
        self.verbose = verbose

    def BuildForMaster(self) -> None:
        self.SetUserAction(
            RunAction(self.output_stem, self.source.name, self.verbose)
        )

    def Build(self) -> None:
        self.SetUserAction(PrimarySource(self.source, self.detector))
        self.SetUserAction(
            RunAction(self.output_stem, self.source.name, self.verbose)
        )
        event_action = EventAction(self.event_offset)
        self.SetUserAction(event_action)
        self.SetUserAction(SteppingAction(self.detector, event_action))
