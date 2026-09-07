"""User actions: primary generation, run/event/stepping hooks and scoring.

Two clearly separated scoring paths live here:

* the radioactive-source path (``RunAction``/``EventAction``) declares the
  ntuple and the 4096-bin spectrum histogram and merges crystal deposits
  into 10 us pulses;
* the matrix-mode path (``MatrixRunAction``/``MatrixEventAction``) declares
  only the primary-to-deposition histogram G (TH2D, x = energy deposition, y =
  primary energy, variable-width axes taken from the calibration file)
  and the zero-deposition counter (TH1D), accumulates the total per-event
  crystal deposit without pulse merging, and writes no ntuple.
"""

from __future__ import annotations

from geant4_pybind import (
    G4GeneralParticleSource,
    G4RootAnalysisManager,
    G4UserEventAction,
    G4UserRunAction,
    G4UserSteppingAction,
    G4VUserActionInitialization,
    G4VUserPrimaryGeneratorAction,
    G4doubleVector,
    keV,
    s,
    us,
)
from .config import SourceSpec
from .generator import GammaEventState, make_gamma_generator
from .paths import (
    MATRIX_G_HIST_NAME,
    MATRIX_ZERO_HIST_NAME,
    NTUPLE_COLUMNS,
    NTUPLE_NAME,
    SPECTRUM_HIST_NAME,
    ntuple_title,
)
from .sources import MatrixSource

G4AnalysisManager = G4RootAnalysisManager

# Geant4 root-manager column method, chosen by the numpy dtype *name* in
# ``paths.NTUPLE_COLUMNS`` (int32/float32/float64).  Note ``dtype.kind`` is
# not usable here: both float32 and float64 report kind ``'f'``.
_NTUPLE_COLUMN_CREATORS = {
    "int32": "CreateNtupleIColumn",
    "float32": "CreateNtupleFColumn",
    "float64": "CreateNtupleDColumn",
}

_EDEP_HISTOGRAM_BINS = 4096
_EDEP_HISTOGRAM_MAX_KEV = 4096.0
RESOLUTION_TIME = 10 * us


class PrimaryGeneratorAction(G4VUserPrimaryGeneratorAction):
    def __init__(self, source: SourceSpec):
        super().__init__()
        self.source = source
        self.gps = G4GeneralParticleSource()

    def GeneratePrimaries(self, event) -> None:
        self.gps.GeneratePrimaryVertex(event)


class _AnalysisRunAction(G4UserRunAction):
    """Shared open/write/close lifecycle; subclasses declare the schema.

    The histogram declarations happen in the constructor (before the run
    starts), the output file is opened at begin-of-run and written at
    end-of-run -- identical for both scoring paths.
    """

    def __init__(self, output_stem: str, verbose: int = 0):
        super().__init__()
        self.output_stem = output_stem
        am = G4AnalysisManager.Instance()
        am.SetVerboseLevel(verbose)
        self._declare(am)

    def _declare(self, am) -> None:
        """Declare ntuples/histograms; subclass hook."""
        raise NotImplementedError

    def BeginOfRunAction(self, run) -> None:
        am = G4AnalysisManager.Instance()
        am.OpenFile(self.output_stem)

    def EndOfRunAction(self, run) -> None:
        am = G4AnalysisManager.Instance()
        am.Write()
        am.CloseFile()


class RunAction(_AnalysisRunAction):
    """Radioactive path: declares the ntuple and the spectrum schema."""

    def __init__(self, output_stem: str, source_name: str, verbose: int = 0):
        self.source_name = source_name
        super().__init__(output_stem, verbose)

    def _declare(self, am) -> None:
        am.CreateNtuple(NTUPLE_NAME, ntuple_title(self.source_name))
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


class MatrixRunAction(_AnalysisRunAction):
    """Matrix path: declares only the G histogram and the zero-dep counter.

    Both axes of G use the variable-width edges of the input calibration
    file (x = energy deposition, y = primary energy).  No ntuple and no
    spectrum histogram exist on this path.
    """

    def __init__(self, output_stem: str, primary_edges,
                 verbose: int = 0):
        self.edges = primary_edges
        super().__init__(output_stem, verbose)

    def _declare(self, am) -> None:
        # The axes carry the calibration file's keV edge values verbatim
        # (no unit conversion, which would perturb the last bits through
        # the MeV<->keV round trip): the merged file's axes then match the
        # calibration deposition-energy edges bit-for-bit, and the event
        # action fills with the keV values of the same convention.  The
        # G4 physics itself is unaffected (the generator sets the particle
        # energy in MeV).
        edges = G4doubleVector([float(e) for e in self.edges])
        am.CreateH2(
            MATRIX_G_HIST_NAME,
            "Primary-to-deposition matrix (counts)",
            edges,
            edges,
        )
        am.CreateH1(
            MATRIX_ZERO_HIST_NAME,
            "Zero-deposition events per primary-energy bin",
            edges,
        )


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


class MatrixEventAction(G4UserEventAction):
    """Matrix path: total per-event crystal deposit, no pulse merging.

    Each event emits exactly one gamma, so a single pulse concept is
    meaningless here: the total deposit (sum of all stepping deposits in
    the crystal) is the scored quantity.  Events with a strictly positive
    deposit fill the G histogram at ``(e_gamma, total)``; events with no
    deposit at all (geometric misses, absorption in the housing, ...) fill
    the zero-deposition counter -- exactly zero is a physical boundary,
    not a threshold: any positive deposit, however small, lands inside the
    deposition axis, whose lower edge is negative.
    """

    def __init__(self, state: GammaEventState):
        super().__init__()
        self.state = state
        self.total = 0.0

    def BeginOfEventAction(self, event) -> None:
        self.total = 0.0

    def AddDeposit(self, global_time: float, edep: float) -> None:
        # Same signature as EventAction so the shared SteppingAction needs
        # no knowledge of the scoring path; the time is unused here.
        self.total += edep

    def EndOfEventAction(self, event) -> None:
        am = G4AnalysisManager.Instance()
        # Fills in keV; the internal deposit/energy are in MeV.
        if self.total > 0.0:
            am.FillH2(0, self.total / keV, self.state.e_gamma / keV)
        else:
            am.FillH1(0, self.state.e_gamma / keV)


class SteppingAction(G4UserSteppingAction):
    """Collects crystal deposits and hands them to the event action.

    Shared by both scoring paths: the radioactive ``EventAction`` merges
    the timed deposits into pulses, the matrix ``MatrixEventAction`` sums
    them into the per-event total.
    """

    def __init__(self, detector, event_action):
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
        self.SetUserAction(PrimaryGeneratorAction(self.source))
        self.SetUserAction(
            RunAction(self.output_stem, self.source.name, self.verbose)
        )
        event_action = EventAction(self.event_offset)
        self.SetUserAction(event_action)
        self.SetUserAction(SteppingAction(self.detector, event_action))


class MatrixActionInitialization(G4VUserActionInitialization):
    """Action set for the matrix modes: generator + G scoring, no ntuple."""

    def __init__(
        self,
        source: MatrixSource,
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

    def _build_actions(self) -> None:
        state = GammaEventState()
        self.SetUserAction(
            make_gamma_generator(self.source, state, self.event_offset)
        )
        self.SetUserAction(
            MatrixRunAction(self.output_stem, self.source.axis.edges,
                            self.verbose)
        )
        event_action = MatrixEventAction(state)
        self.SetUserAction(event_action)
        self.SetUserAction(SteppingAction(self.detector, event_action))

    def BuildForMaster(self) -> None:
        self.SetUserAction(
            MatrixRunAction(self.output_stem, self.source.axis.edges,
                            self.verbose)
        )

    def Build(self) -> None:
        self._build_actions()
