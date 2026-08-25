"""User actions: primary generation, run/event/stepping hooks and scoring.

Each energy deposition in the CsI(Tl) crystal is recorded together with its
global time.  At the end of the event the deposits are merged into detector
pulses with a fixed resolution time (10 us): deposits that fall within one
resolution window of the first deposit of a group are summed, then the
iteration continues with the next deposit.  This separates the decays of a
cascade chain (e.g. Th-232 / Ra-226) that are far apart in time into distinct
pulses instead of piling them up into one.

All arithmetic uses ``double``; values are only converted to ``float``
(``edep``, keV) at storage time.
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
    keV,
    s,
    us,
)

G4AnalysisManager = G4RootAnalysisManager

#: ROOT ntuple / spectrum-histogram names and energy range.
NTUPLE_NAME = "kc761_data"
SPECTRUM_HIST_NAME = "kc761_spectrum"
#: number of bins and upper edge of the deposited-energy histogram (keV);
#: 4096 bins of 1 keV each.
_EDEP_HISTOGRAM_BINS = 4096
_EDEP_HISTOGRAM_MAX_KEV = 4096.0
#: detector pulse-resolution time: deposits closer together than this are
#: merged into a single pulse (10 us).
RESOLUTION_TIME = 10 * us


class PrimaryGeneratorAction(G4VUserPrimaryGeneratorAction):
    """One source nucleus (ion at rest) per event.

    The vertex is produced by the general particle source (G4GeneralParticleSource),
    which has been configured with the ``/gps/pos/confine`` command to sample
    uniformly inside the source volume (see :func:`kc761sim.physics.configure_gps`);
    the radioactive-decay process then emits the gammas.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.gps = G4GeneralParticleSource()

    def GeneratePrimaries(self, event):
        self.gps.GeneratePrimaryVertex(event)


class RunAction(G4UserRunAction):

    def __init__(self, output_stem: str, source_name: str, verbose: int = 0):
        super().__init__()
        self.output_stem = output_stem
        am = G4AnalysisManager.Instance()
        am.SetVerboseLevel(verbose)
        am.CreateNtuple(
            NTUPLE_NAME,
            f"{NTUPLE_NAME} - {source_name} - per-pulse energy deposition",
        )
        am.CreateNtupleIColumn("event_id")
        # total energy of the pulse, keV (float)
        am.CreateNtupleFColumn("edep")
        # global time of the pulse (first deposit of the group), seconds
        am.CreateNtupleDColumn("time")
        am.FinishNtuple()
        # Note: xmin/xmax of G4AnalysisManager::CreateH1 are in Geant4 internal
        # units (MeV here); unitName only selects the axis unit.  Passing
        # xmax = 4096*keV therefore yields a 0-4096 keV axis with 1 keV/bin.
        am.CreateH1(
            SPECTRUM_HIST_NAME,
            "Energy deposited in CsI crystal per pulse",
            _EDEP_HISTOGRAM_BINS,
            0.0,
            _EDEP_HISTOGRAM_MAX_KEV * keV,
            "keV",
        )

    def BeginOfRunAction(self, run):
        am = G4AnalysisManager.Instance()
        am.OpenFile(self.output_stem)

    def EndOfRunAction(self, run):
        am = G4AnalysisManager.Instance()
        am.Write()
        am.CloseFile()


class EventAction(G4UserEventAction):

    def __init__(self, event_offset: int = 0):
        super().__init__()
        self.event_offset = event_offset
        #: list of (global_time, edep) deposits in the crystal, internal units
        self.deposits: list[tuple[float, float]] = []

    def BeginOfEventAction(self, event):
        self.deposits = []

    def AddDeposit(self, global_time: float, edep: float):
        self.deposits.append((global_time, edep))

    def _merge_pulses(self):
        """Merge the deposits into detector pulses with the resolution time.

        Starting from the first deposit, all deposits whose global time falls
        within ``RESOLUTION_TIME`` of it are summed into one pulse; the
        iteration then continues with the next unmerged deposit until the
        (time-sorted) sequence is exhausted.  Returns (pulse_time, edep).
        """
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

    def EndOfEventAction(self, event):
        # skip events without any energy deposition in the crystal
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
    """Adds the energy deposited in the CsI(Tl) crystal to the current event.

    The crystal logical volume is resolved lazily on the first step (the
    geometry is only constructed during run initialization)."""

    def __init__(self, detector, event_action: EventAction):
        super().__init__()
        self.detector = detector
        self.event_action = event_action
        self._crystal_lv = None

    def UserSteppingAction(self, step):
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
        source,
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

    def BuildForMaster(self):
        self.SetUserAction(
            RunAction(self.output_stem, self.source.name, self.verbose)
        )

    def Build(self):
        self.SetUserAction(PrimaryGeneratorAction(self.source))
        self.SetUserAction(
            RunAction(self.output_stem, self.source.name, self.verbose)
        )
        event_action = EventAction(self.event_offset)
        self.SetUserAction(event_action)
        self.SetUserAction(SteppingAction(self.detector, event_action))
