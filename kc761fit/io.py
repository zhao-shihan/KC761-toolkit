"""Reading of experimental and simulated spectra from ROOT files (uproot).

All spectra are returned as :class:`Spectrum` objects holding the per-bin
counts, per-bin errors and the bin edges of the histogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import uproot


@dataclass
class Spectrum:
    """A 1-D histogram: counts, errors (may be None) and bin edges."""

    counts: np.ndarray
    errors: np.ndarray | None
    edges: np.ndarray  # bin edges, length = len(counts) + 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def n_bins(self) -> int:
        return len(self.counts)


def _load_spectrum(path: str, hist_name: str, with_errors: bool) -> Spectrum:
    """Read the TH1D ``hist_name`` from a ROOT file as a Spectrum.

    With ``with_errors=True`` the per-bin errors are read from the histogram;
    otherwise they are dropped (``errors=None``).
    """
    with uproot.open(path) as f:
        h = f[hist_name]
    counts = np.asarray(h.values(), dtype=float)
    errors = np.asarray(h.errors(), dtype=float) if with_errors else None
    edges = np.asarray(h.axis().edges(), dtype=float)
    return Spectrum(counts, errors, edges)


def load_data_spectrum(path: str, hist_name: str = "kc761_spectrum") -> Spectrum:
    """Read an experimental spectrum (counts + errors) from a ROOT file.

    Typical input: the background-subtracted file written by subbkg.cxx
    (TH1D "kc761_spectrum" with per-bin errors stored).
    """
    return _load_spectrum(path, hist_name, with_errors=True)


def load_sim_spectrum(path: str, hist_name: str = "kc761_spectrum") -> Spectrum:
    """Read a simulated (intrinsic, not resolution-smeared) spectrum.

    The simulation histogram is assumed to be in energy (keV), e.g. the
    deposited-energy spectrum written by the Geant4 kc761sim application
    (TH1D "kc761_spectrum", 1 keV bins).  Its per-bin errors are not needed
    for the fit and are dropped.
    """
    return _load_spectrum(path, hist_name, with_errors=False)
