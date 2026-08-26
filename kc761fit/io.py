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
    """A 1-D histogram: counts, errors (may be None) and bin edges.

    ``counts`` may be negative for background-subtracted spectra; ``errors``
    must be finite and non-negative (NaN/negative errors corrupt the chi^2
    weights).  The structure is validated in ``__post_init__``.
    """

    counts: np.ndarray
    errors: np.ndarray | None
    edges: np.ndarray  # bin edges, length = len(counts) + 1

    def __post_init__(self):
        counts = np.asarray(self.counts, dtype=float)
        edges = np.asarray(self.edges, dtype=float)
        if counts.ndim != 1:
            raise ValueError(
                f"histogram counts must be 1-D, got shape {counts.shape} "
                "(a TH1 is required, not a TH2/TH3)")
        if edges.ndim != 1 or len(edges) != len(counts) + 1:
            raise ValueError(
                f"bin edges must be 1-D with len(counts)+1 = {len(counts) + 1} "
                f"entries, got shape {edges.shape}")
        if not np.isfinite(counts).all():
            raise ValueError("histogram counts contain NaN/inf")
        if self.errors is not None:
            errors = np.asarray(self.errors, dtype=float)
            if errors.shape != counts.shape:
                raise ValueError(
                    f"per-bin errors shape {errors.shape} does not match "
                    f"counts shape {counts.shape}")
            if not np.isfinite(errors).all():
                raise ValueError(
                    "per-bin errors contain NaN/inf (the ROOT histogram "
                    "likely has no stored sumw2 buffer and uproot fell back "
                    "to sqrt(counts), which is NaN where the "
                    "background-subtracted counts are negative)")
            if (errors < 0).any():
                raise ValueError("per-bin errors contain negative values")
            self.errors = errors
        self.counts = counts
        self.edges = edges

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def n_bins(self) -> int:
        return len(self.counts)


def _load_spectrum(path: str, hist_name: str, with_errors: bool) -> Spectrum:
    """Read the TH1D ``hist_name`` from a ROOT file as a Spectrum.

    With ``with_errors=True`` the per-bin errors are read from the histogram;
    otherwise they are dropped (``errors=None``).  Missing files / histogram
    keys and structurally invalid histograms raise errors that name the file
    and histogram.
    """
    try:
        with uproot.open(path) as f:
            h = f[hist_name]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ROOT file not found: {path}") from exc
    except KeyError as exc:
        raise KeyError(f"histogram '{hist_name}' not found in {path}") from exc
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
