"""Read binned spectra with their per-bin errors from ROOT files (uproot).

:data:`SPECTRUM_HIST_NAME` is the toolkit-wide histogram-name convention:
the simulation, csv2root and subbkg pipelines all write spectra under this
name, and the analysis stages read them back through
:func:`load_spectrum`.

:func:`load_spectrum` serves both experimental and simulated spectra: the
data spectrum's errors enter the fit's data-side uncertainty (statistical,
plus whatever the analysis that produced the file stored), and the
simulated spectrum's errors are the Monte Carlo statistical errors -- the
histogram's ``sumw2`` buffer (the sum of squared fill weights) when the
file stores one, and the Poisson ``sqrt(counts)`` estimate -- which equals
``sumw2`` for unit-weight fills -- when it does not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import uproot

#: Histogram name under which all toolkit pipelines store binned spectra.
SPECTRUM_HIST_NAME = "kc761_spectrum"


@dataclass
class Spectrum:
    counts: np.ndarray
    errors: np.ndarray | None
    edges: np.ndarray

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
    def n_bins(self) -> int:
        return len(self.counts)

    @property
    def variances(self) -> np.ndarray | None:
        """Per-bin variance (``sumw2``): ``errors**2`` when errors are stored.

        ``None`` means the spectrum carries no variance information; for a
        Monte Carlo histogram the caller then falls back to the Poisson
        estimate ``max(counts, 0)``.
        """
        if self.errors is None:
            return None
        return self.errors ** 2


def _file_label(source) -> str:
    """Human-readable file name for an open ROOT file object."""
    path = getattr(source, "file_path", None)
    return str(path) if path else "<open ROOT file>"


def load_spectrum(source: str | os.PathLike | uproot.ReadOnlyDirectory,
                  hist_name: str = SPECTRUM_HIST_NAME) -> Spectrum:
    """Read a spectrum with its per-bin errors from a ROOT file.

    ``source`` is a ROOT file path (``str``/``os.PathLike``, opened here)
    or an already-open ``uproot`` file/directory object, so callers that
    read several objects from one file can open it once and pass the open
    object around (the arrays are materialized before returning, so the
    file may be closed immediately afterwards).

    The errors are the histogram's stored ``sumw2`` buffer.  When the file
    does not store one, uproot falls back to the Poisson ``sqrt(counts)``
    estimate: for a simulated histogram filled with unit weights that
    equals ``sumw2``, so it is exactly the Monte Carlo statistical error,
    while for background-subtracted data it is NaN where the counts are
    negative and :class:`Spectrum` rejects the file with an explanation.
    The same function serves both roles: the data spectrum's errors enter
    the fit's data-side uncertainty and the simulated spectrum's errors
    feed the Monte Carlo finite-statistics term.
    """
    opened = not isinstance(source, uproot.ReadOnlyDirectory)
    if opened:
        path = source
        label = str(path)
        try:
            file = uproot.open(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"ROOT file not found: {path}") from exc
    else:
        file = source
        label = _file_label(source)
    try:
        h = file[hist_name]
    except KeyError as exc:
        raise KeyError(f"histogram '{hist_name}' not found in {label}") from exc
    try:
        counts = np.asarray(h.values(), dtype=float)
        errors = np.asarray(h.errors(), dtype=float)
        edges = np.asarray(h.axis().edges(), dtype=float)
    finally:
        # The arrays are materialized above, so a file this function opened
        # can be closed immediately; callers keep the object they passed.
        if opened:
            file.close()
    return Spectrum(counts, errors, edges)
