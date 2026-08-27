"""Read experimental and simulated spectra from ROOT files (uproot)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import uproot


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
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def n_bins(self) -> int:
        return len(self.counts)


def _load_spectrum(path: str, hist_name: str, with_errors: bool) -> Spectrum:
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
    return _load_spectrum(path, hist_name, with_errors=True)


def load_sim_spectrum(path: str, hist_name: str = "kc761_spectrum") -> Spectrum:
    return _load_spectrum(path, hist_name, with_errors=False)
