"""Physical-layer simulation certificates (F-SIM-1..3, F-SIM-6; D-127).

The production helpers (``binomial_variance``, ``source_spectrum_variance``)
are the single implementation of the stored-variance formulas; the certificate
functions call them and raise :class:`kc761.errors.CertificateError` with the
formula ID. Product-level certificates (F-SIM-2 on the stored `fSumw2`) also
run in :mod:`kc761.schema.io` under strict mode; these functions guard the
physics layer before a product is built.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761.errors import CertificateError, ValidationError

_RTOL = 1e-9


def binomial_variance(
    counts: NDArray[np.float64], totals: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Exact fixed-total variance ``N_j p (1 - p)`` with ``p = counts/N_j``.

    ``counts`` has shape ``(n_deposition, n_primary)`` and ``totals`` shape
    ``(n_primary,)``. Zero-total columns give zero variance.
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    probabilities = np.divide(
        counts, totals[None, :], out=np.zeros_like(counts), where=totals[None, :] > 0.0
    )
    return totals[None, :] * probabilities * (1.0 - probabilities)


def source_spectrum_variance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conditional-binomial spectrum variance ``c (1 - c/P)`` (D-128).

    ``P = sum(counts)`` is the recorded pulse total; an all-zero spectrum gives
    zero variance. This is the plug-in approximation to the multinomial over
    pulses documented in F-SIM-2.
    """
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total == 0.0:
        return np.zeros_like(values)
    return values * (1.0 - values / total)


def verify_event_accounting(
    counts: NDArray[np.float64],
    totals: NDArray[np.float64],
    zero_counts: NDArray[np.float64],
    n_events: int,
) -> None:
    """F-SIM-1: ``sum_bins counts_j + zero_j = N_j`` and ``sum_j N_j = n_events``."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    zero_counts = np.asarray(zero_counts, dtype=np.float64)
    if counts.ndim != 2 or totals.ndim != 1 or zero_counts.shape != totals.shape:
        raise ValidationError(
            "accounting shapes must be counts (n_dep, n_prim), totals and zero "
            f"(n_prim,); got {counts.shape}, {totals.shape}, {zero_counts.shape}"
        )
    if counts.shape[1] != totals.size:
        raise ValidationError(
            f"counts has {counts.shape[1]} primary columns, totals has {totals.size}"
        )
    if np.any(counts < 0.0) or np.any(zero_counts < 0.0) or np.any(totals < 0.0):
        raise CertificateError("F-SIM-1", "counts, zero counts and totals must be non-negative")
    recorded = counts.sum(axis=0) + zero_counts
    if np.any(np.abs(recorded - totals) > _RTOL * np.maximum(1.0, totals)):
        offending = int(np.argmax(np.abs(recorded - totals)))
        raise CertificateError(
            "F-SIM-1",
            f"column {offending}: sum(counts) + zero = {recorded[offending]:.12g} "
            f"!= N_j = {totals[offending]:.12g} (events lost to under/overflow?)",
        )
    if not np.isclose(totals.sum(), float(n_events), rtol=0.0, atol=_RTOL):
        raise CertificateError(
            "F-SIM-1",
            f"sum(N_j) = {totals.sum():.12g} != n_events = {n_events}",
        )


def verify_binomial_variance(
    counts: NDArray[np.float64], totals: NDArray[np.float64]
) -> None:
    """F-SIM-2 production guard: the stored variance must equal the exact form."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    expected = binomial_variance(counts, totals)
    if np.any(expected < 0.0):
        raise CertificateError("F-SIM-2", "exact binomial variance is negative")
    if np.any(counts > totals[None, :] + _RTOL):
        raise CertificateError("F-SIM-2", "a deposition bin exceeds its column total")


def verify_efficiency(
    counts: NDArray[np.float64], totals: NDArray[np.float64]
) -> None:
    """F-SIM-3: the detection efficiency ``column_sum_j / N_j`` lies in [0, 1]."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    detected = counts.sum(axis=0)
    efficiency = np.divide(
        detected, totals, out=np.zeros_like(totals), where=totals > 0.0
    )
    if np.any(efficiency < -_RTOL) or np.any(efficiency > 1.0 + _RTOL):
        raise CertificateError(
            "F-SIM-3", "derived detection efficiency leaves [0, 1]"
        )


def verify_source_spectrum_variance(
    values: NDArray[np.float64], variances: NDArray[np.float64]
) -> None:
    """F-SIM-2 source variant: ``fSumw2 = c (1 - c/P)`` with ``P = sum(counts)``."""
    values = np.asarray(values, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if values.shape != variances.shape:
        raise ValidationError(
            f"spectrum values {values.shape} and variances {variances.shape} differ"
        )
    expected = source_spectrum_variance(values)
    if np.any(np.abs(variances - expected) > _RTOL * np.maximum(1.0, expected)):
        raise CertificateError(
            "F-SIM-2",
            "source-spectrum fSumw2 does not match c (1 - c/P) with P = sum(counts)",
        )


def verify_physical_boundary(
    counts: NDArray[np.float64],
    deposition_edges_kev: NDArray[np.float64],
    primary_edges_kev: NDArray[np.float64],
    *,
    strict: bool,
) -> None:
    """F-SIM-6 (D-127): entries with deposition lower edge > primary upper edge are 0.

    A gamma cannot deposit more energy than it carries, so the lower triangle
    above the physical limit must be exactly empty. The certificate runs in
    strict mode only.
    """
    if not strict:
        return
    counts = np.asarray(counts, dtype=np.float64)
    deposition_edges_kev = np.asarray(deposition_edges_kev, dtype=np.float64)
    primary_edges_kev = np.asarray(primary_edges_kev, dtype=np.float64)
    if counts.shape != (deposition_edges_kev.size - 1, primary_edges_kev.size - 1):
        raise ValidationError(
            "physical-boundary check: counts shape does not match the axis binnings"
        )
    impossible = deposition_edges_kev[:-1, None] > primary_edges_kev[None, 1:]
    if np.any(counts[impossible] != 0.0):
        offending = np.argwhere(impossible & (counts != 0.0))
        first = offending[0]
        raise CertificateError(
            "F-SIM-6",
            "deposition above the primary energy: entry "
            f"(deposition bin {int(first[0])}, primary bin {int(first[1])}) = "
            f"{counts[first[0], first[1]]:.12g}",
        )


__all__ = [
    "binomial_variance",
    "source_spectrum_variance",
    "verify_binomial_variance",
    "verify_efficiency",
    "verify_event_accounting",
    "verify_physical_boundary",
    "verify_source_spectrum_variance",
]
