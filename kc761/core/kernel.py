"""Response kernel: exact Gaussian bin integrals and soft support truncation.

Formula IDs (docs/derivations.md): F-KERN-1 .. F-KERN-4.

* F-KERN-1: exact bin probability
  ``P(i | j) = Phi((e_{i+1} - c_j) / s_j) - Phi((e_i - c_j) / s_j)``,
  where ``Phi`` is the standard normal CDF computed through ``erf``. No
  midpoint approximation.
* F-KERN-2: smoothstep support taper (D-76) over the bin centre
  ``|x| < n_sigma * sigma`` followed by exact column renormalization so that
  every non-empty C column sums to 1; empty columns are exactly zero (D-79).
* F-KERN-3: sparse triple assembly of the response matrix.
* F-KERN-4: kernel parameter derivatives ``d p / d c``, ``d p / d sigma`` and
  the edge derivatives ``d p / d e_lo``, ``d p / d e_hi`` used by F-RESP-4.

All values and derivatives come from ``kc761/core/_gen/kernel_expr.py``
(docs/derivations.md section 2); nothing here transcribes a derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kc761.core import _gen
from kc761.core._checks import as_float_array
from kc761.errors import CertificateError, ValidationError

N_SIGMA = 5.0
"""Default support half-width in resolution sigmas."""

COLUMN_SUM_TOL = 1e-10
"""F-KERN-2 certificate tolerance for column sums."""


@dataclass(frozen=True)
class SparseTriples:
    """COO-style triples ``(rows, cols, values)`` for a sparse matrix."""

    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class KernelGradients:
    """Kernel derivatives on the ``response_triples`` pattern (F-KERN-4).

    ``values`` are the **unnormalized** tapered bin probabilities ``n``;
    ``triples.values`` are the normalized ``p = n / D``. The derivative
    arrays are partial derivatives of ``n`` (the bin-centre contribution is
    already folded into ``dn_de_lo``/``dn_de_hi``). Renormalization couples
    every entry of a column through ``D``; the owning caller (F-RESP-4)
    assembles the quotient rule with ``column_denominator`` and per-column
    sums of the weighted local derivatives, which this split makes possible
    without materializing dense edge-by-source matrices.
    """

    triples: SparseTriples
    values: NDArray[np.float64]
    dn_de_lo: NDArray[np.float64]
    dn_de_hi: NDArray[np.float64]
    dn_dc: NDArray[np.float64]
    dn_dsigma: NDArray[np.float64]
    column_denominator: NDArray[np.float64]


def _check_n_sigma(n_sigma: float) -> float:
    if not np.isfinite(n_sigma) or n_sigma < 1.0:
        raise ValidationError(f"n_sigma must be finite and >= 1, got {n_sigma!r}")
    return float(n_sigma)


def gaussian_bin_probabilities(
    edges_kev: NDArray[np.float64],
    centers_kev: NDArray[np.float64],
    sigma_kev: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact Gaussian bin probabilities for every (bin, source) pair (F-KERN-1).

    Returns an array of shape ``(n_bins, n_sources)``; rows are spectrum bins
    and columns are source energies (``c_j``) with widths ``sigma_kev``.
    """
    edges = as_float_array("edges_kev", edges_kev, ndim=1)
    centers = as_float_array("centers_kev", centers_kev, ndim=1)
    sigma = as_float_array("sigma_kev", sigma_kev, ndim=1)
    if edges.size < 2:
        raise ValidationError("edges_kev needs at least 2 entries")
    if centers.shape != sigma.shape:
        raise ValidationError(
            f"centers_kev and sigma_kev must match, got {centers.shape} and {sigma.shape}"
        )
    if np.any(sigma <= 0.0):
        raise ValidationError("sigma_kev must be strictly positive")
    e_lo = edges[:-1, None]
    e_hi = edges[1:, None]
    return _gen.kernel_expr.gaussian_bin_probability(e_lo, e_hi, centers[None, :], sigma[None, :])


def support_taper(
    offsets_kev: NDArray[np.float64],
    sigma_kev: NDArray[np.float64],
    *,
    n_sigma: float = N_SIGMA,
) -> NDArray[np.float64]:
    """Smooth taper weight for kernel offsets (F-KERN-2).

    ``w = 3 s**2 - 2 s**3`` with ``s = clip((n_sigma * sigma - |offset|) / sigma, 0, 1)``
    (D-76): exactly 1 on the plateau ``|offset| <= (n_sigma - 1) sigma`` and
    exactly 0 beyond ``|offset| >= n_sigma sigma``. ``offsets_kev`` and
    ``sigma_kev`` broadcast against each other.
    """
    offsets = as_float_array("offsets_kev", offsets_kev)
    sigma = as_float_array("sigma_kev", sigma_kev)
    if np.any(sigma <= 0.0):
        raise ValidationError("sigma_kev must be strictly positive")
    return _gen.kernel_expr.taper(offsets, sigma, _check_n_sigma(n_sigma))


def taper_grad(
    offsets_kev: NDArray[np.float64],
    sigma_kev: NDArray[np.float64],
    *,
    n_sigma: float = N_SIGMA,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """F-KERN-4 taper value and derivatives ``d/d(offset)``, ``d/d(sigma)``."""
    offsets = as_float_array("offsets_kev", offsets_kev)
    sigma = as_float_array("sigma_kev", sigma_kev)
    if np.any(sigma <= 0.0):
        raise ValidationError("sigma_kev must be strictly positive")
    return _gen.kernel_expr.taper_grad(offsets, sigma, _check_n_sigma(n_sigma))


def response_triples(
    spectrum_edges_kev: NDArray[np.float64],
    source_centers_kev: NDArray[np.float64],
    source_sigma_kev: NDArray[np.float64],
    *,
    n_sigma: float = N_SIGMA,
) -> SparseTriples:
    """Sparse response triples with tapered, renormalized columns (F-KERN-3).

    Only bin centres strictly inside ``n_sigma * sigma`` are evaluated; the
    taper is exactly zero elsewhere, so the pruned sum is identical to the
    full one and the objective stays continuous in the parameters (D-79).
    """
    pieces = _kernel_pattern(
        spectrum_edges_kev, source_centers_kev, source_sigma_kev, n_sigma, gradients=False
    )
    return pieces[0]


def response_triples_grad(
    spectrum_edges_kev: NDArray[np.float64],
    source_centers_kev: NDArray[np.float64],
    source_sigma_kev: NDArray[np.float64],
    *,
    n_sigma: float = N_SIGMA,
) -> KernelGradients:
    """Unnormalized kernel derivatives on the response pattern (F-KERN-4)."""
    triples, values, d_lo, d_hi, d_c, d_sigma, denominator = _kernel_pattern(
        spectrum_edges_kev, source_centers_kev, source_sigma_kev, n_sigma, gradients=True
    )
    return KernelGradients(
        triples=triples,
        values=values,
        dn_de_lo=d_lo,
        dn_de_hi=d_hi,
        dn_dc=d_c,
        dn_dsigma=d_sigma,
        column_denominator=denominator,
    )


def _kernel_pattern(
    spectrum_edges_kev: NDArray[np.float64],
    source_centers_kev: NDArray[np.float64],
    source_sigma_kev: NDArray[np.float64],
    n_sigma: float,
    *,
    gradients: bool,
) -> tuple:
    """Evaluate the tapered kernel on its nonzero pattern.

    Returns ``(triples,)`` without gradients, or the 7-tuple ``(triples, n,
    dp_de_lo, dp_de_hi, dp_dc, dp_dsigma, column_denominator)``; the derivative
    arrays are aligned with ``triples``, while ``n`` and ``column_denominator``
    carry the F-KERN-3 renormalization bookkeeping.
    """
    edges = as_float_array("spectrum_edges_kev", spectrum_edges_kev, ndim=1)
    centers = as_float_array("source_centers_kev", source_centers_kev, ndim=1)
    sigma = as_float_array("source_sigma_kev", source_sigma_kev, ndim=1)
    if edges.size < 2:
        raise ValidationError("spectrum_edges_kev needs at least 2 entries")
    if centers.shape != sigma.shape:
        raise ValidationError(
            f"source_centers_kev and source_sigma_kev must match, "
            f"got {centers.shape} and {sigma.shape}"
        )
    if centers.size == 0:
        raise ValidationError("at least one source energy is required")
    if np.any(sigma <= 0.0):
        raise ValidationError("source_sigma_kev must be strictly positive")
    n_sigma = _check_n_sigma(n_sigma)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    rows_all: list[NDArray[np.int64]] = []
    cols_all: list[NDArray[np.int64]] = []
    values_all: list[NDArray[np.float64]] = []
    if gradients:
        d_c_all: list[NDArray[np.float64]] = []
        d_sigma_all: list[NDArray[np.float64]] = []
        d_lo_all: list[NDArray[np.float64]] = []
        d_hi_all: list[NDArray[np.float64]] = []
        n_all: list[NDArray[np.float64]] = []
        column_denominator = np.zeros(centers.size, dtype=np.float64)

    for column, (center, width) in enumerate(zip(centers, sigma, strict=True)):
        radius = n_sigma * float(width)
        start = int(np.searchsorted(bin_centers, float(center) - radius, side="right"))
        stop = int(np.searchsorted(bin_centers, float(center) + radius, side="left"))
        if start >= stop:
            continue
        rows = np.arange(start, stop, dtype=np.int64)
        e_lo = edges[rows]
        e_hi = edges[rows + 1]
        row_centers = bin_centers[rows]

        if gradients:
            denominator_values, d_lo, d_hi, d_c, d_sigma, d_center = (
                _gen.kernel_expr.tapered_bin_grad(
                    e_lo, e_hi, float(center), float(width), row_centers, n_sigma
                )
            )
            # The bin centre is (e_lo + e_hi) / 2, so a change of either edge
            # also moves the taper argument by half the edge displacement.
            d_lo = d_lo + 0.5 * d_center
            d_hi = d_hi + 0.5 * d_center
            denominator = float(denominator_values.sum())
            if denominator <= 0.0:
                continue
            probability = denominator_values / denominator
            keep = probability > 0.0
            if not np.any(keep):
                continue
            rows_all.append(rows[keep])
            cols_all.append(np.full(int(keep.sum()), column, dtype=np.int64))
            values_all.append(probability[keep])
            n_all.append(denominator_values[keep])
            d_c_all.append(d_c[keep])
            d_sigma_all.append(d_sigma[keep])
            d_lo_all.append(d_lo[keep])
            d_hi_all.append(d_hi[keep])
            column_denominator[column] = denominator
        else:
            denominator_values = _gen.kernel_expr.tapered_bin(
                e_lo, e_hi, float(center), float(width), row_centers, n_sigma
            )
            denominator = float(denominator_values.sum())
            if denominator <= 0.0:
                continue
            probability = denominator_values / denominator
            keep = probability > 0.0
            if not np.any(keep):
                continue
            rows_all.append(rows[keep])
            cols_all.append(np.full(int(keep.sum()), column, dtype=np.int64))
            values_all.append(probability[keep])

    triples = SparseTriples(
        rows=_concatenate(rows_all, np.int64),
        cols=_concatenate(cols_all, np.int64),
        values=_concatenate(values_all, np.float64),
    )
    if not gradients:
        return (triples,)
    return (
        triples,
        _concatenate(n_all, np.float64),
        _concatenate(d_lo_all, np.float64),
        _concatenate(d_hi_all, np.float64),
        _concatenate(d_c_all, np.float64),
        _concatenate(d_sigma_all, np.float64),
        column_denominator,
    )


def _concatenate(arrays: list[NDArray], dtype: type) -> NDArray:
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def verify_column_sums(
    triples: SparseTriples,
    n_channels: int,
    *,
    strict: bool,
    expected_atol: float = COLUMN_SUM_TOL,
) -> NDArray[np.float64]:
    """F-KERN-2 certificate: every non-empty column sums to one.

    Returns the per-column sums (a diagnostic in both modes); strict mode
    raises :class:`kc761.errors.CertificateError` when a sum is neither 0 nor
    within ``expected_atol`` of 1.
    """
    if not isinstance(n_channels, int) or n_channels < 1:
        raise ValidationError(f"n_channels must be a positive int, got {n_channels!r}")
    sums = np.zeros(n_channels, dtype=np.float64)
    if triples.values.size:
        if np.any(triples.rows < 0) or np.any(triples.rows >= n_channels):
            raise ValidationError("kernel triple rows outside the channel range")
        if np.any(triples.cols < 0):
            raise ValidationError("kernel triple columns must be non-negative")
        if not np.isfinite(triples.values).all() or np.any(triples.values < 0.0):
            raise ValidationError("kernel triple values must be finite and non-negative")
        np.add.at(sums, triples.cols, triples.values)
    if strict:
        bad = (sums > expected_atol) & (np.abs(sums - 1.0) > expected_atol)
        if np.any(bad):
            worst = int(np.argmax(np.abs(sums - 1.0)))
            raise CertificateError(
                "F-KERN-2",
                f"column {worst} sums to {sums[worst]:.12g} instead of 1",
            )
    return sums
