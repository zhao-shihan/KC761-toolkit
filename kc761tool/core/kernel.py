"""Response kernel: exact Gaussian bin integrals and soft support truncation.

Formula IDs (docs/derivations.md): F-KERN-1 .. F-KERN-4.

* F-KERN-1: exact bin probability
  ``P(i | j) = Phi((e_{i+1} - c_j) / s_j) - Phi((e_i - c_j) / s_j)``,
  where ``Phi`` is the standard normal CDF computed through ``erf``. No
  midpoint approximation.
* F-KERN-2: smoothstep support taper (D-76) over the bin center
  ``|x| < n_sigma * sigma`` followed by exact column renormalization so that
  every non-empty C column sums to 1; empty columns are exactly zero (D-79).
* F-KERN-3: sparse triple assembly of the response matrix.
* F-KERN-4: kernel parameter derivatives ``d p / d c``, ``d p / d sigma`` and
  the edge derivatives ``d p / d e_lo``, ``d p / d e_hi`` used by F-RESP-4.

All values and derivatives come from ``kc761tool/core/_gen/kernel_expr.py``
(docs/derivations.md section 2); nothing here transcribes a derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numba import prange
from numpy.typing import NDArray

from kc761tool.core import _gen
from kc761tool.core._checks import as_float_array
from kc761tool.core._gen.kernel_expr import scalar_tapered_bin, scalar_tapered_bin_grad
from kc761tool.errors import CertificateError, ValidationError

N_SIGMA = 5.0
"""Default support half-width in resolution sigmas."""

COLUMN_SUM_TOL = 1e-10
"""F-KERN-2 certificate tolerance for column sums."""


def support_bounds(
    bin_centers: NDArray[np.float64],
    centers: NDArray[np.float64],
    sigma: NDArray[np.float64],
    n_sigma: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per-column support windows ``[start, stop)`` over ``bin_centers``.

    Shared by the response/Jacobian assembly so the search convention (``right``
    for the lower edge, ``left`` for the upper edge) has a single definition.
    """
    radius = n_sigma * sigma
    starts = np.searchsorted(bin_centers, centers - radius, side="right").astype(np.int64)
    stops = np.searchsorted(bin_centers, centers + radius, side="left").astype(np.int64)
    return starts, stops


@numba.njit(cache=True, parallel=True)
def _pattern_fill_grad(  # noqa: ANN001
    edges, centers, sigma, n_sigma, starts, stops, offsets,
    rows, cols, raw, d_lo, d_hi, d_c, d_sigma,
):
    """Fill the F-KERN-4 flat arrays; each column writes a disjoint slice."""
    for column in prange(centers.size):
        center = centers[column]
        width = sigma[column]
        for row in range(starts[column], stops[column]):
            e_lo = edges[row]
            e_hi = edges[row + 1]
            row_center = 0.5 * (e_lo + e_hi)
            value, dl, dh, dc, ds, dcenter = scalar_tapered_bin_grad(
                e_lo, e_hi, center, width, row_center, n_sigma
            )
            position = offsets[column] + (row - starts[column])
            rows[position] = row
            cols[position] = column
            raw[position] = value
            half = 0.5 * dcenter
            d_lo[position] = dl + half
            d_hi[position] = dh + half
            d_c[position] = dc
            d_sigma[position] = ds


@numba.njit(cache=True, parallel=True)
def _pattern_fill_value(  # noqa: ANN001
    edges, centers, sigma, n_sigma, starts, stops, offsets, rows, cols, raw
):
    """Fill the F-KERN-2 flat value array; each column writes a disjoint slice."""
    for column in prange(centers.size):
        center = centers[column]
        width = sigma[column]
        for row in range(starts[column], stops[column]):
            e_lo = edges[row]
            e_hi = edges[row + 1]
            position = offsets[column] + (row - starts[column])
            rows[position] = row
            cols[position] = column
            raw[position] = scalar_tapered_bin(
                e_lo, e_hi, center, width, 0.5 * (e_lo + e_hi), n_sigma
            )


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
    arrays are partial derivatives of ``n`` (the bin-center contribution is
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

    Only bin centers strictly inside ``n_sigma * sigma`` are evaluated; the
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


def _pattern_flat(  # noqa: ANN001
    edges, centers, sigma, n_sigma, starts, stops, total, gradients
):
    """Parallel scalar fill of the flat pattern arrays (D-174).

    The support windows come from :func:`support_bounds`; the scalar kernels are
    the numba renderings of the same sympy expressions as ``_gen``. Each column
    writes a disjoint slice, so the result is thread-count independent.
    """
    offsets = np.cumsum(stops - starts) - (stops - starts)
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    raw = np.empty(total, dtype=np.float64)
    if not gradients:
        _pattern_fill_value(
            edges, centers, sigma, float(n_sigma), starts, stops, offsets, rows, cols, raw
        )
        return rows, cols, raw, None, None, None, None
    d_lo = np.empty(total, dtype=np.float64)
    d_hi = np.empty(total, dtype=np.float64)
    d_c = np.empty(total, dtype=np.float64)
    d_sigma = np.empty(total, dtype=np.float64)
    _pattern_fill_grad(
        edges, centers, sigma, float(n_sigma), starts, stops, offsets,
        rows, cols, raw, d_lo, d_hi, d_c, d_sigma,
    )
    return rows, cols, raw, d_lo, d_hi, d_c, d_sigma


def _finish_pattern(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    raw: NDArray[np.float64],
    d_lo: NDArray[np.float64] | None,
    d_hi: NDArray[np.float64] | None,
    d_c: NDArray[np.float64] | None,
    d_sigma: NDArray[np.float64] | None,
    n_columns: int,
    gradients: bool,
) -> tuple:
    """Renormalize columns (F-KERN-2) and keep the strictly positive entries."""
    denominator = np.bincount(cols, weights=raw, minlength=n_columns)
    probability = _gen.kernel_expr.normalize(raw, denominator[cols])
    keep = probability > 0.0
    triples = SparseTriples(
        rows=rows[keep].astype(np.int64),
        cols=cols[keep].astype(np.int64),
        values=probability[keep].astype(np.float64),
    )
    if not gradients:
        return (triples,)
    assert d_lo is not None and d_hi is not None and d_c is not None and d_sigma is not None
    return (
        triples,
        raw[keep].astype(np.float64),
        d_lo[keep].astype(np.float64),
        d_hi[keep].astype(np.float64),
        d_c[keep].astype(np.float64),
        d_sigma[keep].astype(np.float64),
        denominator,
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

    The support windows come from a vectorized search; large problems are
    filled by the parallel scalar numba kernels (D-174) and small ones by a
    single vectorized generated call, both rendering the same sympy
    expressions as ``_gen``.
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
    starts, stops = support_bounds(bin_centers, centers, sigma, n_sigma)
    total = int(np.maximum(stops - starts, 0).sum())
    flat = _pattern_flat(edges, centers, sigma, n_sigma, starts, stops, total, gradients)
    return _finish_pattern(*flat, centers.size, gradients)


def verify_column_sums(
    triples: SparseTriples,
    n_channels: int,
    *,
    strict: bool,
    expected_atol: float = COLUMN_SUM_TOL,
) -> NDArray[np.float64]:
    """F-KERN-2 certificate: every non-empty column sums to one.

    Returns the per-column sums (a diagnostic in both modes); strict mode
    raises :class:`kc761tool.errors.CertificateError` when a sum is neither 0 nor
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
