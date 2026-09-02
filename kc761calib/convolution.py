"""Shared extended energy grid and sparse convolution matrix.

The channel axis is the primary uniform binning (bin width 1, edges
``-0.5 .. channel_max + 0.5``, bin centers equal to the channel indices) and
the energy axis is a pure relabeling of it through the calibration, so the
energy bins are non-uniform but carry the same counts.  For every chi-square
evaluation one :class:`Convolution` (an extended grid plus its response
matrix) is built from the shared calibration/resolution parameters and reused
by all datasets.

The extended grid covers the union of the datasets' fit channel ranges (the
*work range*) plus every channel bin whose Gaussian kernel can reach it, so
truncating the matrix to this grid does not affect the work-range bins.

Matrix convention: ``A[i, j]`` is the probability that a count in input
(true-energy) bin ``j`` is detected in output (smeared) bin ``i``:
``A[i, j] = gaussian_density(c_i - c_j; sigma_j) * width_i``, with the
kernel parameter evaluated at the source-bin center ``c_j`` and ``width_i``
the output-bin energy width (midpoint-of-PDF times bin-width quadrature).
``A[i, j]`` is kept nonzero only inside the kernel support
``[c_j - n_sigma sigma_j, c_j + n_sigma sigma_j]`` -- the same condition the
grid extension uses, so the two are self-consistent -- and each column is then
renormalized to sum exactly 1, absorbing the ~1e-4 truncation/quadrature error
of the finite support.

The band spans are located with searchsorted and the ``(indptr, indices,
data)`` triple is assembled by a single fused, parallel numba kernel
(column-major, i.e. CSC layout), which avoids the numpy fancy-indexing and
COO->CSR conversion overheads of a vectorized build; the matrix build is the
dominant cost of an evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from scipy import sparse

from .response import (gaussian_pdf, calib_model, resol_sigma_model)

N_SIGMA = 4.0


@dataclass
class ConvolutionGrid:
    """Uniform extended channel grid and its energy relabeling."""

    channel_lo: int            # first extended channel (inclusive)
    channel_hi: int            # last extended channel (inclusive)
    work_channel_lo: int       # union lower fit channel
    work_channel_hi: int       # union upper fit channel
    channel_edges: np.ndarray  # n_ext + 1, uniform width 1
    energy_edges: np.ndarray   # n_ext + 1, E(channel_edges), non-uniform
    energy_centers: np.ndarray  # n_ext
    energy_widths: np.ndarray  # n_ext

    @property
    def n_bins(self) -> int:
        return self.channel_hi - self.channel_lo + 1

    def channel_slice(self, channel_low: int, channel_high: int) -> slice:
        """Bin slice into the grid arrays for [channel_low, channel_high]."""
        if not (self.channel_lo <= channel_low <= channel_high <= self.channel_hi):
            raise ValueError(
                f"channel range [{channel_low}, {channel_high}] is outside the "
                f"grid [{self.channel_lo}, {self.channel_hi}]")
        return slice(channel_low - self.channel_lo,
                     channel_high - self.channel_lo + 1)

    def channel_edge_slice(self, channel_low: int, channel_high: int) -> slice:
        """Edge slice (one longer than the bin slice) for the channel range."""
        if not (self.channel_lo <= channel_low <= channel_high <= self.channel_hi):
            raise ValueError(
                f"channel range [{channel_low}, {channel_high}] is outside the "
                f"grid [{self.channel_lo}, {self.channel_hi}]")
        return slice(channel_low - self.channel_lo,
                     channel_high - self.channel_lo + 2)


def build_convolution_grid(calib_params: np.ndarray | list[float],
                           resol_params: np.ndarray | list[float],
                           channel_max: float, work_channel_lo: int,
                           work_channel_hi: int, last_channel: int,
                           n_sigma: float = N_SIGMA) -> ConvolutionGrid:
    """Extended channel grid covering the work range plus the kernel support.

    ``work_channel_lo..work_channel_hi`` (inclusive channel indices) is the
    union of the datasets' fit ranges.  The extension scans outward one
    channel bin at a time and includes a bin while its kernel -- evaluated at
    the bin-center energy, as in the matrix -- still reaches the work-range
    energy edges:

    * lower side: ``E(k) + n_sigma sigma >= E(work_lo - 0.5)``
    * upper side: ``E(k) - n_sigma sigma <= E(work_hi + 0.5)``

    The scan is evaluated vectorized over all candidate channels (equivalent
    to stopping at the first non-reaching bin when the conditions are
    monotone, and a physically safe superset otherwise) and is inherently
    clamped to the detector range ``[0, last_channel]``.
    """
    work_lo = int(work_channel_lo)
    work_hi = int(work_channel_hi)
    last = int(last_channel)
    if not (0 <= work_lo <= work_hi <= last):
        raise ValueError(
            f"work channel range [{work_lo}, {work_hi}] must satisfy "
            f"0 <= work_lo <= work_hi <= last_channel ({last})")

    e_lo_work = float(calib_model(calib_params, work_lo - 0.5, channel_max))
    e_hi_work = float(calib_model(calib_params, work_hi + 0.5, channel_max))

    grid_lo = work_lo
    low_channels = np.arange(work_lo, dtype=float)  # 0 .. work_lo - 1
    if low_channels.size:
        e_low = calib_model(calib_params, low_channels, channel_max)
        support_hi = n_sigma * resol_sigma_model(resol_params, e_low)
        reaches = e_low + support_hi >= e_lo_work
        if reaches.any():
            grid_lo = int(low_channels[np.argmax(reaches)])

    grid_hi = work_hi
    high_channels = np.arange(work_hi + 1, last + 1, dtype=float)
    if high_channels.size:
        e_hi = calib_model(calib_params, high_channels, channel_max)
        support_lo = n_sigma * resol_sigma_model(resol_params, e_hi)
        reaches = e_hi - support_lo <= e_hi_work
        if reaches.any():
            grid_hi = int(high_channels[reaches.size - 1
                                        - np.argmax(reaches[::-1])])

    channel_edges = np.arange(grid_lo, grid_hi + 2, dtype=float) - 0.5
    energy_edges = calib_model(calib_params, channel_edges, channel_max)
    if np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(
            "energy calibration is not strictly increasing on the extended "
            f"grid [{grid_lo}, {grid_hi}]; the convolution grid requires a "
            "monotone calibration")
    return ConvolutionGrid(
        channel_lo=grid_lo,
        channel_hi=grid_hi,
        work_channel_lo=work_lo,
        work_channel_hi=work_hi,
        channel_edges=channel_edges,
        energy_edges=energy_edges,
        energy_centers=0.5 * (energy_edges[:-1] + energy_edges[1:]),
        energy_widths=np.diff(energy_edges),
    )


@numba.njit(parallel=True)
def _assemble_matrix(centers, widths, sigma, lo, hi):
    """Fused column-major assembly of the response matrix nonzero triple.

    Column ``j`` (true-energy bin) contributes the output rows ``lo[j] ..
    hi[j]-1``.  Returns ``(indptr, indices, data)`` in CSC layout (``indptr``
    indexes columns, ``indices`` holds the row of each entry).  A single pass
    over the nonzeros computes the row indices and the Gaussian density values,
    reusing the per-column sigma, so there are no numpy fancy-indexing or
    ``np.repeat`` intermediates.
    """
    n = centers.shape[0]
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    for j in range(n):
        indptr[j + 1] = indptr[j] + (hi[j] - lo[j])
    nnz = indptr[n]
    indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    for j in numba.prange(n):
        start = indptr[j]
        c_j = centers[j]
        s_j = sigma[j]
        for k in range(lo[j], hi[j]):
            idx = start + (k - lo[j])
            indices[idx] = k
            data[idx] = gaussian_pdf(centers[k] - c_j, s_j) * widths[k]
    return indptr, indices, data


def build_convolution_matrix(grid: ConvolutionGrid,
                             resol_params: np.ndarray | list[float],
                             n_sigma: float = N_SIGMA) -> sparse.csc_matrix:
    """Sparse response matrix mapping true grid bins to smeared grid bins.

    Row ``i``, column ``j``: ``gaussian_density(c_i - c_j; sigma_j) *
    width_i`` for ``c_i`` inside the kernel support of column ``j``; zero
    otherwise.  Columns are renormalized to sum exactly 1, absorbing the
    truncation/quadrature error of the finite kernel support.

    The nonzero triple is assembled column-major, so the returned matrix is
    CSC; ``A @ v`` is bit-identical to the CSR form.
    """
    centers = grid.energy_centers
    widths = grid.energy_widths
    n = centers.size
    sigma = resol_sigma_model(resol_params, centers)
    support = n_sigma * sigma

    lo = np.searchsorted(centers, centers - support)
    hi = np.searchsorted(centers, centers + support, side="right")
    indptr, indices, data = _assemble_matrix(centers, widths, sigma, lo, hi)
    col_sums = np.add.reduceat(data, indptr[:-1])
    data /= np.repeat(col_sums, np.diff(indptr))
    return sparse.csc_matrix((data, indices, indptr), shape=(n, n))


def rebin_exact(counts: np.ndarray, edges: np.ndarray,
                target_edges: np.ndarray) -> np.ndarray:
    """Exact rebin of a piecewise-constant histogram onto ``target_edges``.

    Each target bin receives the sum of ``overlap_fraction * counts`` over all
    source bins, computed by interpolating the cumulative counts at the target
    edges.  Outside ``[edges[0], edges[-1]]`` the density is zero, so target
    bins beyond the source histogram get zero counts.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(counts)))
    lows = np.interp(target_edges[:-1], edges, cumulative)
    highs = np.interp(target_edges[1:], edges, cumulative)
    return highs - lows


class Convolution:
    """Extended grid + response matrix pair shared by all datasets.

    Built once per chi-square evaluation from the shared calibration and
    resolution parameters; each dataset rebins its simulation onto the grid,
    applies the same matrix, and slices its own channel range.
    """

    def __init__(self, grid: ConvolutionGrid, matrix: sparse.csc_matrix):
        self.grid = grid
        self.matrix = matrix

    @classmethod
    def build(cls, calib_params: np.ndarray | list[float],
              resol_params: np.ndarray | list[float], channel_max: float,
              work_channel_lo: int, work_channel_hi: int, last_channel: int,
              n_sigma: float = N_SIGMA) -> Convolution:
        """Construct the grid and matrix for one evaluation."""
        grid = build_convolution_grid(
            calib_params, resol_params, channel_max, work_channel_lo,
            work_channel_hi, last_channel, n_sigma)
        matrix = build_convolution_matrix(grid, resol_params, n_sigma)
        return cls(grid, matrix)

    def rebinned(self, sim) -> np.ndarray:
        """Exact rebin of the sim histogram onto the grid bins."""
        return rebin_exact(sim.counts, sim.edges, self.grid.energy_edges)

    def apply(self, rebinned_counts: np.ndarray) -> np.ndarray:
        """Apply the response matrix to counts already on the grid bins."""
        return self.matrix @ rebinned_counts

    def smeared(self, sim) -> np.ndarray:
        """Rebinned, resolution-smeared sim counts on the grid bins."""
        return self.apply(self.rebinned(sim))

    def smeared_many(self, sims) -> list[np.ndarray]:
        """Rebinned, resolution-smeared sim counts for several sims at once.

        Rebins each simulation onto the grid, stacks the vectors, and applies
        the shared response matrix in one sparse @ dense multiply (better
        reuse of the matrix structure than N separate matvecs).  Returns one
        grid-bin vector per input sim; each is bit-identical to ``smeared``.
        """
        stacked = np.column_stack([self.rebinned(sim) for sim in sims])
        result = self.matrix @ stacked
        return [result[:, j] for j in range(result.shape[1])]
