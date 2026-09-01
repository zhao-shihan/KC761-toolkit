"""Shared extended energy grid and sparse convolution matrix.

The channel axis is the primary uniform binning (bin width 1, edges
``-0.5 .. channel_max + 0.5``, bin centers equal to the channel indices) and
the energy axis is a pure relabeling of it through the calibration, so the
energy bins are non-uniform but carry the same counts.  For every chi-square
evaluation one :class:`Convolution` (an extended grid plus its response
matrix) is built from the shared calibration/resolution parameters and reused
by all datasets.

The extended grid covers the union of the datasets' fit channel ranges (the
*work range*) plus every channel bin whose EMG kernel can reach it, so
truncating the matrix to this grid does not affect the work-range bins.

Matrix convention: ``A[i, j]`` is the probability that a count in input
(true-energy) bin ``j`` is detected in output (smeared) bin ``i``:
``A[i, j] = emg_density(c_i - c_j; sigma_j, tau_j) * width_i``, with the
kernel parameters evaluated at the source-bin center ``c_j`` and ``width_i``
the output-bin energy width (midpoint-of-PDF times bin-width quadrature;
columns are not renormalized -- their sums equal 1 up to the ~1e-4
truncation/quadrature error).  ``A[i, j]`` is kept nonzero only inside the
kernel support ``[c_j - nsigma sigma_j, c_j + max(nsigma sigma_j,
ntail tau_j)]`` -- the same condition the grid extension uses, so the two are
self-consistent.

The matrix assembly is vectorized (searchsorted band spans, numpy repeat
bookkeeping) and the density values are computed by a parallel numba kernel;
the matrix build is the dominant cost of an evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .response import (calib_model, emg_density_vec, resol_sigma_model,
                       resol_tau_model)

NSIGMA = 4.0
NTAIL = 10.0


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
                           nsigma: float = NSIGMA,
                           ntail: float = NTAIL) -> ConvolutionGrid:
    """Extended channel grid covering the work range plus the kernel support.

    ``work_channel_lo..work_channel_hi`` (inclusive channel indices) is the
    union of the datasets' fit ranges.  The extension scans outward one
    channel bin at a time and includes a bin while its kernel -- evaluated at
    the bin-center energy, as in the matrix -- still reaches the work-range
    energy edges:

    * lower side: ``E(k) + max(nsigma sigma, ntail tau) >= E(work_lo - 0.5)``
    * upper side: ``E(k) - nsigma sigma <= E(work_hi + 0.5)``

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
        support_hi = np.maximum(
            nsigma * resol_sigma_model(resol_params, e_low),
            ntail * resol_tau_model(resol_params, e_low))
        reaches = e_low + support_hi >= e_lo_work
        if reaches.any():
            grid_lo = int(low_channels[np.argmax(reaches)])

    grid_hi = work_hi
    high_channels = np.arange(work_hi + 1, last + 1, dtype=float)
    if high_channels.size:
        e_hi = calib_model(calib_params, high_channels, channel_max)
        support_lo = nsigma * resol_sigma_model(resol_params, e_hi)
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


def build_convolution_matrix(grid: ConvolutionGrid,
                             resol_params: np.ndarray | list[float],
                             nsigma: float = NSIGMA,
                             ntail: float = NTAIL) -> sparse.csr_matrix:
    """Sparse response matrix mapping true grid bins to smeared grid bins.

    Row ``i``, column ``j``: ``emg_density(c_i - c_j; sigma_j, tau_j) *
    width_i`` for ``c_i`` inside the kernel support of column ``j``; zero
    otherwise.  Columns are not renormalized (their sums approximate 1).
    """
    centers = grid.energy_centers
    widths = grid.energy_widths
    n = centers.size
    sigma = resol_sigma_model(resol_params, centers)
    tau = resol_tau_model(resol_params, centers)
    support_lo = nsigma * sigma
    support_hi = np.maximum(nsigma * sigma, ntail * tau)

    lo = np.searchsorted(centers, centers - support_lo)
    hi = np.searchsorted(centers, centers + support_hi, side="right")
    counts = hi - lo
    nnz = int(counts.sum())

    cols = np.repeat(np.arange(n), counts)
    starts = np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    rows = np.repeat(lo, counts) + (np.arange(nnz) - starts)

    offsets = centers[rows] - centers[cols]
    values = emg_density_vec(offsets, sigma[cols], tau[cols]) * widths[rows]
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


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

    def __init__(self, grid: ConvolutionGrid, matrix: sparse.csr_matrix):
        self.grid = grid
        self.matrix = matrix

    @classmethod
    def build(cls, calib_params: np.ndarray | list[float],
              resol_params: np.ndarray | list[float], channel_max: float,
              work_channel_lo: int, work_channel_hi: int, last_channel: int,
              nsigma: float = NSIGMA, ntail: float = NTAIL) -> "Convolution":
        """Construct the grid and matrix for one evaluation."""
        grid = build_convolution_grid(
            calib_params, resol_params, channel_max, work_channel_lo,
            work_channel_hi, last_channel, nsigma, ntail)
        matrix = build_convolution_matrix(grid, resol_params, nsigma, ntail)
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
