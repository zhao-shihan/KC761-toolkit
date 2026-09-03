"""Shared extended binning and sparse detector response matrix (energy to channel).

The measured spectra are binned in channels, so the channel axis is the
detected (output) axis of the response matrix: a uniform binning of bin
width 1, edges ``-0.5 .. channel_max + 0.5``, bin centers equal to the
channel indices.  The data counts/errors live on it and never move.  The
simulation instead is a true-energy histogram: the response matrix maps it
to the detected channel space, ``R[i, j]`` being the probability that a
count in the true-energy bin ``j`` is detected in the channel bin ``i``.

The true-energy (input) axis is the calibration image of the channel axis --
energy bin ``j`` is the relabeling of channel bin ``j`` through ``E(ch)`` --
so the true-energy bins are non-uniform, and both axes share one extended
bin set.  For every chi-square evaluation one :class:`Response` (an extended
binning plus its response matrix) is built from the shared
calibration/resolution parameters and reused by all datasets.

The extended binning covers the union of the datasets' fit channel ranges
(the *fit range*) plus every channel bin whose Gaussian kernel can reach it,
so truncating the matrix to this binning does not affect the fit-range bins.

Response-matrix convention: ``R[i, j]`` is the probability that a count in
the true-energy bin ``j`` is detected in channel bin ``i``:
``R[i, j] = gaussian_density(c_i - c_j; sigma_j) * dE_i``, with ``c_i`` and
``c_j`` the energy-bin centers (the midpoints of the bin energy edges,
``c_k ~ E(ch_k)``), ``sigma_j`` the resolution at ``c_j``, and
``dE_i = E(ch_i + 0.5) - E(ch_i - 0.5)`` the energy width of channel bin
``i`` -- the midpoint quadrature weight of the Gaussian integral over that
channel bin.  ``R[i, j]`` is kept nonzero only inside the kernel support
``[c_j - n_sigma sigma_j, c_j + n_sigma sigma_j]`` -- the same condition the
binning extension uses, so the two are self-consistent -- and each column is
then renormalized to sum exactly 1, absorbing the ~1e-6 truncation/quadrature
error of the finite support.  The resolution ``sigma`` saturates at the
low-energy edge of its model domain (``E = 0``; see
:mod:`kc761calib.response`), so the kernel support never collapses to a
sub-bin delta from negative-variance clamping below the fit range.
``n_sigma = 5`` places the cutoff deep in the
Gaussian tail (``exp(-12.5) ~ 3.7e-6``), which makes the truncation negligible
and keeps the residual smooth in the resolution parameters to numerical
precision: finite-difference covariance steps (``~1e-6`` relative) then sample
the same derivative everywhere instead of catching the bin enter/leave jumps
that a tighter cutoff produces.

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

N_SIGMA = 5.0


@dataclass
class ExtendedBinning:
    """Extended detected-channel binning and its true-energy relabeling.

    The channel axis is the detected (output) axis of the response matrix:
    uniform bins of width 1 whose centers are the channel indices.  The
    energy arrays are the true-energy (input) axis: the calibration image of
    the same extended bins, onto which the simulation is rebinned before
    folding and on which the Gaussian kernel is evaluated.  Both axes share
    one extended bin set (energy bin ``j`` is the image of channel bin
    ``j``), so the matrix is square with identically indexed rows and
    columns.
    """

    channel_lo: int            # first extended channel (inclusive)
    channel_hi: int            # last extended channel (inclusive)
    fit_channel_lo: int        # union lower fit channel
    fit_channel_hi: int        # union upper fit channel
    channel_edges: np.ndarray  # n_ext + 1, uniform width 1 (detected axis)
    energy_edges: np.ndarray   # n_ext + 1, E(channel_edges); true-energy binning (input axis)
    energy_centers: np.ndarray  # n_ext, true-energy bin centers
    energy_widths: np.ndarray  # n_ext, energy width dE_i of each channel bin

    def channel_slice(self, channel_low: int, channel_high: int) -> slice:
        """Bin slice into the binning arrays for [channel_low, channel_high]."""
        if not (self.channel_lo <= channel_low <= channel_high <= self.channel_hi):
            raise ValueError(
                f"channel range [{channel_low}, {channel_high}] is outside the "
                f"binning [{self.channel_lo}, {self.channel_hi}]")
        return slice(channel_low - self.channel_lo,
                     channel_high - self.channel_lo + 1)

    def channel_edge_slice(self, channel_low: int, channel_high: int) -> slice:
        """Edge slice (one longer than the bin slice) for the channel range."""
        if not (self.channel_lo <= channel_low <= channel_high <= self.channel_hi):
            raise ValueError(
                f"channel range [{channel_low}, {channel_high}] is outside the "
                f"binning [{self.channel_lo}, {self.channel_hi}]")
        return slice(channel_low - self.channel_lo,
                     channel_high - self.channel_lo + 2)


def _bin_center_energies(calib_params, channels, channel_max):
    """Energy-bin centers of the channel bins (midpoints of the bin edges).

    ``channels`` is a float64 scalar or array of channel indices; the center
    of channel bin ``k`` is ``(E(k - 0.5) + E(k + 0.5)) / 2``, the midpoint
    of its energy edges.  These are the quadrature nodes the response matrix
    evaluates the Gaussian kernel at, so the kernel-support conditions of
    :func:`build_extended_binning` use the same positions and the two stay
    exactly consistent.
    """
    ch = np.asarray(channels, dtype=float)
    lo = calib_model(calib_params, ch - 0.5, channel_max)
    hi = calib_model(calib_params, ch + 0.5, channel_max)
    return 0.5 * (lo + hi)


def build_extended_binning(calib_params: np.ndarray, resol_params: np.ndarray,
                           channel_max: float, fit_channel_lo: int,
                           fit_channel_hi: int, last_channel: int,
                           n_sigma: float = N_SIGMA) -> ExtendedBinning:
    """Extended channel binning covering the fit range plus the kernel support.

    ``fit_channel_lo..fit_channel_hi`` (inclusive channel indices) is the
    union of the datasets' fit ranges.  The extension scans outward one
    channel bin at a time and includes a bin while its kernel -- evaluated at
    the energy-bin center (the midpoint of the channel bin's energy edges),
    exactly as in the matrix -- still reaches the fit-range energy edges:

    * lower side: ``c_k + n_sigma sigma(c_k) >= E(fit_lo - 0.5)``
    * upper side: ``c_k - n_sigma sigma(c_k) <= E(fit_hi + 0.5)``

    The scan is evaluated vectorized over all candidate channels (equivalent
    to stopping at the first non-reaching bin when the conditions are
    monotone, and a physically safe superset otherwise) and is inherently
    clamped to the detector range ``[0, last_channel]``.

    The returned binning carries both axes of the response matrix: the
    uniform channel bins (detected axis) and their calibration image
    ``E(channel_edges)`` (true-energy input axis).
    """
    fit_lo = int(fit_channel_lo)
    fit_hi = int(fit_channel_hi)
    last = int(last_channel)
    if not (0 <= fit_lo <= fit_hi <= last):
        raise ValueError(
            f"fit channel range [{fit_lo}, {fit_hi}] must satisfy "
            f"0 <= fit_lo <= fit_hi <= last_channel ({last})")

    e_lo_fit = float(calib_model(calib_params, fit_lo - 0.5, channel_max))
    e_hi_fit = float(calib_model(calib_params, fit_hi + 0.5, channel_max))

    binning_lo = fit_lo
    low_channels = np.arange(fit_lo, dtype=float)  # 0 .. fit_lo - 1
    if low_channels.size:
        c_low = _bin_center_energies(calib_params, low_channels, channel_max)
        support_hi = n_sigma * resol_sigma_model(resol_params, c_low)
        reaches = c_low + support_hi >= e_lo_fit
        if reaches.any():
            binning_lo = int(low_channels[np.argmax(reaches)])

    binning_hi = fit_hi
    high_channels = np.arange(fit_hi + 1, last + 1, dtype=float)
    if high_channels.size:
        c_hi = _bin_center_energies(calib_params, high_channels, channel_max)
        support_lo = n_sigma * resol_sigma_model(resol_params, c_hi)
        reaches = c_hi - support_lo <= e_hi_fit
        if reaches.any():
            binning_hi = int(high_channels[reaches.size - 1
                                           - np.argmax(reaches[::-1])])

    channel_edges = np.arange(binning_lo, binning_hi + 2, dtype=float) - 0.5
    energy_edges = calib_model(calib_params, channel_edges, channel_max)
    if np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(
            "energy calibration is not strictly increasing on the extended "
            f"binning [{binning_lo}, {binning_hi}]; the response binning "
            "requires a monotone calibration")
    return ExtendedBinning(
        channel_lo=binning_lo,
        channel_hi=binning_hi,
        fit_channel_lo=fit_lo,
        fit_channel_hi=fit_hi,
        channel_edges=channel_edges,
        energy_edges=energy_edges,
        energy_centers=_bin_center_energies(
            calib_params, np.arange(binning_lo, binning_hi + 1, dtype=float),
            channel_max),
        energy_widths=np.diff(energy_edges),
    )


@numba.njit(parallel=True, cache=True)
def _assemble_matrix(centers, widths, sigma, lo, hi):
    """Fused column-major assembly of the response-matrix nonzero triple.

    Column ``j`` (true-energy bin) contributes the output rows ``lo[j] ..
    hi[j]-1`` (detected channel bins).  Returns ``(indptr, indices, data)``
    in CSC layout (``indptr`` indexes columns, ``indices`` holds the row of
    each entry).  A single pass over the nonzeros computes the row indices
    and the Gaussian density values, reusing the per-column sigma, so there
    are no numpy fancy-indexing or ``np.repeat`` intermediates.
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


def build_response_matrix(binning: ExtendedBinning,
                          resol_params: np.ndarray,
                          n_sigma: float = N_SIGMA) -> sparse.csc_matrix:
    """Sparse response matrix mapping true-energy bins to detected channels.

    Row ``i`` (channel bin ``i``), column ``j`` (true-energy bin ``j``):
    ``gaussian_density(c_i - c_j; sigma_j) * dE_i`` for ``c_i`` inside the
    kernel support of column ``j``; zero otherwise.  ``c_i`` and ``c_j`` are
    the energy-bin centers (midpoints of the bin energy edges, ``~ E(ch)``),
    ``sigma_j`` the resolution at ``c_j``, and ``dE_i`` the energy width of
    channel bin ``i`` (the midpoint quadrature weight of the Gaussian
    integral over that channel bin).  Columns are renormalized to sum
    exactly 1, absorbing the truncation/quadrature error of the finite
    kernel support.

    Rows and columns traverse the same extended bin set, so the row kernel
    position ``c_i`` and its quadrature weight ``dE_i`` are the shared
    ``energy_centers[i]`` and ``energy_widths[i]``.  The nonzero triple is
    assembled column-major, so the returned matrix is CSC; ``R @ v`` is
    bit-identical to the CSR form.
    """
    centers = binning.energy_centers
    widths = binning.energy_widths
    n = centers.size
    sigma = resol_sigma_model(resol_params, centers)
    support = n_sigma * sigma

    lo = np.searchsorted(centers, centers - support)
    hi = np.searchsorted(centers, centers + support, side="right")
    indptr, indices, data = _assemble_matrix(centers, widths, sigma, lo, hi)
    col_sums = np.add.reduceat(data, indptr[:-1])
    data /= np.repeat(col_sums, np.diff(indptr))
    return sparse.csc_matrix((data, indices, indptr), shape=(n, n))


@numba.njit(cache=True)
def rebin_exact(counts, edges, target_edges):
    """Exact rebin of a piecewise-constant histogram onto ``target_edges``.

    Each target bin receives the sum of ``overlap_fraction * counts`` over all
    source bins, computed by interpolating the cumulative counts at the target
    edges.  Outside ``[edges[0], edges[-1]]`` the density is zero, so target
    bins beyond the source histogram get zero counts.  All inputs are float64
    1-D arrays.
    """
    cumulative = np.empty(counts.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(counts)
    lows = np.interp(target_edges[:-1], edges, cumulative)
    highs = np.interp(target_edges[1:], edges, cumulative)
    return highs - lows


class Response:
    """Extended binning + energy-to-channel response matrix shared by all datasets.

    Built once per chi-square evaluation from the shared calibration and
    resolution parameters; each dataset rebins its simulation onto the
    true-energy binning, folds it through the response matrix into channel
    space, and slices its own channel range.
    """

    def __init__(self, binning: ExtendedBinning, matrix: sparse.csc_matrix):
        self.binning = binning
        self.matrix = matrix

    @classmethod
    def build(cls, calib_params: np.ndarray,
              resol_params: np.ndarray, channel_max: float,
              fit_channel_lo: int, fit_channel_hi: int, last_channel: int,
              n_sigma: float = N_SIGMA) -> Response:
        """Construct the binning and response matrix for one evaluation."""
        binning = build_extended_binning(
            calib_params, resol_params, channel_max, fit_channel_lo,
            fit_channel_hi, last_channel, n_sigma)
        matrix = build_response_matrix(binning, resol_params, n_sigma)
        return cls(binning, matrix)

    def rebinned(self, sim) -> np.ndarray:
        """Exact rebin of the sim histogram onto the true-energy binning.

        The result lives on the matrix input axis (true-energy bins of the
        extended binning, indexed by channel).
        """
        return rebin_exact(sim.counts, sim.edges, self.binning.energy_edges)

    def fold(self, rebinned_counts: np.ndarray) -> np.ndarray:
        """Fold true-energy counts through the response matrix.

        Applies the energy-to-channel response to counts already on the
        true-energy binning; returns detected-channel counts on the extended
        binning, directly comparable to the per-channel data.
        """
        return self.matrix @ rebinned_counts

    def smeared(self, sim) -> np.ndarray:
        """Rebinned, resolution-smeared sim counts per channel bin."""
        return self.fold(self.rebinned(sim))

    def smeared_many(self, sims) -> list[np.ndarray]:
        """Rebinned, resolution-smeared per-channel sim counts for several sims.

        Rebins each simulation onto the true-energy binning, stacks the
        vectors, and folds them through the shared response matrix in one
        sparse @ dense multiply (better reuse of the matrix structure than N
        separate matvecs).  Returns one per-channel vector per input sim;
        each is bit-identical to ``smeared``.
        """
        stacked = np.column_stack([self.rebinned(sim) for sim in sims])
        result = self.matrix @ stacked
        return [result[:, j] for j in range(result.shape[1])]
