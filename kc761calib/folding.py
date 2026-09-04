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
error of the finite support.  That renormalization is a fit-internal choice:
the exported full-range matrix (:mod:`kc761calib.export`) keeps the columns
unnormalized, so the probability beyond the detector channel range stays
truncated -- physically lost -- instead of being redistributed onto the
edge bins.  The resolution ``sigma`` saturates at the
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

Besides the smeared per-channel counts, each projection carries their Monte
Carlo statistical variance.  The simulation histogram has per-source-bin
variance ``v`` (its ``sumw2`` buffer, or the Poisson estimate when the file
stores none); the exact rebin onto the true-energy bins is the linear map
``W`` (each source bin is redistributed over the target bins it overlaps
with weights equal to the overlap fractions), and the smeared model is
``m = R W n``.  With independent source bins,
``Var(m) = diag(R W diag(v) W^T R^T) = (R W)^2 v``, i.e. the exact diagonal
of the propagated covariance -- including the small correlation that a
source bin straddling a true-energy bin boundary induces between adjacent
true-energy bins, since ``(R W)^2`` mixes them through the full matrix
square.  The rebinned counts and the banded rebinned covariance ``B = W
diag(v) W^T`` (banded because each source bin overlaps at most a few
consecutive target bins) are accumulated by a fused numba kernel over the
source-major rebin triples, and the smeared variances ``diag(R B R^T)`` are
evaluated by a row-parallel numba kernel over the response matrix's CSR
triples -- the sparse matrices ``W`` and ``R W`` are never materialized.
The per-bin Monte Carlo error enters the chi-square denominator in
quadrature with the data's statistical and systematic errors
(:mod:`kc761calib.fitmodel`).
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

    channel_lo: int  # first extended channel (inclusive)
    channel_hi: int  # last extended channel (inclusive)
    fit_channel_lo: int  # union lower fit channel
    fit_channel_hi: int  # union upper fit channel
    channel_edges: np.ndarray  # n_ext + 1, uniform width 1 (detected axis)
    energy_edges: np.ndarray  # n_ext + 1, true-energy binning (input axis)
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
def _assemble_rebin_weights(src_edges, tgt_edges):
    """Assemble the exact-rebin weight triple of the linear map ``W``.

    ``W[j, s]`` is the overlap fraction of source bin ``s`` with target bin
    ``j`` (both axis conventions: ``edges`` is one longer than the bin
    count).  Source bins fully outside the target range contribute nothing;
    density is zero beyond the source histogram, matching the exact
    cumulative-interpolation rebin.  Returns ``(rows, weights, offsets,
    max_band)``: ``rows`` the target index of each entry and ``weights`` its
    overlap fraction, laid out source-major (all entries of one source bin
    are consecutive, so the source index is ``offsets[s] .. offsets[s+1]``),
    ``offsets`` (length ``n_src + 1``) giving each source bin's entry range,
    and ``max_band`` the largest number of target bins a single source bin
    overlaps minus one (the bandwidth of the rebinned counts' covariance).
    All inputs are float64 1-D arrays.
    """
    n_src = src_edges.shape[0] - 1
    n_tgt = tgt_edges.shape[0] - 1

    # Pass 1: count the overlaps to size the triples exactly.
    nnz = 0
    max_band = 0
    for s in range(n_src):
        e_lo = src_edges[s]
        e_hi = src_edges[s + 1]
        if e_hi <= tgt_edges[0] or e_lo >= tgt_edges[n_tgt] or e_hi <= e_lo:
            continue
        j = np.searchsorted(tgt_edges, e_lo, side="right") - 1
        if j < 0:
            j = 0
        n_entries = 0
        while j < n_tgt and tgt_edges[j] < e_hi:
            n_entries += 1
            j += 1
        nnz += n_entries
        if n_entries - 1 > max_band:
            max_band = n_entries - 1

    rows = np.empty(nnz, dtype=np.int64)
    weights = np.empty(nnz, dtype=np.float64)
    offsets = np.empty(n_src + 1, dtype=np.int64)

    # Pass 2: fill the triples.
    k = 0
    for s in range(n_src):
        offsets[s] = k
        e_lo = src_edges[s]
        e_hi = src_edges[s + 1]
        if e_hi <= tgt_edges[0] or e_lo >= tgt_edges[n_tgt] or e_hi <= e_lo:
            continue
        width = e_hi - e_lo
        j = np.searchsorted(tgt_edges, e_lo, side="right") - 1
        if j < 0:
            j = 0
        while j < n_tgt and tgt_edges[j] < e_hi:
            ov_lo = e_lo if e_lo > tgt_edges[j] else tgt_edges[j]
            ov_hi = e_hi if e_hi < tgt_edges[j + 1] else tgt_edges[j + 1]
            if ov_hi > ov_lo:
                rows[k] = j
                weights[k] = (ov_hi - ov_lo) / width
                k += 1
            j += 1
    offsets[n_src] = k
    return rows, weights, offsets, max_band


@numba.njit(cache=True)
def _rebin_accumulate(rows, weights, offsets, counts, variances, n_target,
                      max_band):
    """Rebinned counts and the rebinned covariance's bands: ``W n`` and ``B``.

    ``B = W diag(v) W^T`` is the exact covariance of the rebinned counts
    over independent source bins; it is banded (a source bin overlaps at
    most ``max_band + 1`` consecutive target bins, and the triples of one
    source are laid out consecutively), so it is stored as
    ``bands[d, j] = B[j, j + d]`` for ``d = 0 .. max_band``.  ``bands[0]``
    is the per-bin variance ``W^2 v``.  Serial: the triples carry only a few
    entries per source, and the parallel per-thread-buffer form costs more
    in buffer zeroing than it saves.
    """
    n_src = counts.shape[0]
    rebinned = np.zeros(n_target)
    bands = np.zeros((max_band + 1, n_target))
    for s in range(n_src):
        k0 = offsets[s]
        n_entries = offsets[s + 1] - k0
        ns = counts[s]
        vs = variances[s]
        for a in range(n_entries):
            j = rows[k0 + a]
            w = weights[k0 + a]
            rebinned[j] += w * ns
            bands[0, j] += w * w * vs
            for b in range(a + 1, n_entries):
                bands[rows[k0 + b] - j, j] += w * weights[k0 + b] * vs
    return rebinned, bands


@numba.njit(parallel=True, cache=True)
def _smeared_variances_csr(indptr, indices, data, bands_all, band_dims):
    """Exact ``Var(R W n)`` for several sims from the response's CSR triples.

    ``Var = diag(R B R^T)`` with ``B`` the banded rebinned covariance of
    :func:`_rebin_accumulate`, i.e. row ``i`` is
    ``sum_{a,b} R[i,j_a] R[i,j_b] B[j_a, j_b]`` over the row's nonzeros.
    The nonzeros of a CSR row are sorted by column index, so each pair
    within ``max_band`` columns of each other contributes via the
    corresponding band.  ``bands_all[s]`` is sim ``s``'s band array
    (``(max_band_s + 1, n_rows)``, zero-padded to a common ``max_band``
    across sims) and ``band_dims[s]`` its ``max_band_s``; each sim
    contributes one contiguous block of ``n_rows`` output entries.  All
    sims share one response matrix, so a single parallel region serves
    them all: parallel over the output rows, every row writes only its own
    entry (no accumulation buffers or reduction).
    """
    n_rows = indptr.shape[0] - 1
    n_sims = bands_all.shape[0]
    out = np.empty(n_sims * n_rows)
    for g in numba.prange(n_sims * n_rows):
        s = g // n_rows
        i = g - s * n_rows
        max_band = band_dims[s]
        acc = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        for a in range(start, end):
            ja = indices[a]
            va = data[a]
            acc += va * va * bands_all[s, 0, ja]
            b = a + 1
            while b < end and indices[b] - ja <= max_band:
                acc += 2.0 * va * data[b] * bands_all[s, indices[b] - ja, ja]
                b += 1
        out[g] = acc
    return out


@dataclass
class SimProjection:
    """One simulation projected through the response onto channel bins.

    ``counts`` are the rebinned, resolution-smeared sim counts per channel
    bin of the extended binning (``m = R W n``); ``variances`` are their
    exact Monte Carlo statistical variances per channel bin
    (``Var(m) = diag(R B R^T)`` with ``B = W diag(v) W^T`` the rebinned
    counts' covariance and ``v`` the source-bin variances), assuming
    independent source bins.  ``rebinned`` and ``rebinned_variances`` carry
    the pre-folding rebinned counts and their variances on the true-energy
    bins (``W n`` and ``W^2 v``), so consumers needing the raw-sim spectrum
    do not recompute the rebin.
    """

    counts: np.ndarray
    variances: np.ndarray
    rebinned: np.ndarray
    rebinned_variances: np.ndarray


class Response:
    """Extended binning + energy-to-channel response matrix shared by all datasets.

    Built once per chi-square evaluation from the shared calibration and
    resolution parameters; each dataset rebins its simulation onto the
    true-energy binning, folds it through the response matrix into channel
    space, and slices its own channel range.  The matrix is stored in CSR
    layout (the row-major form of the column-major assembly): rows drive
    both the sparse @ dense folding and the row-parallel Monte Carlo
    variance kernel.  Projections carry both the smeared counts and their
    Monte Carlo statistical variances (:class:`SimProjection`).
    """

    def __init__(self, binning: ExtendedBinning, matrix: sparse.spmatrix):
        self.binning = binning
        # Normalize to CSR (the row-major form of the column-major
        # assembly): rows drive both the sparse @ dense folding and the
        # row-parallel Monte Carlo variance kernel.
        self.matrix = matrix.tocsr()

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

    def _rebin_structure(self, sim):
        """Source-major rebin triples and bandwidth ``(rows, weights, offsets,
        max_band)`` for ``sim``."""
        return _assemble_rebin_weights(sim.edges, self.binning.energy_edges)

    @staticmethod
    def _source_variances(sim) -> np.ndarray:
        """Per-source-bin variances: stored ``sumw2`` or the Poisson estimate."""
        variances = sim.variances
        if variances is None:
            return np.maximum(sim.counts, 0.0)
        return variances

    @staticmethod
    def _variance_stack(bands_list: list[np.ndarray]):
        """Pad per-sim band arrays to a common ``max_band`` and stack them."""
        max_band = max(b.shape[0] - 1 for b in bands_list)
        n_rows = bands_list[0].shape[1]
        stacked = np.zeros((len(bands_list), max_band + 1, n_rows))
        for s, bands in enumerate(bands_list):
            stacked[s, :bands.shape[0], :] = bands
        return stacked, np.array([b.shape[0] - 1 for b in bands_list],
                                 dtype=np.int64)

    def project(self, sim) -> SimProjection:
        """Rebinned, smeared sim counts per channel bin with their MC variance."""
        return self.project_many([sim])[0]

    def project_many(self, sims) -> list[SimProjection]:
        """Project several sims; single-sim ``project`` delegates here.

        Rebins each simulation onto the true-energy binning, stacks the
        vectors, and folds them through the shared response matrix in one
        sparse @ dense multiply (better reuse of the matrix structure than N
        separate matvecs).  The per-sim Monte Carlo variances are propagated
        exactly as ``diag(R B R^T)`` with ``B`` the banded rebinned
        covariance, by one row-parallel numba kernel over the response
        matrix's CSR triples shared by all sims.  Sims sharing a binning
        (equal edge arrays) reuse one rebin structure, since it depends only
        on the edges.
        """
        if not sims:
            return []
        n_target = self.binning.energy_centers.size
        structure = None
        prev_edges = None
        rebinned_list = []
        bands_list = []
        for sim in sims:
            if structure is None or not np.array_equal(sim.edges, prev_edges):
                structure = self._rebin_structure(sim)
                prev_edges = sim.edges
            rows, weights, offsets, max_band = structure
            rebinned, bands = _rebin_accumulate(
                rows, weights, offsets, sim.counts,
                self._source_variances(sim), n_target, max_band)
            rebinned_list.append(rebinned)
            bands_list.append(bands)
        smeared = self.matrix @ np.column_stack(rebinned_list)
        bands_all, band_dims = self._variance_stack(bands_list)
        variances_all = _smeared_variances_csr(
            self.matrix.indptr, self.matrix.indices, self.matrix.data,
            bands_all, band_dims)
        return [
            SimProjection(
                counts=smeared[:, j],
                variances=variances_all[j * n_target:(j + 1) * n_target],
                rebinned=rebinned_list[j],
                rebinned_variances=bands_list[j][0])
            for j in range(len(sims))
        ]
