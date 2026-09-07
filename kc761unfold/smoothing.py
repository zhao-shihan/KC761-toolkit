"""Resolution-floor smoothing operator for the unfolded presentation.

The unfolded spectrum is parameterized as ``mu = S nu`` with ``S`` a
Gaussian smoother whose width is ``resol_frac`` times the analytic
resolution model (the same :func:`kc761calib.response.resol_sigma_model`
kernel that builds the deposition response, so the resolution floor and
the calibration share one source of truth).  The effective response seen
by the solver is ``R S``, and presented features are at least
``resol_frac * sigma`` wide; ``resol_frac = 0`` disables the floor
(unfold ``mu`` directly).

The kernel support is truncated at ``+-5`` sigma (the same cutoff as the
calibration folding) and each column is renormalized to sum exactly 1,
so the floor conserves peak areas.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kc761calib.response import resol_sigma_model

N_SIGMA = 5


def build_resolution_smoother(energy_edges, resol_params,
                              resol_frac: float) -> sparse.csr_matrix:
    """Gaussian smoother ``S[i, j]`` with width ``resol_frac * sigma_j``.

    ``energy_edges`` are the bin edges (keV, length n + 1); ``sigma_j``
    is the analytic resolution at the bin-center energy (the same
    evaluation grid as the response); the row-major kernel uses
    midpoint-quadrature weights ``dE_i`` and is truncated at
    ``+-N_SIGMA`` sigma with columns renormalized to 1.
    """
    if resol_frac <= 0.0:
        raise ValueError("resol_frac must be positive for the smoother")
    edges = np.asarray(energy_edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    de = np.diff(edges)
    n = centers.size
    sigma = np.maximum(resol_frac * resol_sigma_model(
        np.asarray(resol_params, dtype=float), centers), 1e-9)

    lo = np.searchsorted(centers, centers - N_SIGMA * sigma)
    hi = np.searchsorted(centers, centers + N_SIGMA * sigma, side="right")
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, n)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for j in range(n):
        idx = np.arange(lo[j], hi[j])
        if idx.size == 0:
            idx = np.array([j])
        d = centers[idx] - centers[j]
        kern = np.exp(-0.5 * (d / sigma[j]) ** 2) \
            / (np.sqrt(2.0 * np.pi) * sigma[j]) * de[idx]
        kern = kern / kern.sum()  # column renormalized to 1
        rows.append(idx)
        cols.append(np.full(idx.size, j))
        data.append(kern)
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n))
