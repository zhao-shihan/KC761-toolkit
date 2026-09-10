"""Deposition-to-channel response and its calibration-parameter derivatives.

The nominal deposition-to-channel matrix is the calibration file's stored
matrix itself -- the same operator the fit used and the export wrote --
and its seven parameter derivatives come from the shared analytic
Jacobian of :mod:`kc761calib.matrixjac`, evaluated on the full
calibration binning and sliced to the working subrange.  Value and
derivatives therefore describe the exact stored operator, including its
5-sigma kernel-support cutoff and column renormalization.

For kc761sim matrix-mode files the primary-to-channel model is
``R(q) = C(q) p_tilde diag(eta)`` with the q-independent conditional
deposition distribution ``p_tilde`` (the column-normalized primary-to-deposition counts) and
the per-column detection efficiency ``eta`` from the simulation file; the
derivatives become ``dR/dq = (dC/dq) p_tilde diag(eta)``.
:func:`energy_geometry_derivatives` is the single source of the
channel-to-energy geometry derivatives (also used by the penalty
gradient in kc761unfold.penalty and the uncertainties in
kc761unfold.uncertainties).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kc761calib.matrixjac import build_matrix_jacobian
from kc761util.simfile import SimFile

from .reader import threshold_matrix
from .types import CalibrationFile


def energy_geometry_derivatives(channel_low: int, channel_high: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Exact derivatives of the center/width geometry wrt (c0..c3).

    Returns ``(d_centers, d_widths)``, each ``(4, n_bins)`` with
    ``[p] = d/dc_p``.  The bin edges are the cubic at the half-integer
    positions (bin ``k`` spanning ``k-0.5`` and ``k+0.5``), so

        d center_k/dc_p = 0.5 ((k-0.5)^p + (k+0.5)^p),
        d width_k/dc_p  = (k+0.5)^p - (k-0.5)^p.

    The penalty gradient consumes these arrays (indexed by the local bin
    number) so the penalty geometry derivatives stay consistent with the
    response's own geometry.
    """
    k = np.arange(channel_low, channel_high + 1, dtype=float)
    lo = k - 0.5
    hi = k + 0.5
    powers = np.arange(4, dtype=float)[:, None]
    d_centers = 0.5 * (lo ** powers + hi ** powers)
    d_widths = hi ** powers - lo ** powers
    return d_centers, d_widths


def deposition_to_channel(calib: CalibrationFile) -> sparse.csr_matrix:
    """The stored deposition-to-channel matrix on the snapshot's subrange.

    The calibration file's own operator (5-sigma cutoff and column
    renormalization already applied at export time), so the unfold model
    uses exactly what the fit and the export used.
    """
    return calib.to_channel


def to_channel_and_derivatives(calib: CalibrationFile,
                               sim: SimFile
                               ) -> tuple[sparse.csr_matrix,
                                          list[sparse.csr_matrix]]:
    """Nominal to-channel matrix and its seven parameter derivatives.

    Returns ``(r, [dr_0 .. dr_6])`` as thresholded CSR matrices on the
    snapshot's subrange, with the composition step
    ``R = C p_tilde diag(eta)`` applied: ``C`` is the calibration file's
    stored matrix, ``p_tilde`` the column-normalized simulation counts
    and ``eta`` the per-column detection efficiency (both q-independent),
    so the derivatives are ``dR/dq = (dC/dq) p_tilde diag(eta)`` with
    ``dC/dq`` the analytic Jacobian of the stored operator.  A missing
    simulation file is an error (the unfold is defined on the composed
    response).
    """
    if sim is None:
        raise ValueError(
            "the unfold response requires the simulation file "
            "(R = C p_tilde diag(eta)); calibration-only mode does not "
            "build a response")
    c = calib.to_channel
    # The stored operator was built on the full calibration binning; the
    # Jacobian is evaluated there and sliced, so its column
    # renormalization matches the stored matrix exactly.
    jac = build_matrix_jacobian(calib.energy_edges_full, calib.calib_coeffs,
                                calib.resol_params)
    sl = slice(calib.channel_low, calib.channel_high + 1)

    g = np.asarray(sim.counts, dtype=float)
    col_sums = g.sum(axis=0)
    # Conditional primary-to-deposition distribution (columns normalized
    # over the deposited events; zero-deposition excluded).  Skipped
    # (zero-total) columns are all-zero and carry efficiency 0, so they
    # stay zero through the composition.
    p_tilde = threshold_matrix(np.divide(
        g, col_sums, out=np.zeros_like(g), where=col_sums > 0.0))
    eta_diag = sparse.diags(np.asarray(sim.efficiency, dtype=float))

    r = (c @ p_tilde @ eta_diag).tocsr()
    derivatives = [
        (sparse.csr_matrix(jac[sl, sl, p]) @ p_tilde @ eta_diag).tocsr()
        for p in range(7)
    ]
    return r, derivatives
