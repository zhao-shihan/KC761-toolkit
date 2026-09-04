"""Binary export of the fitted detector response for the ROOT writer.

The fitted calibration and resolution fully determine the detector
response.  After the fit, this module builds the complete response matrix
over the full detector channel range -- the true-energy bins are the
calibration image of the channel bins -- and serializes it together with
the model formulas (:data:`kc761calib.response.CALIB_FORMULA` /
:data:`kc761calib.response.RESOL_FORMULA`), the fitted parameters, their
7x7 covariance and the per-element errors of the response matrix into a
temporary binary file that :file:`kc761calib/calib2root.cxx` reads and
turns into the final ROOT file.

The response-matrix convention matches :mod:`kc761calib.folding`:
``matrix[i, j]`` is the probability that a count in the true-energy bin
``j`` is detected in the channel bin ``i``.  Unlike the fit's sparse
matrix, the exported matrix is dense over the full channel range, keeps
the full Gaussian (no kernel-support cutoff) and does not renormalize the
columns: the part of the Gaussian outside the detector channel range is
truncated -- physically lost -- so columns near the range edges sum to
less than 1.  The resolution is evaluated at the true-energy bin centers
with :func:`kc761calib.response.resol_sigma_model`, which saturates
``sigma`` at ``b0`` for ``E <= 0``, where the variance polynomial is not
usable.

Parameter and matrix errors: the response depends only on the 7 shared
core parameters ``q = (c0, k1, k2, k3, b0, b1, b2)`` (the per-dataset
scale parameters never enter it).  The per-element 1-sigma error is
propagated linearly from the fit's core covariance ``cov`` -- the same
Gauss-Newton estimate that defines the reported parameter errors, so the
two mean exactly the same thing:

    err[i, j]^2 = G[i, j]^T cov G[i, j],   G_p[i, j] = dR[i, j]/dq_p.

The Jacobian is analytic (the Gaussian kernel, the cubic calibration
polynomial and the Bernstein resolution polynomial are all elementary
functions, and the two clamps of ``sigma(E)`` -- the ``t = 0`` saturation
and the ``MIN_SIGMA`` floor -- contribute exact one-sided derivatives
through :func:`kc761calib.response.resol_sigma_model_grad`), so the error
matrix is deterministic and bit-reproducible.  Undetermined parameters
(all-NaN covariance rows/columns) are treated as fixed for the matrix
errors -- their gradient contributions are dropped, matching the error-band
treatment in :mod:`kc761calib.plot` -- while the stored covariance keeps
their rows/columns as NaN.  If every core parameter is undetermined the
error matrix is all-NaN.

The serialized covariance is in the reported basis ``(c0, c1, c2, c3, b0,
b1, b2)`` -- the parameterization the stored formulas and parameters use
-- obtained from the internal-basis core covariance by
:func:`kc761calib.response.reported_core_cov`, which shares the NaN-aware
transform semantics of :func:`kc761calib.response.reported_calib`.

Temporary-file layout (native byte order; the file is produced and
consumed on the same machine):

========  =====================================================
line 1    magic ``"kc761calib-export-v2\\n"``
line 2    calibration formula text + ``"\\n"``
line 3    resolution formula text + ``"\\n"``
binary    int64 n_calib; float64 calib_coeffs[n_calib] (c0..c3)
binary    int64 n_resol; float64 resol_params[n_resol] (b0..b2)
binary    float64 resol_e_ref (keV)
binary    int64 n_core; float64 param_cov[n_core * n_core] (row-major,
          basis (c0..c3, b0..b2))
binary    int64 n_channels
binary    float64 energy_edges[n_channels + 1]
binary    float64 matrix[n_channels * n_channels] (row-major,
          row = channel bin, column = true-energy bin)
binary    float64 matrix_errors[n_channels * n_channels] (row-major,
          per-element 1-sigma, same layout as matrix)
========  =====================================================
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numba
import numpy as np

from .response import (CALIB_FORMULA, N_CALIB, N_RESOL, RESOL_E_REF,
                       RESOL_FORMULA, calib_model, c0k1k2k3_to_c0c1c2c3,
                       gaussian_pdf, jac_c0k1k2k3, poly_basis,
                       reported_core_cov, resol_sigma_model,
                       resol_sigma_model_grad)
from .types import FitResult

_MAGIC = b"kc761calib-export-v2\n"


@dataclass
class FullResponse:
    """Complete detector response on the full channel range.

    ``matrix[i, j]`` is the probability that a count in the true-energy
    bin ``j`` is detected in the channel bin ``i``.  Channel bins are the
    full detector range ``0 .. n_channels-1`` (uniform width 1, integer
    centers); the true-energy bins are their calibration image (variable
    width).  Columns are not renormalized, so the Gaussian probability
    truncated by the detector range edges is lost, not redistributed.

    ``matrix_errors`` holds the per-element 1-sigma uncertainty propagated
    linearly from the fit's core covariance (same convention and layout as
    ``matrix``); ``param_cov`` is the 7x7 covariance of the stored
    parameters in the reported basis ``(c0, c1, c2, c3, b0, b1, b2)``,
    including the calib-resol cross block.
    """

    n_channels: int
    energy_edges: np.ndarray  # n_channels + 1; true-energy bin edges
    matrix: np.ndarray  # (n_channels, n_channels) float64
    matrix_errors: np.ndarray  # (n_channels, n_channels) float64
    calib_coeffs: np.ndarray  # c0..c3 cubic calibration coefficients
    resol_params: np.ndarray  # b0..b2 resolution parameters (keV)
    param_cov: np.ndarray  # (7, 7) covariance of (c0..c3, b0..b2)


@numba.njit(parallel=True, cache=True)
def _matrix_error_variance(centers, widths, sigma, ds_dE, ds_db,
                           center_grad, width_grad, cov):
    """Squared 1-sigma error of each response-matrix element.

    ``err2[i, j] = G[i, j]^T cov G[i, j]`` with ``G_p[i, j] = dR[i, j]/dq_p``
    and ``q = (c0, k1, k2, k3, b0, b1, b2)`` (internal basis; ``cov`` is the
    NaN-free core covariance -- undetermined parameters are treated as
    fixed).  Column-parallel: column ``j`` uses only the shared per-bin
    arrays and its own per-column scalars, so every ``(i, j)`` entry is
    written exactly once.
    """
    n = centers.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    for j in numba.prange(n):
        c_j = centers[j]
        s_j = sigma[j]
        s2 = s_j * s_j
        ds_dE_j = ds_dE[j]
        grad = np.empty(7, dtype=np.float64)
        for i in range(n):
            d = centers[i] - c_j
            pdf = gaussian_pdf(d, s_j)
            dg_dd = -d / s2 * pdf
            dg_ds = pdf * (d * d / s2 - 1.0) / s_j
            width_i = widths[i]
            for p in range(4):
                grad[p] = (dg_dd * (center_grad[i, p] - center_grad[j, p])
                           * width_i
                           + pdf * width_grad[i, p]
                           + dg_ds * ds_dE_j * center_grad[j, p] * width_i)
            for p in range(4, 7):
                grad[p] = dg_ds * ds_db[j, p - 4] * width_i
            acc = 0.0
            for p in range(7):
                gp = grad[p]
                for q in range(p, 7):
                    term = cov[p, q] * gp * grad[q]
                    if p != q:
                        term += term
                    acc += term
            out[i, j] = acc
    return out


def build_full_response(result: FitResult, channel_max: float,
                        last_channel: int) -> FullResponse:
    """Build the complete energy-to-channel response and its errors.

    ``result`` is the fitted :class:`kc761calib.types.FitResult`; the
    response is built from its fitted core parameters
    ``(c0, k1, k2, k3, b0, b1, b2)`` and the per-element errors from their
    7x7 covariance (including the calib-resol cross block).  The exported
    coefficients are the equivalent plain cubic ``(c0, c1, c2, c3)`` of the
    calibration formula (see
    :func:`kc761calib.response.c0k1k2k3_to_c0c1c2c3`), so the stored
    formula is self-contained, and the exported covariance is in the same
    reported basis.  ``channel_max`` is the upper edge of the detector
    channel axis and ``last_channel`` its last channel index
    (``n_channels = last_channel + 1``).
    """
    calib_params = np.asarray(result.calib_params, dtype=float)
    resol_params = np.asarray(result.resol_params, dtype=float)
    core_cov = np.asarray(result.core_cov, dtype=float)
    if calib_params.shape != (N_CALIB,):
        raise ValueError(
            f"calib_params must have shape ({N_CALIB},), got {calib_params.shape}")
    if resol_params.shape != (N_RESOL,):
        raise ValueError(
            f"resol_params must have shape ({N_RESOL},), got {resol_params.shape}")
    n = int(last_channel) + 1
    if n < 1:
        raise ValueError(f"last_channel must be >= 0, got {last_channel}")

    # Channel bin edges over the full detector range: uniform bins of
    # width 1 with integer centers 0 .. n-1 (edges -0.5 .. n-0.5).  The
    # true-energy edges are the calibration image of these channel edges.
    channel_edges = np.arange(n + 1, dtype=float) - 0.5
    energy_edges = calib_model(calib_params, channel_edges, channel_max)
    if np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(
            "energy calibration is not strictly increasing over the full "
            "channel range; the response-matrix binning requires a "
            "monotone calibration")

    # True-energy bin centers are the midpoints of the bin energy edges --
    # the same quadrature nodes the fit's response matrix uses.
    centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    widths = np.diff(energy_edges)

    # Resolution at the true-energy bin centers; sigma saturates at b0 for
    # E <= 0 (the variance polynomial is only valid on t in [0, 1]).
    sigma = resol_sigma_model(resol_params, centers)

    # R[i, j] = gaussian_pdf(c_i - c_j; sigma_j) * dE_i: the midpoint
    # quadrature of the Gaussian integral over channel bin i, with the
    # density evaluated by the same shared kernel the fit's response
    # matrix uses.  The full Gaussian is kept (no kernel-support cutoff)
    # and the columns are not renormalized, so the probability beyond the
    # detector channel range is truncated, not redistributed onto the
    # edge bins.
    matrix = (gaussian_pdf(centers[:, None] - centers[None, :],
                           sigma[None, :]) * widths[:, None])

    # Per-element 1-sigma errors: linear propagation of the core
    # covariance through the analytic Jacobian of R w.r.t.
    # (c0, k1, k2, k3, b0, b1, b2).
    edge_grad = poly_basis(channel_edges, 3) @ jac_c0k1k2k3(channel_max)
    center_grad = 0.5 * (edge_grad[:-1] + edge_grad[1:])
    width_grad = edge_grad[1:] - edge_grad[:-1]
    ds_dE, ds_db = resol_sigma_model_grad(resol_params, centers)
    if np.all(np.isnan(core_cov)):
        matrix_errors = np.full((n, n), np.nan)
    else:
        # Undetermined parameters are treated as fixed for the matrix
        # errors (their gradient contributions are dropped), matching the
        # band treatment in kc761calib.plot.
        cov_work = np.where(np.isnan(core_cov), 0.0, core_cov)
        var = _matrix_error_variance(centers, widths, sigma, ds_dE, ds_db,
                                     center_grad, width_grad, cov_work)
        # Clamp and square root in place (the tolerance for the tiny
        # negative round-off of the quadratic form) to avoid two extra
        # full-matrix temporaries.
        np.maximum(var, 0.0, out=var)
        np.sqrt(var, out=var)
        matrix_errors = var

    return FullResponse(
        n_channels=n,
        energy_edges=np.asarray(energy_edges, dtype=float),
        matrix=np.asarray(matrix, dtype=float),
        matrix_errors=np.asarray(matrix_errors, dtype=float),
        calib_coeffs=c0k1k2k3_to_c0c1c2c3(calib_params, channel_max),
        resol_params=np.asarray(resol_params, dtype=float),
        param_cov=reported_core_cov(core_cov, channel_max),
    )


def _put_i64(fh, value: int) -> None:
    fh.write(np.int64(value).tobytes())


def _put_f64(fh, values) -> None:
    fh.write(np.asarray(values, dtype=np.float64).tobytes())


def write_export_file(response: FullResponse) -> str:
    """Serialize the response into a new temporary export file.

    Returns the temporary file path.  The file is transient: the ROOT
    writer (:file:`kc761calib/calib2root.cxx`) deletes it after a
    successful conversion, and callers keep it on failure for inspection.
    """
    fd, path = tempfile.mkstemp(prefix="kc761calib-export-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_MAGIC)
            fh.write(CALIB_FORMULA.encode("ascii") + b"\n")
            fh.write(RESOL_FORMULA.encode("ascii") + b"\n")
            _put_i64(fh, response.calib_coeffs.size)
            _put_f64(fh, response.calib_coeffs)
            _put_i64(fh, response.resol_params.size)
            _put_f64(fh, response.resol_params)
            _put_f64(fh, RESOL_E_REF)
            _put_i64(fh, response.param_cov.shape[0])
            _put_f64(fh, response.param_cov.ravel())
            _put_i64(fh, response.n_channels)
            _put_f64(fh, response.energy_edges)
            _put_f64(fh, response.matrix.ravel())
            _put_f64(fh, response.matrix_errors.ravel())
    except BaseException:
        os.unlink(path)
        raise
    return path
