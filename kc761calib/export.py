"""Binary export of the fitted detector response for the ROOT writer.

After the fit this module builds the complete deposition-to-channel
response matrix over the full detector channel range -- the
energy-deposition bins are the calibration image of the channels --
and serializes it with the model formulas
(:data:`kc761calib.response.CALIB_FORMULA` /
:data:`kc761calib.response.RESOL_FORMULA`), the fitted parameters, their
7x7 covariance and the per-element uncertainties into a temporary binary
file that :file:`kc761calib/calib2root.cxx` reads and turns into the final
ROOT file.

The convention matches :mod:`kc761calib.folding` (``matrix[i, j]`` is the
probability that a count in energy bin ``j`` is detected in channel
``i``); unlike the fit's sparse matrix, the export is dense over the full
channel range, keeps the full Gaussian (no kernel-support cutoff) and does
not renormalize the columns, so the probability beyond the detector range
is truncated, not redistributed.  The response depends only on the 7 core
parameters ``(c0, k1, k2, k3, b0, b1, b2)``; the per-element 1-sigma
uncertainties are the linear propagation ``G^T cov G`` (``G_p =
dR/dq_p``) of the fit's core covariance, matching the reported parameter
uncertainties.  Undetermined parameters (all-NaN covariance rows/columns)
are treated as fixed for the matrix uncertainties while the stored
covariance keeps their
rows/columns as NaN; the stored ``param_cov`` is in the reported basis
``(c0, c1, c2, c3, b0, b1, b2)``.

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
          row = channel, column = energy-deposition bin)
binary    float64 matrix_uncertainties[n_channels * n_channels]
          (row-major, per-element 1-sigma, same layout as matrix)
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
    """Complete dense response on the full channel range (module docstring
    for conventions and uncertainty/covariance semantics)."""

    n_channels: int
    energy_edges: np.ndarray  # n_channels + 1; energy-deposition bin edges
    matrix: np.ndarray  # (n_channels, n_channels) float64
    matrix_uncertainties: np.ndarray  # (n_channels, n_channels) float64
    calib_coeffs: np.ndarray  # c0..c3 cubic calibration coefficients
    resol_params: np.ndarray  # b0..b2 resolution parameters (keV)
    param_cov: np.ndarray  # (7, 7) covariance of (c0..c3, b0..b2)


@numba.njit(parallel=True, cache=True)
def _matrix_uncertainty_variance(centers, widths, sigma, ds_dE, ds_db,
                                 center_grad, width_grad, cov):
    """Squared 1-sigma uncertainty of each response-matrix element.

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
    """Build the complete response from the fitted core parameters.

    The exported coefficients are the equivalent plain cubic
    ``(c0, c1, c2, c3)`` (module docstring), so the stored formula is
    self-contained, and the exported covariance is in the same reported
    basis.  ``channel_max`` is the upper edge of the detector channel
    axis; ``last_channel`` its last index (``n_channels = last_channel +
    1``).
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

    # Channel bins over the full detector range: width 1, integer centers
    # 0 .. n-1 (edges -0.5 .. n-0.5); the energy-deposition edges are the
    # calibration image of these edges.
    channel_edges = np.arange(n + 1, dtype=float) - 0.5
    energy_edges = calib_model(calib_params, channel_edges, channel_max)
    if np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError(
            "energy calibration is not strictly increasing over the full "
            "channel range; the response-matrix binning requires a "
            "monotone calibration")

    # Bin centers are the midpoints of the energy edges -- the same
    # quadrature nodes the fit's sparse matrix uses.
    centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    widths = np.diff(energy_edges)
    sigma = resol_sigma_model(resol_params, centers)

    # Midpoint quadrature of the Gaussian integral over each channel
    # (same kernel as the fit); full Gaussian, no column renormalization.
    matrix = (gaussian_pdf(centers[:, None] - centers[None, :],
                           sigma[None, :]) * widths[:, None])

    # Per-element 1-sigma uncertainties: linear propagation of the core
    # covariance through the analytic Jacobian of R.
    edge_grad = poly_basis(channel_edges, 3) @ jac_c0k1k2k3(channel_max)
    center_grad = 0.5 * (edge_grad[:-1] + edge_grad[1:])
    width_grad = edge_grad[1:] - edge_grad[:-1]
    ds_dE, ds_db = resol_sigma_model_grad(resol_params, centers)
    if np.all(np.isnan(core_cov)):
        matrix_uncertainties = np.full((n, n), np.nan)
    else:
        # Undetermined parameters are fixed for the matrix uncertainties
        # (their gradient contributions are dropped), as in the plot band.
        cov_work = np.where(np.isnan(core_cov), 0.0, core_cov)
        var = _matrix_uncertainty_variance(
            centers, widths, sigma, ds_dE, ds_db,
            center_grad, width_grad, cov_work)
        # Clamp and sqrt in place (tolerate tiny negative round-off of the
        # quadratic form) to avoid extra full-matrix temporaries.
        np.maximum(var, 0.0, out=var)
        np.sqrt(var, out=var)
        matrix_uncertainties = var

    return FullResponse(
        n_channels=n,
        energy_edges=np.asarray(energy_edges, dtype=float),
        matrix=np.asarray(matrix, dtype=float),
        matrix_uncertainties=np.asarray(matrix_uncertainties, dtype=float),
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
            _put_f64(fh, response.matrix_uncertainties.ravel())
    except BaseException:
        os.unlink(path)
        raise
    return path
