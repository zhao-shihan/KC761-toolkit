"""Binary export of the fitted detector response for the ROOT writer.

The fitted calibration and resolution fully determine the detector
response.  After the fit, this module builds the complete response matrix
over the full detector channel range -- the true-energy bins are the
calibration image of the channel bins -- and serializes it together with
the model formulas (:data:`kc761calib.response.CALIB_FORMULA` /
:data:`kc761calib.response.RESOL_FORMULA`) and the fitted parameters into
a temporary binary file that :file:`kc761calib/calib2root.cxx` reads and
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

Temporary-file layout (native byte order; the file is produced and
consumed on the same machine):

========  =====================================================
line 1    magic ``"kc761calib-export-v1\\n"``
line 2    calibration formula text + ``"\\n"``
line 3    resolution formula text + ``"\\n"``
binary    int64 n_calib; float64 calib_coeffs[n_calib] (c0..c3)
binary    int64 n_resol; float64 resol_params[n_resol] (b0..b2)
binary    float64 resol_e_ref (keV)
binary    int64 n_channels
binary    float64 energy_edges[n_channels + 1]
binary    float64 matrix[n_channels * n_channels] (row-major,
          row = channel bin, column = true-energy bin)
========  =====================================================
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .response import (CALIB_FORMULA, N_CALIB, N_RESOL, RESOL_E_REF,
                       RESOL_FORMULA, calib_model, c0k1k2k3_to_c0c1c2c3,
                       gaussian_pdf, resol_sigma_model)

_MAGIC = b"kc761calib-export-v1\n"


@dataclass
class FullResponse:
    """Complete detector response on the full channel range.

    ``matrix[i, j]`` is the probability that a count in the true-energy
    bin ``j`` is detected in the channel bin ``i``.  Channel bins are the
    full detector range ``0 .. n_channels-1`` (uniform width 1, integer
    centers); the true-energy bins are their calibration image (variable
    width).  Columns are not renormalized, so the Gaussian probability
    truncated by the detector range edges is lost, not redistributed.
    """

    n_channels: int
    energy_edges: np.ndarray  # n_channels + 1; true-energy bin edges
    matrix: np.ndarray  # (n_channels, n_channels) float64
    calib_coeffs: np.ndarray  # c0..c3 cubic calibration coefficients
    resol_params: np.ndarray  # b0..b2 resolution parameters (keV)


def build_full_response(calib_params, resol_params, channel_max: float,
                        last_channel: int) -> FullResponse:
    """Build the complete energy-to-channel response on the full channel range.

    ``calib_params`` is the fitted ``(c0, k1, k2, k3)`` parameterization;
    the exported coefficients are the equivalent plain cubic ``(c0, c1,
    c2, c3)`` of the calibration formula (see
    :func:`kc761calib.response.c0k1k2k3_to_c0c1c2c3`), so the stored
    formula is self-contained.  ``channel_max`` is the upper edge of the
    detector channel axis and ``last_channel`` its last channel index
    (``n_channels = last_channel + 1``).
    """
    calib_params = np.asarray(calib_params, dtype=float)
    resol_params = np.asarray(resol_params, dtype=float)
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
    return FullResponse(
        n_channels=n,
        energy_edges=np.asarray(energy_edges, dtype=float),
        matrix=np.asarray(matrix, dtype=float),
        calib_coeffs=c0k1k2k3_to_c0c1c2c3(calib_params, channel_max),
        resol_params=np.asarray(resol_params, dtype=float),
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
            _put_i64(fh, response.n_channels)
            _put_f64(fh, response.energy_edges)
            _put_f64(fh, response.matrix.ravel())
    except BaseException:
        os.unlink(path)
        raise
    return path
