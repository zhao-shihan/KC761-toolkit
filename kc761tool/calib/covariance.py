"""Calibration covariance extraction (F-CAL-5, D-105).

The reported ``(c0..c3, b0..b2)`` covariance is the ``(c, b)`` block of the
full-parameter Fisher inverse with the per-dataset scale **marginalized**
(scaled by the global ``s**2 = chi2/dof``, F-COV-2) and transformed from the
internal slope basis to the reported cubic basis with the constant F-MODEL-2
Jacobian (``b`` unchanged).

The quadratic-Bezier scale has a singular stratum: for a constant scale
or a quadratic-polynomial scale (``s0`` at the window midpoint) the ``s0``
control channel is an exact gauge of the four-parameter representation, and
more generally it is only weakly identified when the data are nearly
polynomial. Forming and inverting the full Fisher matrix then loses all
precision (or fails outright). The block inverse identity

    (F^-1)_core = (F_cc - F_cs F_ss^+ F_sc)^-1

is used instead, with Jacobi (diagonal) preconditioning and the Moore-Penrose
inverse ``F_ss^+`` of the scale block. For an invertible scale block this is
algebraically the same core block as inverting the full matrix (verified in the
tests); when the scale block has a gauge direction it is the correct
marginalization rather than a numerical accident. A non-positive-definite core
Schur complement is still a hard failure: the fit parameters themselves are not
identifiable (no pseudo-inverse fallback for the core).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761tool.core._checks import as_float_array
from kc761tool.core.covariance import CovarianceEstimate, fisher_information
from kc761tool.core.model import N_REPORTED_PARAMS, internal_jacobian
from kc761tool.errors import SolverError, ValidationError

N_REPORTED: int = N_REPORTED_PARAMS
"""Reported core size ``(c0, c1, c2, c3, b0, b1, b2)``."""

_SCALE_GAUGE_RCOND: float = 1e-12
"""Relative cutoff of the scale-block pseudo-inverse (the gauge directions)."""


def calibration_covariance(
    jacobian: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    chi2: float,
    dof: int,
    channel_max: float,
) -> CovarianceEstimate:
    """Reported-basis 7x7 covariance of the fitted calibration (F-CAL-5).

    ``jacobian`` is the prediction Jacobian ``(n_bins, n_free)`` in the
    internal basis (F-CAL-4) and ``variance`` the matching F-CAL-1 fit
    variance. A singular or non-positive-definite core Schur complement raises
    ``SolverError``.
    """
    jac = as_float_array("jacobian", jacobian, ndim=2)
    var = as_float_array("variance", variance, ndim=1)
    if jac.shape[0] != var.size:
        raise ValidationError(
            f"jacobian has {jac.shape[0]} rows but variance has {var.size} entries"
        )
    if jac.shape[1] < N_REPORTED:
        raise ValidationError(
            f"jacobian must carry at least {N_REPORTED} parameters, got {jac.shape[1]}"
        )
    if dof < 1:
        raise ValidationError(f"covariance scale needs dof >= 1, got {dof}")
    if not np.isfinite(chi2) or chi2 < 0.0:
        raise ValidationError(f"chi2 must be finite and non-negative, got {chi2!r}")

    fisher = np.asarray(fisher_information(jac, np.sqrt(var)), dtype=np.float64)
    fisher = 0.5 * (fisher + fisher.T)
    diagonal = np.diag(fisher)
    if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
        raise SolverError("fisher diagonal is not strictly positive; parameters are not identified")
    scale = np.sqrt(diagonal)
    preconditioned = fisher / np.outer(scale, scale)

    core = slice(0, N_REPORTED)
    f_cc = preconditioned[core, core]
    f_cs = preconditioned[core, N_REPORTED:]
    f_ss = preconditioned[N_REPORTED:, N_REPORTED:]
    if f_ss.size:
        f_ss_inverse = np.linalg.pinv(f_ss, rcond=_SCALE_GAUGE_RCOND, hermitian=True)
        schur = f_cc - f_cs @ f_ss_inverse @ f_cs.T
    else:
        schur = f_cc
    schur = 0.5 * (schur + schur.T)
    try:
        core_scaled = np.linalg.inv(schur)
    except np.linalg.LinAlgError as exc:
        raise SolverError(f"calibration core Schur complement is singular: {exc}") from exc
    core_covariance = core_scaled / np.outer(scale[:N_REPORTED], scale[:N_REPORTED])

    transform = np.zeros((N_REPORTED, N_REPORTED), dtype=np.float64)
    transform[:4, :4] = internal_jacobian(channel_max=channel_max)
    transform[4:, 4:] = np.eye(3)
    reported = transform @ core_covariance @ transform.T
    reported = 0.5 * (reported + reported.T)
    scale_squared = chi2 / dof
    return CovarianceEstimate(
        matrix=np.asarray(reported * scale_squared, dtype=np.float64),
        scale=float(scale_squared),
        chi2=float(chi2),
        dof=int(dof),
        estimator="fisher-x2dof-marginalized-reported",
    )


__all__ = ["N_REPORTED", "calibration_covariance"]
