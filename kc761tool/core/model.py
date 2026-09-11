"""Detector response model: energy calibration ``E(channel)`` and resolution.

Formula IDs (docs/derivations.md): F-MODEL-1 .. F-MODEL-5.

Frozen decisions (docs/plan.md):

* F-MODEL-1/F-MODEL-2: the fit works in the internal slope basis
  ``(c0, k1, k2, k3)``; products store the plain cubic ``(c0, c1, c2, c3)``.
  The basis transform and its constant Jacobian have exactly one
  implementation, in ``kc761tool/core/_gen/model_expr.py`` (sympy-generated).
* F-MODEL-3: the fitted bounds keep ``E(channel)`` strictly increasing on
  ``[0, channel_max]``; :func:`verify_energy_monotonicity` is the strict-mode
  certificate.
* F-MODEL-4: the resolution is the quadratic Bernstein form in
  ``t = max(E, 0) / RESOL_E_REF_KEV`` with control values
  ``(b0**2, b1**2, b2**2)``. The variance is used exactly as written: no
  absolute value and no variance floor.
* F-MODEL-5/D-73: strict mode raises :class:`kc761tool.errors.CertificateError`
  when ``sigma**2 < 0``. Outside strict mode the resolution is explicitly
  clamped to :data:`SIGMA_FLOOR_KEV`, a warning is emitted and
  :func:`resolution_clamped_mask` lets the caller record the affected bins,
  which the unfold layer stores in ``meta``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kc761tool.core import _gen
from kc761tool.core._checks import as_float_array, require_positive
from kc761tool.errors import CertificateError, ValidationError

RESOL_E_REF_KEV = 2000.0
"""Reference energy of the Bernstein resolution polynomial (keV)."""

SIGMA_POSITIVITY_TOL_KEV2 = 1e-9
"""Numerical tolerance of the F-MODEL-5 certificate (float noise only)."""

SIGMA_FLOOR_KEV = 1e-3
"""Documented non-strict floor of ``sigma`` when ``sigma**2 <= 0`` (D-73)."""

N_CALIB = 4
N_RESOL = 3

N_REPORTED_PARAMS = N_CALIB + N_RESOL
"""Reported fit parameters ``(c0..c3, b0..b2)`` (single source; F-RESP-4)."""

PARAM_NAMES_REPORTED: tuple[str, ...] = (
    "c0", "c1", "c2", "c3", "b0", "b1", "b2")


@dataclass(frozen=True)
class InternalCalibration:
    """Internal slope-basis calibration ``(c0, k1, k2, k3)`` (F-MODEL-1)."""

    c0: float
    k1: float
    k2: float
    k3: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.c0, self.k1, self.k2, self.k3], dtype=np.float64)

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> InternalCalibration:
        c0, k1, k2, k3 = (float(v) for v in values)
        return cls(c0=c0, k1=k1, k2=k2, k3=k3)


@dataclass(frozen=True)
class ReportedCalibration:
    """Plain cubic ``(c0, c1, c2, c3)`` as stored in products (F-MODEL-2)."""

    c0: float
    c1: float
    c2: float
    c3: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.c0, self.c1, self.c2, self.c3], dtype=np.float64)

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> ReportedCalibration:
        c0, c1, c2, c3 = (float(v) for v in values)
        return cls(c0=c0, c1=c1, c2=c2, c3=c3)


def _resol_params(resol_params: NDArray[np.float64]) -> NDArray[np.float64]:
    params = as_float_array("resol_params", resol_params, ndim=1)
    if params.size != N_RESOL:
        raise ValidationError(
            f"resol_params must have {N_RESOL} entries, got {params.size}")
    return params


def _check_channel_max(channel_max: float) -> float:
    return require_positive("channel_max", channel_max)


def energy_kev(
    channel: NDArray[np.float64],
    calibration: InternalCalibration,
    *,
    channel_max: float,
) -> NDArray[np.float64]:
    """Evaluate ``E(channel)`` in keV from the internal slope basis.

    Formula: F-MODEL-1. ``channel_max`` is the maximum channel number of the
    acquisition and is a fixed constant of the data.
    """
    channel_array = as_float_array("channel", channel)
    maximum = _check_channel_max(channel_max)
    return _gen.model_expr.energy_kev(
        channel_array, calibration.c0, calibration.k1, calibration.k2, calibration.k3, maximum
    )


def energy_derivative_kev_per_channel(
    channel: NDArray[np.float64],
    calibration: InternalCalibration,
    *,
    channel_max: float,
) -> NDArray[np.float64]:
    """``dE/dch`` in keV per channel (F-MODEL-3 monotonicity check)."""
    channel_array = as_float_array("channel", channel)
    maximum = _check_channel_max(channel_max)
    return _gen.model_expr.energy_derivative_kev_per_channel(
        channel_array, calibration.k1, calibration.k2, calibration.k3, maximum
    )


def energy_grad_internal(
    channel: NDArray[np.float64],
    *,
    channel_max: float,
) -> NDArray[np.float64]:
    """``dE/d(c0, k1, k2, k3)`` in keV, shape ``(4, *channel.shape)`` (F-MODEL-2)."""
    channel_array = as_float_array("channel", channel)
    maximum = _check_channel_max(channel_max)
    return _gen.model_expr.energy_grad_internal(channel_array, maximum)


def energy_grad_reported(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    """``dE/d(c0, c1, c2, c3)`` in keV, shape ``(4, *channel.shape)`` (F-MODEL-2)."""
    channel_array = as_float_array("channel", channel)
    return _gen.model_expr.energy_grad_reported(channel_array)


def internal_to_reported(
    calibration: InternalCalibration, *, channel_max: float
) -> ReportedCalibration:
    """Map the internal slope basis to the reported cubic (F-MODEL-2)."""
    maximum = _check_channel_max(channel_max)
    values = _gen.model_expr.internal_to_reported(
        calibration.c0, calibration.k1, calibration.k2, calibration.k3, maximum
    )
    return ReportedCalibration.from_array(np.asarray(values, dtype=np.float64))


def reported_to_internal(
    reported: ReportedCalibration, *, channel_max: float
) -> InternalCalibration:
    """Invert the reported-to-internal map (F-MODEL-2)."""
    maximum = _check_channel_max(channel_max)
    values = _gen.model_expr.reported_to_internal(
        reported.c0, reported.c1, reported.c2, reported.c3, maximum
    )
    return InternalCalibration.from_array(np.asarray(values, dtype=np.float64))


def internal_jacobian(*, channel_max: float) -> NDArray[np.float64]:
    """Constant Jacobian ``d(reported)/d(internal)`` (F-MODEL-2)."""
    maximum = _check_channel_max(channel_max)
    return _gen.model_expr.internal_jacobian(maximum)


def resolution_variance_kev2(
    energy_kev: NDArray[np.float64], resol_params: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Bernstein ``sigma**2(E)`` in keV^2, used exactly as written (F-MODEL-4)."""
    energy = as_float_array("energy_kev", energy_kev)
    params = _resol_params(resol_params)
    return _gen.model_expr.resolution_variance(
        energy, params[0], params[1], params[2], RESOL_E_REF_KEV
    )


def resolution_sigma_kev(
    energy_kev: NDArray[np.float64],
    resol_params: NDArray[np.float64],
    *,
    strict: bool = False,
) -> NDArray[np.float64]:
    """``sigma(E)`` in keV (F-MODEL-4/F-MODEL-5).

    Strict mode raises the F-MODEL-5 certificate error on a negative variance.
    Outside strict mode a non-positive variance is clamped to
    :data:`SIGMA_FLOOR_KEV` and a warning is emitted (D-73).
    """
    energy = as_float_array("energy_kev", energy_kev)
    params = _resol_params(resol_params)
    variance = _gen.model_expr.resolution_variance(
        energy, params[0], params[1], params[2], RESOL_E_REF_KEV
    )
    clip_mask = _evaluate_clip_mask(variance, strict=strict)
    return _sigma_from_variance(energy, params, variance, clip_mask, strict=strict)


def resolution_sigma_grad(
    energy_kev: NDArray[np.float64],
    resol_params: NDArray[np.float64],
    *,
    strict: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``sigma(E)`` and ``d sigma / d(b0, b1, b2)`` (F-MODEL-4).

    Value and derivative come from one sympy-generated expression; the
    derivative is never hand-written (docs/derivations.md section 2). In the
    clamped branch the derivative is exactly zero, consistent with the
    constant floor (D-73).
    """
    energy = as_float_array("energy_kev", energy_kev)
    params = _resol_params(resol_params)
    variance = _gen.model_expr.resolution_variance(
        energy, params[0], params[1], params[2], RESOL_E_REF_KEV
    )
    clip_mask = _evaluate_clip_mask(variance, strict=strict)

    flat_energy = energy.reshape(-1)
    flat_mask = clip_mask.reshape(-1)
    gap = ~flat_mask
    sigma = np.empty(flat_energy.shape, dtype=np.float64)
    gradient = np.zeros((N_RESOL, flat_energy.size), dtype=np.float64)
    sigma[flat_mask] = 0.0 if strict else SIGMA_FLOOR_KEV
    if gap.any():
        sigma_gap, gradient_gap = _gen.model_expr.resolution_sigma_grad(
            flat_energy[gap],
            params[0],
            params[1],
            params[2],
            RESOL_E_REF_KEV,
        )
        sigma[gap] = sigma_gap
        gradient[:, gap] = gradient_gap
    return sigma.reshape(energy.shape), gradient.reshape((N_RESOL, *energy.shape))


def _evaluate_clip_mask(variance: NDArray[np.float64], *, strict: bool) -> NDArray[np.bool_]:
    """Return the mask of energies whose variance is not usable in the kernel.

    Strict mode fails the F-MODEL-5 certificate for a negative variance
    beyond the float-noise tolerance and maps values in
    ``[-tol, 0]`` to the zero-width branch (the certificate allows zero).
    Outside strict mode every non-positive variance is clamped to the floor.
    """
    if strict:
        if np.any(variance < -SIGMA_POSITIVITY_TOL_KEV2):
            worst = float(np.min(variance))
            raise CertificateError(
                "F-MODEL-5",
                f"negative resolution variance below tolerance: min={worst:.6g} keV^2",
            )
        return variance <= 0.0
    mask = variance <= 0.0
    if np.any(variance < 0.0):
        count = int(np.count_nonzero(variance < 0.0))
        warnings.warn(
            "F-MODEL-5: sigma**2 < 0 at "
            f"{count} of {variance.size} energies; sigma clamped to "
            f"{SIGMA_FLOOR_KEV} keV (D-73)",
            RuntimeWarning,
            stacklevel=3,
        )
    return mask


def _sigma_from_variance(
    energy: NDArray[np.float64],
    params: NDArray[np.float64],
    variance: NDArray[np.float64],
    clip_mask: NDArray[np.bool_],
    *,
    strict: bool,
) -> NDArray[np.float64]:
    """Evaluate the generated sigma expression away from the clamped branch."""
    flat_energy = energy.reshape(-1)
    flat_mask = clip_mask.reshape(-1)
    sigma = np.empty(flat_energy.shape, dtype=np.float64)
    sigma[flat_mask] = 0.0 if strict else SIGMA_FLOOR_KEV
    gap = ~flat_mask
    if gap.any():
        sigma[gap] = _gen.model_expr.resolution_sigma(
            flat_energy[gap],
            params[0],
            params[1],
            params[2],
            RESOL_E_REF_KEV,
        )
    return sigma.reshape(energy.shape)


def resolution_clamped_mask(
    energy_kev: NDArray[np.float64],
    resol_params: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Energies whose resolution variance is clamped outside strict mode (D-73).

    The caller uses this to record how many bins were affected by the documented
    floor.
    """
    energy = as_float_array("energy_kev", energy_kev)
    params = _resol_params(resol_params)
    variance = _gen.model_expr.resolution_variance(
        energy, params[0], params[1], params[2], RESOL_E_REF_KEV
    )
    return variance <= 0.0


def verify_resolution_positivity(
    energy_kev: NDArray[np.float64],
    resol_params: NDArray[np.float64],
    *,
    strict: bool,
) -> None:
    """F-MODEL-5 certificate: ``sigma**2 >= -tol`` on the export grid.

    Strict mode raises :class:`kc761tool.errors.CertificateError` and reports the
    formula ID. Outside strict mode a violation only emits a warning; the
    value clamp itself lives in :func:`resolution_sigma_kev` (D-73).
    """
    energy = as_float_array("energy_kev", energy_kev)
    params = _resol_params(resol_params)
    variance = _gen.model_expr.resolution_variance(
        energy, params[0], params[1], params[2], RESOL_E_REF_KEV
    )
    if strict:
        if np.any(variance < -SIGMA_POSITIVITY_TOL_KEV2):
            worst = float(np.min(variance))
            raise CertificateError(
                "F-MODEL-5",
                f"negative resolution variance below tolerance: min={worst:.6g} keV^2",
            )
        return
    if np.any(variance < 0.0):
        count = int(np.count_nonzero(variance < 0.0))
        warnings.warn(
            "F-MODEL-5: sigma**2 < 0 at "
            f"{count} of {variance.size} energies; strict mode would fail (D-73)",
            RuntimeWarning,
            stacklevel=2,
        )


def verify_energy_monotonicity(
    calibration: InternalCalibration,
    *,
    channel_max: float,
    strict: bool,
) -> None:
    """F-MODEL-3 certificate: ``dE/dch > 0`` on ``[0, channel_max]``.

    ``E(ch)`` is cubic, so ``dE/dch`` is quadratic; its minimum on the closed
    interval is attained at an endpoint or at the exact vertex, which this
    check evaluates (no sampling grid can hide a dip between points).
    Strict mode raises :class:`kc761tool.errors.CertificateError` when the slope
    is not positive.
    """
    maximum = _check_channel_max(channel_max)
    if not strict:
        return
    quarter = 0.5 * maximum
    derivative = energy_derivative_kev_per_channel(
        np.array([0.0, quarter, maximum]), calibration, channel_max=maximum
    )
    d0, dmid, d1 = (float(value) for value in derivative)
    # q(ch) = A ch**2 + B ch + C with d0 = C, dmid = A h**2/4 + B h/2 + C,
    # d1 = A h**2 + B h + C, h = channel_max.
    coefficient_a = 2.0 * (d1 + d0 - 2.0 * dmid) / maximum**2
    coefficient_b = (d1 - d0) / maximum - coefficient_a * maximum
    candidates = [0.0, maximum]
    if coefficient_a != 0.0:
        vertex = -coefficient_b / (2.0 * coefficient_a)
        if 0.0 < vertex < maximum:
            candidates.append(vertex)
    minimum = min(
        float(
            energy_derivative_kev_per_channel(
                np.array([candidate]), calibration, channel_max=maximum
            )[0]
        )
        for candidate in candidates
    )
    if minimum <= 0.0:
        raise CertificateError(
            "F-MODEL-3",
            f"E(channel) is not strictly increasing on [0, {maximum}]: "
            f"min dE/dch={minimum:.6g} keV/channel",
        )
