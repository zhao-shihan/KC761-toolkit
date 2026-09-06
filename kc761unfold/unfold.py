"""Top-level mode orchestration: SWR unfold and calibration-only.

SWR (significance-weighted robust) unfolding removes the detector's
resolution smearing from background-subtracted KC761 channel spectra
using the response matrix of a kc761calib export.  The unfolded
spectrum ``mu`` (counts per variable-width energy bin) minimizes

    chi2 = sum_j (y_j - (R mu)_j)^2 / sigma_j^2,
    sigma_j^2 = max(stat_j^2, 1) + (f y_j)^2

under ``mu >= 0``, regularized by the SWR penalty

    Omega = sum_j rho_delta( m_j (L_k rho)_j / sigma_d,j ),  rho = mu / w

with ``L_k`` the k-th difference on the energy grid, ``sigma_d,j`` the
statistical noise of that difference, ``m_j`` a SNIP peak mask, and
``rho_delta`` the Huber loss.  See kc761unfold.penalty for the design
and the literature sources of the three components (SNIP: Ryan et al.,
Nucl. Instrum. Methods B 34 (1988) 396-402; Huber loss: Huber, Ann.
Math. Statist. 35 (1964) 73-101; the significance normalization and the
combination are this toolkit's design).  The unfolded spectrum keeps
the Compton continuum; sub-resolution peak widths are regularization-
limited (not identifiable from data) while peak areas are conserved.

Errors are analytic (no Monte Carlo): the statistical covariance
``M Sigma_y M^T`` with ``M = 2 H^-1 R^T W`` and the systematic
(calibration-parameter) covariance ``J Sigma_q J^T`` with
``J = d mu/dq = -H^-1 d^2L/d mu d q``; the total per-bin error is their
quadrature sum (see kc761unfold.errors).

The calibration-only mode relabels the channel axis to energy without
unfolding; the calibration-model error is propagated vertically through
the spectrum derivative, ``|dy/dE| sigma_E``, into the errors.

Both modes produce a :class:`kc761unfold.types.UnfoldResult`, the single
structure consumed by the ROOT export and the plot paths.  The ROOT
output (kc761unfold/unfold2root.cxx) contains the spectrum TH1D with
the total errors, the statistical and systematic covariance TH2Ds and
the refolded spectrum (unfold mode), and the settings (``syst_frac``,
``elo``/``ehi``, ``chlo``/``chhi``, ``calib_only``, and the SWR
parameters in unfold mode).
"""

from __future__ import annotations

import numpy as np

from .errors import (calibration_vertical_term, compute_covariances,
                     energy_center_errors)
from .penalty import peak_mask, snip_baseline, swr_operator
from .solver import SWRProblem
from .types import CalibrationFile, SWRSettings, UnfoldResult


def _calibrated_layer(counts: np.ndarray, errors: np.ndarray,
                      calib: CalibrationFile,
                      settings: SWRSettings) -> tuple[np.ndarray, ...]:
    """Per-bin errors of the calibrated-spectrum layer.

    Returns ``(sigma_total, sigma_syst, sigma_stat, sigma_calib)`` with
    the fractional systematic and the calibration-model vertical term
    combined into the systematic part.
    """
    ch = np.arange(calib.channel_low, calib.channel_high + 1, dtype=float)
    sigma_e = energy_center_errors(ch, calib.param_cov)
    sigma_calib = calibration_vertical_term(counts, sigma_e, calib.centers,
                                            calib.resol_params)
    stat2 = np.asarray(errors, dtype=float) ** 2
    syst2 = (settings.syst_frac * counts) ** 2 + sigma_calib ** 2
    return (np.sqrt(stat2 + syst2), np.sqrt(syst2), np.sqrt(stat2),
            sigma_calib)


def _result(calib_only: bool, calib: CalibrationFile, settings: SWRSettings,
            counts: np.ndarray, data_counts: np.ndarray,
            sigma_stat: np.ndarray, sigma_syst: np.ndarray,
            sigma_total: np.ndarray, sigma_calib: np.ndarray,
            data_total: np.ndarray, data_syst: np.ndarray,
            **unfold_fields) -> UnfoldResult:
    """Assemble the shared UnfoldResult fields; unfold fields come last.

    ``counts`` is the primary spectrum (unfolded mu_hat, or the raw
    counts in calibration-only mode) and ``data_counts`` the calibrated
    spectrum layer (the raw counts on the energy axis).
    """
    defaults = dict(stat_cov=None, syst_cov=None, refolded=None,
                    chi2=None, ndof=None, pen_cost=None, n_iter=None)
    defaults.update(unfold_fields)
    return UnfoldResult(
        calib_only=calib_only,
        n_bins=calib.n_bins,
        channel_low=calib.channel_low,
        channel_high=calib.channel_high,
        energy_edges=calib.energy_edges,
        centers=calib.centers,
        counts=counts,
        sigma_total=sigma_total,
        sigma_stat=sigma_stat,
        sigma_syst=sigma_syst,
        sigma_calib=sigma_calib,
        data_counts=np.asarray(data_counts, dtype=float),
        data_sigma_total=data_total,
        data_sigma_syst=data_syst,
        settings=settings,
        **defaults,
    )


def run_unfold(calib: CalibrationFile, data_counts: np.ndarray,
               data_errors: np.ndarray, settings: SWRSettings
               ) -> UnfoldResult:
    """Unfold one spectrum with the SWR regularization."""
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high
    y = np.asarray(data_counts[ch_lo:ch_hi + 1], dtype=float)
    err = np.asarray(data_errors[ch_lo:ch_hi + 1], dtype=float)

    sigma2 = np.maximum(err ** 2, 1.0) + (settings.syst_frac * y) ** 2
    sigma = np.sqrt(sigma2)
    w = 1.0 / sigma2

    baseline = snip_baseline(y, settings.snip_iter)
    mask = peak_mask(y, sigma, baseline, settings.p0, settings.gmin)
    d_op = swr_operator(calib.widths, calib.centers, sigma, mask, settings.k)

    prob = SWRProblem(calib.matrix, y, w, d_op, settings.alpha,
                      settings.delta)
    mu = prob.solve()
    nu = prob.r @ mu

    c_stat, c_sys, sig_stat, sig_syst = compute_covariances(
        prob, mu, prob.free, sigma2, calib, sigma, mask, settings.k)

    data_total, data_syst, _, sigma_calib = _calibrated_layer(
        y, err, calib, settings)

    return _result(
        False, calib, settings, mu, y, sig_stat, sig_syst,
        np.sqrt(sig_stat ** 2 + sig_syst ** 2), sigma_calib,
        data_total, data_syst,
        stat_cov=c_stat, syst_cov=c_sys, refolded=nu,
        chi2=prob.chi2, ndof=int(prob.free.sum()), pen_cost=prob.pen_cost,
        n_iter=prob.n_iter)


def run_calib_only(calib: CalibrationFile, data_counts: np.ndarray,
                   data_errors: np.ndarray, settings: SWRSettings
                   ) -> UnfoldResult:
    """Relabel the channel spectrum onto the energy axis (no unfolding)."""
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high
    y = np.asarray(data_counts[ch_lo:ch_hi + 1], dtype=float)
    err = np.asarray(data_errors[ch_lo:ch_hi + 1], dtype=float)

    total, syst, stat, sigma_calib = _calibrated_layer(y, err, calib,
                                                       settings)

    return _result(True, calib, settings, y, y, stat, syst, total,
                   sigma_calib, total, syst)
