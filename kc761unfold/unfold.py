"""Top-level mode orchestration: Hybrid regularized unfold and calibration-only.

Hybrid regularized unfolding deconvolves the measured KC761
channel spectra according to the given to-channel matrix. The unfolded
spectrum ``mu`` minimizes

    chi2 = sum_j (y_j - (R mu)_j)^2 / sigma_j^2,
    sigma_j^2 = max(stat_j^2, 1) + (syst_frac y_j)^2

under ``mu >= 0`` with the quadratic penalty ``alpha ||D mu||^2``
(``D`` the significance-normalized, SNIP-masked difference operator of
:mod:`kc761unfold.penalty`).

Errors are analytic (:mod:`kc761unfold.errors`): statistical and
systematic (calibration-parameter) covariances of the estimator.  The
calibration-only mode relabels the channel axis to energy without
unfolding.  Both modes produce a
:class:`kc761unfold.types.UnfoldResult` consumed by the ROOT export and
the plot paths.
"""

from __future__ import annotations

import numpy as np

from .errors import (calibration_vertical_term, compute_covariances,
                     energy_center_errors)
from .penalty import peak_mask, snip_baseline, penalty_operator
from .solver import UnfoldProblem
from .types import CalibrationFile, UnfoldSettings, UnfoldResult


def _calibrated_layer(counts: np.ndarray, errors: np.ndarray,
                      calib: CalibrationFile,
                      settings: UnfoldSettings) -> tuple[np.ndarray, ...]:
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


def _result(calib_only: bool, calib: CalibrationFile, settings: UnfoldSettings,
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
               data_errors: np.ndarray, settings: UnfoldSettings
               ) -> UnfoldResult:
    """Unfold one spectrum with the hybrid regularization."""
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high
    y = np.asarray(data_counts[ch_lo:ch_hi + 1], dtype=float)
    err = np.asarray(data_errors[ch_lo:ch_hi + 1], dtype=float)

    sigma2 = np.maximum(err ** 2, 1.0) + (settings.syst_frac * y) ** 2
    sigma = np.sqrt(sigma2)
    w = 1.0 / sigma2

    baseline = snip_baseline(y, settings.snip_iter)
    mask = peak_mask(y, sigma, baseline, settings.mask_z0,
                     settings.mask_floor)
    d_op = penalty_operator(calib.widths, calib.centers,
                            sigma, mask, settings.k)

    prob = UnfoldProblem(calib.to_channel, y, w, d_op, settings.alpha)
    mu = prob.solve()
    refolded = prob.r @ mu

    c_stat, c_sys, sig_stat, sig_syst = compute_covariances(
        prob, mu, prob.free, calib, sigma, mask, settings.k)

    data_total, data_syst, _, sigma_calib = _calibrated_layer(
        y, err, calib, settings)

    return _result(
        False, calib, settings, mu, y, sig_stat, sig_syst,
        np.sqrt(sig_stat ** 2 + sig_syst ** 2), sigma_calib,
        data_total, data_syst,
        stat_cov=c_stat, syst_cov=c_sys, refolded=refolded,
        chi2=prob.chi2, ndof=int(prob.free.sum()), pen_cost=prob.pen_cost,
        n_iter=prob.n_iter)


def run_calib_only(calib: CalibrationFile, data_counts: np.ndarray,
                   data_errors: np.ndarray, settings: UnfoldSettings
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
