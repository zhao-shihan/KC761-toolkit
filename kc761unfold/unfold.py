"""Top-level mode orchestration: Hybrid regularized unfold and calibration-only.

Hybrid regularized unfolding deconvolves the measured KC761
channel spectra according to the composed to-channel matrix
``R = C p_tilde diag(eta)`` (the calibration deposition response
composed with the simulation's primary-to-deposition step; see
:mod:`kc761unfold.response`).  The unfolded spectrum ``mu`` minimizes

    chi2 = sum_j (y_j - (R mu)_j)^2 / sigma_j^2,
    sigma_j^2 = max(stat_j^2, 1) + (syst_frac y_j)^2

under ``mu >= 0`` with the quadratic penalty ``alpha ||D mu||^2``
(``D`` the solution-noise-normalized, SNIP-masked difference operator of
:mod:`kc761unfold.penalty`).

The working window is padded on both sides by ``pad_nsigma`` resolution
widths (clipped to the detector range).  The pads extend the response
both as columns (their mu variables can feed the window through the
resolution tails) and as rows (the window mu's tails have somewhere to
land), but their own data carries zero weight in the chi-square -- the
pad energies cover a region whose response and scale differ strongly
from the window's, and fitting it corrupted the low-energy window
solution.  The pads are cropped from every output and the reported
chi2/ndof counts the window bins only.  The regularization strength
``alpha`` is a fixed CLI setting.

Uncertainties are per-bin only (the full covariance matrices are not
computed or exported): the statistical and systematic 1-sigma bands come
from the diagonals of the analytic propagation
(:mod:`kc761unfold.uncertainties`), with the systematic part combining
the calibration-parameter propagation and the simulation-file Monte Carlo column
contribution.  The calibration-only mode relabels the channel axis to
energy without unfolding.  Both modes produce a
:class:`kc761unfold.types.UnfoldResult` consumed by the ROOT export and
the plot paths.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .uncertainties import (calibration_vertical_term, compute_covariances,
                            effective_ndof, energy_center_uncertainties)
from .penalty import peak_mask, snip_baseline, penalty_operator
from .reader import slice_calibration, slice_sim
from .response import to_channel_and_derivatives
from .solver import UnfoldProblem, embed_and_factor
from .types import CalibrationFile, UnfoldSettings, UnfoldResult
from kc761calib.response import resol_sigma_model

# Weight of the weak zero prior that anchors the pad mu's constant mode.
_PAD_ANCHOR = 1e-8


def _padding(calib: CalibrationFile, settings: UnfoldSettings,
             ) -> tuple[int, int, int, int]:
    """Padded channel range around the window: (pad_lo, pad_hi, win_lo, win_hi).

    The window ``[channel_low, channel_high]`` (channels of the full
    calibration) is widened by ``pad_nsigma`` local resolution widths on
    each side, clipped to the detector range.  ``win_lo``/``win_hi`` are
    the window's indices inside the padded slice.
    """
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high
    n = calib.n_channels
    sigma = resol_sigma_model(np.asarray(calib.resol_params, dtype=float),
                              calib.centers)
    width_lo = max(calib.widths[ch_lo], 1e-9)
    width_hi = max(calib.widths[ch_hi], 1e-9)
    n_pad_lo = int(np.ceil(settings.pad_nsigma * sigma[ch_lo] / width_lo))
    n_pad_hi = int(np.ceil(settings.pad_nsigma * sigma[ch_hi] / width_hi))
    pad_lo = max(0, ch_lo - n_pad_lo)
    pad_hi = min(n - 1, ch_hi + n_pad_hi)
    return pad_lo, pad_hi, ch_lo - pad_lo, ch_hi - pad_lo


def _calibrated_layer(counts: np.ndarray, uncertainties: np.ndarray,
                      calib: CalibrationFile,
                      settings: UnfoldSettings) -> tuple[np.ndarray, ...]:
    """Per-bin uncertainties of the calibrated-spectrum layer.

    Returns ``(sigma_total, sigma_syst, sigma_stat, sigma_calib)`` with
    the fractional systematic and the calibration-model vertical term
    combined into the systematic part.
    """
    ch = np.arange(calib.channel_low, calib.channel_high + 1, dtype=float)
    sigma_e = energy_center_uncertainties(ch, calib.param_cov)
    sigma_calib = calibration_vertical_term(counts, sigma_e, calib.centers,
                                            calib.resol_params)
    stat2 = np.asarray(uncertainties, dtype=float) ** 2
    syst2 = (settings.syst_frac * counts) ** 2 + sigma_calib ** 2
    return (np.sqrt(stat2 + syst2), np.sqrt(syst2), np.sqrt(stat2),
            sigma_calib)


def _result(calib_only: bool, calib: CalibrationFile, settings: UnfoldSettings,
            counts: np.ndarray, data_counts: np.ndarray,
            fit_sigma: np.ndarray | None,
            sigma_stat: np.ndarray, sigma_syst: np.ndarray,
            sigma_total: np.ndarray, sigma_calib: np.ndarray,
            data_total: np.ndarray, data_syst: np.ndarray,
            **unfold_fields) -> UnfoldResult:
    """Assemble the shared UnfoldResult fields; unfold fields come last.

    ``counts`` is the primary spectrum (unfolded mu_hat, or the raw
    counts in calibration-only mode) and ``data_counts`` the calibrated
    spectrum layer (the raw counts on the energy axis).
    """
    defaults = dict(refolded=None,
                    chi2=None, ndof=None, pen_cost=None, n_iter=None,
                    converged=None)
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
        fit_sigma=fit_sigma,
        data_counts=np.asarray(data_counts, dtype=float),
        data_sigma_total=data_total,
        data_sigma_syst=data_syst,
        settings=settings,
        **defaults,
    )


def run_unfold(calib: CalibrationFile, sim, data_counts: np.ndarray,
               data_uncertainties: np.ndarray, settings: UnfoldSettings
               ) -> UnfoldResult:
    """Unfold one spectrum with the hybrid regularization.

    ``calib`` is the FULL-range calibration snapshot and ``sim`` the
    full-range kc761sim simulation file (required; the composed
    response ``R = C p_tilde diag(eta)`` is built inside).  The working
    window of ``settings`` is padded by ``pad_nsigma`` resolution widths
    (clipped to the detector range); the pads participate in the solve
    but carry no regularization and are cropped from the output.  The
    regularization strength is the fixed ``settings.alpha``.
    """
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high

    pad_lo, pad_hi, win_lo, win_hi = _padding(calib, settings)
    calib_pad = slice_calibration(calib, pad_lo, pad_hi)
    sim_pad = slice_sim(sim, pad_lo, pad_hi)
    calib_win = slice_calibration(calib, ch_lo, ch_hi)

    y = np.asarray(data_counts[pad_lo:pad_hi + 1], dtype=float)
    unc = np.asarray(data_uncertainties[pad_lo:pad_hi + 1], dtype=float)

    sigma2 = np.maximum(unc ** 2, 1.0) + (settings.syst_frac * y) ** 2
    sigma = np.sqrt(sigma2)
    w = 1.0 / sigma2
    # The window padding participates in the response (as columns and
    # rows) but not in the chi-square: the pad bins' own data would
    # otherwise drag the window solution, because the pad energies cover
    # a region whose response and scale differ strongly from the
    # window's (fitting it collapsed the low-energy window solution).
    # Zero weights keep the pads as response columns/rows and as mu
    # carriers -- they are determined by the window data through the
    # response tails -- while removing their data constraints.  The
    # reported chi2/ndof therefore counts the window bins only.
    w[:win_lo] = 0.0
    w[win_hi + 1:] = 0.0

    to_channel, derivatives = to_channel_and_derivatives(
        calib_pad, sim_pad)

    # Regularization lives on the window only: the SNIP mask is computed
    # on the window data and embedded into the padded mask with zeros
    # (the pads carry no penalty).
    y_win = y[win_lo:win_hi + 1]
    sigma_win = sigma[win_lo:win_hi + 1]
    baseline = snip_baseline(y_win, settings.snip_iter)
    mask_win = peak_mask(y_win, sigma_win, baseline, settings.mask_z0,
                         settings.mask_floor)
    mask = np.zeros(calib_pad.n_bins)
    mask[win_lo:win_hi + 1] = mask_win

    d_op = penalty_operator(calib_pad.widths, calib_pad.centers,
                            sigma, mask, settings.k)
    # Weak zero prior on the pad mu: the pad bins carry no data and no
    # difference penalty, so their absolute level would otherwise be
    # numerically unconstrained (the difference operator's constant mode)
    # and any inverse-Hessian propagation would blow up.  The anchor rows
    # are constant, so their parameter derivatives are zero (handled in
    # kc761unfold.uncertainties._cross_gradient).
    pad_idx = np.concatenate([np.arange(win_lo),
                              np.arange(win_hi + 1, calib_pad.n_bins)])
    if pad_idx.size:
        anchor = sparse.csr_matrix(
            (np.full(pad_idx.size, np.sqrt(_PAD_ANCHOR)),
             (np.arange(pad_idx.size), pad_idx)),
            shape=(pad_idx.size, calib_pad.n_bins))
        d_op = sparse.vstack([d_op, anchor]).tocsr()

    prob = UnfoldProblem(to_channel, y, w, d_op, settings.alpha)
    mu = prob.solve()
    refolded_full = prob.r @ mu

    # One Hessian factorization serves the covariance solves and the
    # effective-degrees-of-freedom trace; the response derivatives come
    # from the single build above (they are large, never rebuilt).
    ab, u = prob.hessian_banded()
    factor = embed_and_factor(ab, u, prob.free)
    sig_stat, sig_syst = compute_covariances(
        prob, mu, prob.free, calib_pad, sim_pad, sigma, mask,
        settings.k, derivatives, ab=ab, u=u, factor=factor)

    # Effective residual degrees of freedom of the constrained fit.
    ndof = effective_ndof(prob, ab, u, prob.free,
                          int(np.sum(w > 0.0)), factor=factor)

    sl = slice(win_lo, win_hi + 1)
    data_total, data_syst, _, sigma_calib = _calibrated_layer(
        y[sl], unc[sl], calib_win, settings)

    return _result(
        False, calib_win, settings, mu[sl], y[sl], sigma[sl],
        sig_stat[sl], sig_syst[sl],
        np.sqrt(sig_stat[sl] ** 2 + sig_syst[sl] ** 2), sigma_calib,
        data_total, data_syst,
        refolded=refolded_full[sl],
        chi2=prob.chi2, ndof=ndof, pen_cost=prob.pen_cost,
        n_iter=prob.n_iter, converged=prob.converged)


def run_calib_only(calib: CalibrationFile, data_counts: np.ndarray,
                   data_uncertainties: np.ndarray, settings: UnfoldSettings
                   ) -> UnfoldResult:
    """Relabel the channel spectrum onto the energy axis (no unfolding).

    ``calib`` is the FULL-range calibration snapshot; the working window
    of ``settings`` is sliced out without padding.
    """
    ch_lo = settings.channel_low
    ch_hi = settings.channel_high
    calib_win = slice_calibration(calib, ch_lo, ch_hi)
    y = np.asarray(data_counts[ch_lo:ch_hi + 1], dtype=float)
    unc = np.asarray(data_uncertainties[ch_lo:ch_hi + 1], dtype=float)

    total, syst, stat, sigma_calib = _calibrated_layer(y, unc, calib_win,
                                                        settings)

    return _result(True, calib_win, settings, y, y, None, stat, syst, total,
                   sigma_calib, total, syst)
