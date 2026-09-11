"""Unit tests for the SNIP peak mask (F-SOLVE-4/5/6, D-154..D-161)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from kc761tool.core.solver import (
    RegularizationSpec,
    SnipSettings,
    normal_equations,
    snip_baseline,
    snip_peak_mask,
    solve_nonnegative,
    verify_snip_mask,
)
from kc761tool.errors import CertificateError, ValidationError


def _spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Flat continuum with one narrow significant peak."""
    values = np.full(64, 100.0)
    values[30:34] = (800.0, 2000.0, 2000.0, 800.0)
    sigma = np.sqrt(values)
    return values, sigma


def test_snip_baseline_stays_below_a_significant_peak() -> None:
    values, _ = _spectrum()
    baseline = snip_baseline(values, iterations=3)
    assert np.all(baseline >= 0.0)
    assert baseline[31] < values[31]
    # Away from the peak the baseline tracks the flat continuum.
    assert np.allclose(baseline[[0, 10, 50, 63]], 100.0, rtol=0.2)


def test_peak_mask_protects_the_peak_and_spares_flat_bins() -> None:
    values, sigma = _spectrum()
    settings = SnipSettings()
    mask = snip_peak_mask(
        values,
        sigma,
        np.full(values.size, 1.0),
        np.ones(values.size),
        settings,
    )
    assert mask.n_candidates >= 1
    assert np.any(mask.weights[30:34] < 1.0)
    assert np.all(mask.weights[:20] == 1.0)
    assert np.all((mask.weights >= settings.floor) & (mask.weights <= 1.0))
    assert mask.clipped_count == 0


def test_pure_noise_has_no_protected_bins() -> None:
    rng = np.random.default_rng(5)
    values = np.maximum(rng.normal(100.0, 10.0, size=128), 0.0)
    sigma = np.full(values.size, 10.0)
    mask = snip_peak_mask(
        values, sigma, np.full(values.size, 1.0), np.ones(values.size), SnipSettings()
    )
    # A 5-sigma matched filter on white noise must not protect the whole axis.
    assert mask.n_protected < values.size // 4


def test_certificate_detects_a_tampered_mask() -> None:
    values, sigma = _spectrum()
    settings = SnipSettings()
    mask = snip_peak_mask(
        values, sigma, np.full(values.size, 1.0), np.ones(values.size), settings
    )
    verify_snip_mask(
        settings,
        mask.weights,
        values=values,
        sigma=sigma,
        resolution_sigma_kev=np.full(values.size, 1.0),
        bin_width_kev=np.ones(values.size),
    )
    tampered = mask.weights.copy()
    tampered[0] = 0.5
    with pytest.raises(CertificateError, match="F-SOLVE-6"):
        verify_snip_mask(
            settings,
            tampered,
            values=values,
            sigma=sigma,
            resolution_sigma_kev=np.full(values.size, 1.0),
            bin_width_kev=np.ones(values.size),
        )


def test_masked_normal_equations_change_penalty_but_keep_kkt() -> None:
    values, sigma = _spectrum()
    mask = snip_peak_mask(
        values, sigma, np.full(values.size, 1.0), np.ones(values.size), SnipSettings()
    ).weights
    rng = np.random.default_rng(7)
    response = sparse.csr_matrix(rng.normal(size=(values.size, values.size)) ** 2)
    spec = RegularizationSpec(alpha=1e-2)
    plain = normal_equations(response, values, sigma, spec)
    masked = normal_equations(response, values, sigma, spec, mask=mask)
    assert not np.array_equal(plain[0].toarray(), masked[0].toarray())
    for hessian in (plain[0], masked[0]):
        dense = hessian.toarray()
        assert np.allclose(dense, dense.T)
        assert np.min(np.linalg.eigvalsh(dense)) > -1e-9
    solution = solve_nonnegative(
        response, values, sigma, spec, strict=True, mask=mask
    )
    assert solution.certificate.ok
    assert np.all(solution.mu >= 0.0)


def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        SnipSettings(threshold_sigma=0.0)
    with pytest.raises(ValidationError):
        SnipSettings(floor=1.5)
    with pytest.raises(ValidationError):
        SnipSettings(iterations=0)
    with pytest.raises(ValidationError):
        SnipSettings(max_iterations=0)
