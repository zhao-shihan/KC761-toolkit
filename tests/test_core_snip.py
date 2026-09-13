"""Unit tests for the SNIP peak mask (F-SOLVE-4/5/6, D-154..D-161)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from kc761tool.core.solver import (
    DEFAULT_SNIP_PROTECT_BINS,
    RegularizationSpec,
    SnipSettings,
    difference_operator,
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
    mask = snip_peak_mask(values, sigma, np.full(values.size, 1.0), np.ones(values.size), settings)
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
    solution = solve_nonnegative(response, values, sigma, spec, strict=True, mask=mask)
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


def _realistic_spectrum(
    resolution_bins: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat continuum with two peaks, at a production resolution/bin ratio.

    The repository fixtures use ``resolution/bin = 1``; a real detector has
    5-20 resolution widths per bin, which is the regime that made the former
    ``protect_sigma * sigma_E`` rule blanket the axis (D-188).
    """
    size = 512
    values = np.full(size, 400.0)
    grid = np.arange(size, dtype=np.float64)
    for center in (120, 300):
        values += 3000.0 * np.exp(-0.5 * ((grid - center) / resolution_bins) ** 2)
    values = np.maximum(values, 0.0)
    sigma = np.sqrt(np.maximum(values, 1.0))
    resolution_kev = np.full(size, resolution_bins)
    widths = np.ones(size)
    return values, sigma, resolution_kev, widths


def test_protection_stays_local_at_a_production_resolution_ratio() -> None:
    """D-188: the protected set is bounded by (2 * protect_bins + 1) per candidate."""
    values, sigma, resolution_kev, widths = _realistic_spectrum()
    settings = SnipSettings()
    mask = snip_peak_mask(values, sigma, resolution_kev, widths, settings)
    assert mask.n_candidates >= 1
    assert mask.n_protected <= (2 * settings.protect_bins + 1) * mask.n_candidates
    # The old resolution-scaled rule protected 47% of a real 2048-bin axis.
    assert mask.n_protected < values.size // 8


def test_masked_row_weight_is_the_floor_on_every_touched_row() -> None:
    """D-188: rho_r = min over the stencil, so the realized weight is `floor`."""
    from scipy import sparse

    size = 24
    floor = 0.25
    mask = np.ones(size)
    # Two *adjacent* protected bins: a stencil that contains both has
    # min = floor while the retired product rule gives floor**2, so this test
    # fails if the product rule is restored.
    mask[11] = mask[12] = floor
    protected = 12
    order = 2
    plain = difference_operator(size, order)
    rng = np.random.default_rng(11)
    response = sparse.csr_matrix(rng.normal(size=(size, size)) ** 2)
    values = np.full(size, 100.0)
    sigma = np.full(size, 10.0)
    spec = RegularizationSpec(alpha=1.0, difference_order=order)
    base, _, penalty_scale = normal_equations(response, values, sigma, spec)
    masked, _, _ = normal_equations(response, values, sigma, spec, mask=mask)
    penalty = (masked - base).toarray()
    rho = np.ones(plain.shape[0])
    for row in range(plain.shape[0]):
        rho[row] = mask[row : row + order + 1].min()
    # normal_equations returns H_plain + alpha * D'^T D' with D' = diag(sqrt(rho)) D
    # and D_tilde' = D' diag(penalty_scale) (F-SOLVE-1/F-SOLVE-6), so the masked
    # Hessian minus the plain one is the penalty difference with rho - 1.
    scaled = sparse.diags(penalty_scale) @ (plain.T @ sparse.diags(rho - 1.0) @ plain)
    expected = (scaled @ sparse.diags(penalty_scale)).toarray()
    assert np.allclose(penalty, expected)
    # Any row whose stencil touches the protected bin carries exactly the floor.
    assert np.any(rho == floor)
    assert np.allclose(rho, floor**2) is not True  # the product rule is excluded
    touching = [r for r in range(plain.shape[0]) if r <= protected <= r + order]
    assert touching
    assert np.allclose(rho[touching], floor)


def test_baseline_is_not_pulled_down_at_the_axis_ends() -> None:
    """F-SOLVE-4: out-of-range neighbors leave the boundary value untouched."""
    falling = np.array([100.0, 50.0, 50.0, 50.0, 50.0])
    rising = falling[::-1].copy()
    assert np.isclose(snip_baseline(falling, 3)[0], 100.0)
    assert np.isclose(snip_baseline(rising, 3)[-1], 100.0)
    assert np.isclose(snip_baseline(falling, 3)[1], 50.0)
    assert np.isclose(snip_baseline(rising, 3)[-2], 50.0)


def test_iteration_reference_index_selects_the_local_resolution() -> None:
    """D-157: the iteration count is derived at the caller's reference bin."""
    values, sigma, resolution_kev, widths = _realistic_spectrum(resolution_bins=2.0)
    resolution_kev = resolution_kev.copy()
    resolution_kev[:64] = 40.0  # a very wide detector at the low-energy end
    settings = SnipSettings(max_iterations=20)
    coarse = snip_peak_mask(
        values, sigma, resolution_kev, widths, settings, iteration_reference_index=0
    )
    fine = snip_peak_mask(
        values, sigma, resolution_kev, widths, settings, iteration_reference_index=400
    )
    assert coarse.iterations > fine.iterations
    assert coarse.iterations == settings.resolved_iterations(2.3548 * 40.0)
    assert fine.iterations == settings.resolved_iterations(2.3548 * 2.0)
    with pytest.raises(ValidationError, match="reference index"):
        snip_peak_mask(
            values, sigma, resolution_kev, widths, settings, iteration_reference_index=10**6
        )


def test_protect_bins_default_and_validation() -> None:
    assert SnipSettings().protect_bins == DEFAULT_SNIP_PROTECT_BINS
    with pytest.raises(ValidationError, match="protect_bins"):
        SnipSettings(protect_bins=0)
    with pytest.raises(ValidationError, match="protect_bins"):
        SnipSettings(protect_bins=2.5)


def test_library_snip_defaults_are_the_d191_constants() -> None:
    """D-191: the core defaults are the named constants, not inline literals."""
    from kc761tool.core.solver import DEFAULT_SNIP_FLOOR, DEFAULT_SNIP_MAX_ITERATIONS

    defaults = SnipSettings()
    assert defaults.floor == DEFAULT_SNIP_FLOOR
    assert defaults.max_iterations == DEFAULT_SNIP_MAX_ITERATIONS


def test_input_validation_rejects_unusable_arguments() -> None:
    """Malformed SNIP inputs fail as ValidationError, never as IndexError."""
    values, sigma, resolution_kev, widths = _realistic_spectrum()
    with pytest.raises(ValidationError, match="at least one bin"):
        snip_peak_mask(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), SnipSettings())
    for bad in (3.9, True, "3"):
        with pytest.raises(ValidationError, match="reference index"):
            snip_peak_mask(
                values,
                sigma,
                resolution_kev,
                widths,
                SnipSettings(),
                iteration_reference_index=bad,
            )
    with pytest.raises(ValidationError, match="reference index"):
        snip_peak_mask(
            values,
            sigma,
            resolution_kev,
            widths,
            SnipSettings(),
            iteration_reference_index=values.size,
        )
    # numpy integers are accepted, booleans are not silently read as 1.
    assert SnipSettings(protect_bins=np.int64(2)).protect_bins == 2
    with pytest.raises(ValidationError, match="protect_bins"):
        SnipSettings(protect_bins=True)
    with pytest.raises(ValidationError, match="max_iterations"):
        SnipSettings(max_iterations=np.float64(8.0))


def test_iteration_count_beyond_the_array_size_is_a_no_op() -> None:
    """The clipping loop stops when the interior is empty (bounded work)."""
    rng = np.random.default_rng(23)
    values = np.maximum(rng.normal(80.0, 25.0, size=32), 0.0)
    assert np.array_equal(snip_baseline(values, 10**6), snip_baseline(values, values.size))
    assert np.array_equal(
        snip_baseline(values, values.size), snip_baseline(values, values.size // 2 + 1)
    )
