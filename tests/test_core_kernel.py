"""Auxiliary checks for the response kernel (F-KERN-1..4)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from kc761tool.core import kernel
from kc761tool.core._gen import kernel_expr
from kc761tool.errors import CertificateError


def test_gaussian_bin_probabilities_match_cdf_differences() -> None:
    edges = np.linspace(0.0, 10.0, 11)
    centers = np.array([2.5, 5.0, 7.5])
    sigma = np.array([0.5, 1.0, 1.5])
    values = kernel.gaussian_bin_probabilities(edges, centers, sigma)
    assert values.shape == (10, 3)
    reference = ndtr((edges[1:, None] - centers[None, :]) / sigma[None, :]) - ndtr(
        (edges[:-1, None] - centers[None, :]) / sigma[None, :]
    )
    assert np.allclose(values, reference, rtol=1e-12, atol=1e-15)
    assert np.all(values >= 0.0)
    assert np.all(values.sum(axis=0) <= 1.0 + 1e-12)


def test_taper_plateau_and_support() -> None:
    sigma = np.array([2.0])
    offsets = np.array([0.0, 1.0, 3.9, 4.0, 5.0, 6.5])
    weights = kernel.support_taper(offsets, sigma, n_sigma=3.0)
    assert np.allclose(weights[:4], 1.0)  # plateau |x| <= (n-1) sigma
    assert weights[5] == 0.0  # beyond n sigma
    assert 0.0 < weights[4] < 1.0
    # continuity at both boundaries
    epsilon = 1e-9
    for boundary in ((3.0 - 1.0) * 2.0, 3.0 * 2.0):
        before = kernel.support_taper(np.array([boundary - epsilon]), sigma, n_sigma=3.0)
        after = kernel.support_taper(np.array([boundary + epsilon]), sigma, n_sigma=3.0)
        assert abs(float(before[0]) - float(after[0])) < 1e-7


def test_taper_gradient_matches_finite_difference() -> None:
    offsets = np.array([-4.5, -2.0, -0.3, 1.2, 3.4, 4.8])
    sigma = np.array([1.5])
    weights, d_offset, d_sigma = kernel.taper_grad(offsets, sigma, n_sigma=3.0)
    step = 1e-7
    finite_offset = (
        kernel.support_taper(offsets + step, sigma, n_sigma=3.0)
        - kernel.support_taper(offsets - step, sigma, n_sigma=3.0)
    ) / (2.0 * step)
    finite_sigma = (
        kernel.support_taper(offsets, sigma + step, n_sigma=3.0)
        - kernel.support_taper(offsets, sigma - step, n_sigma=3.0)
    ) / (2.0 * step)
    assert np.allclose(weights, kernel.support_taper(offsets, sigma, n_sigma=3.0))
    # At the support boundary the second derivative jumps; the central
    # difference carries an O(step) term, hence the looser absolute tolerance.
    assert np.allclose(d_offset, finite_offset, atol=1e-6)
    assert np.allclose(d_sigma, finite_sigma, atol=1e-6)


def test_local_kernel_derivatives_match_finite_difference() -> None:
    e_lo = np.array([1.0, 2.5, 6.0])
    e_hi = e_lo + 1.0
    center = 0.5 * (e_lo + e_hi)
    source = 3.7
    sigma = 1.2
    n_sigma = 3.0
    values, d_lo, d_hi, d_c, d_sigma, d_center = kernel_expr.tapered_bin_grad(
        e_lo, e_hi, source, sigma, center, n_sigma
    )
    step = 1e-7
    reference = kernel_expr.tapered_bin(e_lo, e_hi, source, sigma, center, n_sigma)
    assert np.allclose(values, reference, rtol=1e-12)
    # tapered_bin_grad holds the center fixed; the kernel assembly adds the
    # center shift (0.5 * d_center for either edge) before using the result.
    finite_lo_fixed = (
        kernel_expr.tapered_bin(e_lo + step, e_hi, source, sigma, center, n_sigma)
        - kernel_expr.tapered_bin(e_lo - step, e_hi, source, sigma, center, n_sigma)
    ) / (2.0 * step)
    finite_hi_fixed = (
        kernel_expr.tapered_bin(e_lo, e_hi + step, source, sigma, center, n_sigma)
        - kernel_expr.tapered_bin(e_lo, e_hi - step, source, sigma, center, n_sigma)
    ) / (2.0 * step)
    finite_lo_moving = (
        kernel_expr.tapered_bin(
            e_lo + step, e_hi, source, sigma, center + 0.5 * step, n_sigma
        )
        - kernel_expr.tapered_bin(
            e_lo - step, e_hi, source, sigma, center - 0.5 * step, n_sigma
        )
    ) / (2.0 * step)
    finite_hi_moving = (
        kernel_expr.tapered_bin(
            e_lo, e_hi + step, source, sigma, center + 0.5 * step, n_sigma
        )
        - kernel_expr.tapered_bin(
            e_lo, e_hi - step, source, sigma, center - 0.5 * step, n_sigma
        )
    ) / (2.0 * step)
    finite_sigma = (
        kernel_expr.tapered_bin(e_lo, e_hi, source, sigma + step, center, n_sigma)
        - kernel_expr.tapered_bin(e_lo, e_hi, source, sigma - step, center, n_sigma)
    ) / (2.0 * step)
    finite_source = (
        kernel_expr.tapered_bin(e_lo, e_hi, source + step, sigma, center, n_sigma)
        - kernel_expr.tapered_bin(e_lo, e_hi, source - step, sigma, center, n_sigma)
    ) / (2.0 * step)
    assert np.allclose(d_lo, finite_lo_fixed, atol=1e-8)
    assert np.allclose(d_hi, finite_hi_fixed, atol=1e-8)
    assert np.allclose(d_lo + 0.5 * d_center, finite_lo_moving, atol=1e-8)
    assert np.allclose(d_hi + 0.5 * d_center, finite_hi_moving, atol=1e-8)
    assert np.allclose(d_sigma, finite_sigma, atol=1e-8)
    assert np.allclose(d_c, finite_source, atol=1e-8)


def test_response_triples_column_sums() -> None:
    edges = np.linspace(0.0, 40.0, 41)
    centers = np.linspace(1.0, 39.0, 21)
    sigma = np.full_like(centers, 1.5)
    triples = kernel.response_triples(edges, centers, sigma, n_sigma=3.0)
    sums = kernel.verify_column_sums(triples, edges.size - 1, strict=True)
    assert np.allclose(sums[: centers.size], 1.0, atol=1e-12)
    assert np.all(sums[centers.size :] == 0.0)


def test_pruned_pattern_equals_dense_evaluation() -> None:
    edges = np.linspace(0.0, 20.0, 21)
    centers = np.array([2.0, 9.9, 17.5])
    sigma = np.array([0.8, 1.5, 2.0])
    triples = kernel.response_triples(edges, centers, sigma, n_sigma=3.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    for column, (center, width) in enumerate(zip(centers, sigma, strict=True)):
        dense = kernel_expr.tapered_bin(
            edges[:-1], edges[1:], float(center), float(width), bin_centers, 3.0
        )
        denominator = dense.sum()
        selected = triples.cols == column
        rows = triples.rows[selected]
        expected = dense[rows] / denominator
        expected = expected[expected > 0.0]
        assert np.allclose(triples.values[selected], expected, rtol=1e-12, atol=1e-15)


def test_empty_columns_and_certificate() -> None:
    edges = np.linspace(0.0, 10.0, 11)
    centers = np.array([50.0])
    sigma = np.array([1.0])
    triples = kernel.response_triples(edges, centers, sigma, n_sigma=3.0)
    assert triples.values.size == 0
    sums = kernel.verify_column_sums(triples, edges.size - 1, strict=True)
    assert sums[0] == 0.0
    broken = kernel.SparseTriples(
        rows=np.array([0]), cols=np.array([0]), values=np.array([0.5])
    )
    with pytest.raises(CertificateError, match="F-KERN-2"):
        kernel.verify_column_sums(broken, 4, strict=True)
