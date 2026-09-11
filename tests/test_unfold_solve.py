"""Solver orchestration, pruning and strict bands (F-UNF-3/F-UNF-4/D-110)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kc761.core.binning import ChannelGrid
from kc761.core.response import slice_response
from kc761.core.solver import RegularizationSpec, solve_nonnegative
from kc761.errors import SolverError
from kc761.unfold import run_unfold
from kc761.unfold.compose import compose_from_products
from kc761.unfold.inputs import (
    covariance_matrix,
    internal_calibration,
    primary_edges_kev,
    resolution_params,
)
from kc761.unfold.selection import select_window
from kc761.unfold.solve import exact_zero_columns, solve_window
from tests.test_unfold_support import (
    N_CHANNELS,
    make_calib_product,
    make_sim_product,
    make_spectrum_product,
    response_of,
    truth_vector,
    write,
)

TRUTH_INDICES = (25, 30, 35, 40, 45, 50)
TRUTH_AMPLITUDES = (500.0, 800.0, 1200.0, 700.0, 400.0, 300.0)
WINDOW = (220.0, 520.0)
ALPHA = 1e-3


def _window(calib, sim, low: float, high: float):
    return select_window(
        calibration=internal_calibration(calib),
        resol_params=resolution_params(calib),
        channel_max=calib.channel_max,
        n_channels=N_CHANNELS,
        primary_edges_kev=primary_edges_kev(sim),
        energy_low_kev=low,
        energy_high_kev=high,
        pad_nsigma=5.0,
    )


def _outcome(calib, sim, values):
    selection = _window(calib, sim, *WINDOW)
    response, composed, _ = compose_from_products(calib, sim, strict=True)
    outcome = solve_window(
        composed=composed,
        response=response,
        selection=selection,
        data_values=np.asarray(values),
        data_variances=np.maximum(np.asarray(values), 1.0),
        deposition_counts=np.asarray(sim.primary_to_deposition.values),
        column_totals=np.asarray(sim.primary_column_totals.values),
        calibration=internal_calibration(calib),
        resol_params=resolution_params(calib),
        channel_grid=ChannelGrid(N_CHANNELS),
        channel_max=calib.channel_max,
        param_cov=covariance_matrix(calib),
        regularization=RegularizationSpec(alpha=ALPHA),
        syst_frac=0.1,
        strict=True,
    )
    return selection, composed, outcome


def test_exact_zero_columns_reduced_solve_is_bitwise_identical() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib, zero_columns=(0, 1, 88, 89))
    _, composed, _ = compose_from_products(calib, sim, strict=True)
    sliced = slice_response(composed, 17, 31).matrix.tocsr()
    kept, pruned = exact_zero_columns(sliced)
    assert np.array_equal(pruned, np.array([0, 1, 88, 89]))
    assert np.array_equal(kept, np.arange(kept[0], kept[-1] + 1))
    y = np.abs(np.arange(sliced.shape[0], dtype=np.float64)) + 5.0
    sigma = np.ones_like(y)
    spec = RegularizationSpec(alpha=ALPHA)
    full = solve_nonnegative(sliced, y, sigma, spec, strict=True).mu
    reduced = solve_nonnegative(sliced[:, kept], y, sigma, spec, strict=True).mu
    assert np.all(full[pruned] == 0.0)
    assert np.array_equal(full[kept], reduced)


def test_exactly_zero_pruning_never_uses_a_tolerance() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib, zero_columns=(0, 1, 88, 89))
    _, composed, _ = compose_from_products(calib, sim, strict=True)
    sliced = slice_response(composed, 17, 31).matrix.tolil()
    sliced[:, 5] = 1e-300
    kept, _ = exact_zero_columns(sliced.tocsr())
    assert 5 in {int(column) for column in kept}


def test_solve_window_matches_refolded_and_prunes_zeros() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib, zero_columns=(0, 1, 88, 89))
    truth = truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    response = response_of(calib, sim)
    signal = response @ truth
    selection, composed, outcome = _outcome(calib, sim, signal)
    assert outcome.certificate.ok
    assert np.all(outcome.mu_full >= 0.0)
    assert np.all(outcome.mu_full[outcome.pruned_columns] == 0.0)
    assert np.allclose(
        outcome.bands.sigma_total,
        np.hypot(outcome.bands.sigma_stat, outcome.bands.sigma_syst),
    )
    full = np.asarray(composed.matrix @ outcome.mu_full, dtype=np.float64)
    window = full[selection.channel_low : selection.channel_high + 1]
    dense = response @ outcome.mu_full
    assert np.allclose(window, dense[selection.channel_low : selection.channel_high + 1])


def test_closed_loop_recovers_lines_and_bands_are_finite() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    truth = truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    signal = response_of(calib, sim) @ truth
    data = make_spectrum_product(signal)
    result = run_unfold(
        data,
        calib,
        sim,
        snip_enabled=False, energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        strict=True,
        plot=False,
    )
    assert np.all(result.mu >= 0.0)
    assert np.all(np.isfinite(result.sigma_total.values))
    assert np.allclose(
        result.sigma_total.values,
        np.hypot(result.sigma_statistical.values, result.sigma_systematic.values),
    )
    low = result.window.report_low
    for index, amplitude in zip(TRUTH_INDICES, TRUTH_AMPLITUDES, strict=True):
        position = index - low
        error = max(result.sigma_total.values[position], 1.0)
        assert abs(result.mu[position] - amplitude) < 5.0 * error


@pytest.mark.slow
def test_pull_distribution_is_centered_and_covers() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    truth = truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    signal = response_of(calib, sim) @ truth
    pulls: list[float] = []
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        data = make_spectrum_product(rng.poisson(signal).astype(np.float64))
        result = run_unfold(
            data,
            calib,
            sim,
            snip_enabled=False, energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=ALPHA,
            strict=True,
            plot=False,
        )
        low = result.window.report_low
        for index in TRUTH_INDICES:
            position = index - low
            sigma = result.sigma_total.values[position]
            if sigma > 0.0:
                pulls.append((result.mu[position] - truth[index]) / sigma)
    values = np.asarray(pulls)
    assert abs(float(values.mean())) < 1.5
    assert 0.2 < float(values.std()) < 1.5
    assert float(np.mean(np.abs(values) < 3.0)) > 0.9


def test_single_bin_pruning_and_solve_is_well_defined() -> None:
    from scipy import sparse

    matrix = sparse.csr_matrix(np.array([[1.0]]))
    kept, pruned = exact_zero_columns(matrix)
    assert np.array_equal(kept, np.array([0]))
    assert pruned.size == 0
    solution = solve_nonnegative(
        matrix, np.array([2.0]), np.ones(1), RegularizationSpec(alpha=0.1), strict=True
    )
    assert solution.mu.shape == (1,)
    assert np.isfinite(solution.mu).all()
    assert solution.mu[0] >= 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_both_difference_orders_run(order: int) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    result = run_unfold(
        make_spectrum_product(signal),
        calib,
        sim,
        snip_enabled=False, energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        difference_order=order,
        strict=True,
        plot=False,
    )
    assert np.all(np.isfinite(result.mu))
    assert np.all(result.mu >= 0.0)


def test_extreme_alpha_is_finite_or_classified() -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    data = make_spectrum_product(signal)
    large = run_unfold(
        data,
        calib,
        sim,
        snip_enabled=False, energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=1e8,
        strict=True,
        plot=False,
    )
    assert np.all(np.isfinite(large.mu))
    try:
        tiny = run_unfold(
            data,
            calib,
            sim,
            snip_enabled=False, energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=1e-14,
            strict=True,
            plot=False,
        )
    except SolverError:
        pass
    else:
        assert np.all(np.isfinite(tiny.mu))


def test_solver_failure_leaves_no_product_or_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calib = make_calib_product()
    sim = make_sim_product(calib)
    truth = truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    data = make_spectrum_product(response_of(calib, sim) @ truth)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", data)
    output = tmp_path / "unfold.root"

    import kc761.unfold.solve as solve_module

    def _fail(*args, **kwargs):
        raise SolverError("forced failure")

    monkeypatch.setattr(solve_module, "solve_nonnegative", _fail)
    with pytest.raises(SolverError, match="forced failure"):
        run_unfold(
            data_path,
            calib_path,
            sim_path,
            snip_enabled=False, energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=ALPHA,
            output=output,
            strict=True,
            plot=False,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".part").exists()
