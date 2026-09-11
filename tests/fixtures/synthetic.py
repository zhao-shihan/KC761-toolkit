"""Deterministic small-scale synthetic inputs for certificate checks.

Everything here is fixed-seed and deliberately tiny (tens of bins, tens of
counts) so strict-mode certificates run in seconds. The fixtures build
contract nodes directly from numpy arrays and satisfy the product-level
certificates (compose is a true ``R = C . p_tilde . diag(eta)`` artifact and
the unfold bands obey ``total**2 = stat**2 + syst**2`` and D-15 storage).

Content:

* ``synthetic_response`` - deposition-to-channel matrix with normalized columns;
* ``synthetic_simulation`` - matrix-mode counts plus per-column totals whose
  binomial variances satisfy the F-SIM-2 relation;
* ``synthetic_spectrum`` - a small channel spectrum with variances;
* product builders for calib, sim, compose, unfold (full and calib-only) and
  spectrum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761.schema.axes import (
    Axis,
    channel_axis,
    energy_axis,
    reported_parameter_axis,
)
from kc761.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    ComposeProduct,
    Histogram1D,
    Histogram2D,
    InputFingerprint,
    Provenance,
    SimProduct,
    SpectrumProduct,
    UnfoldProduct,
)

SEED = 20260910
N_CHANNELS = 24
N_PRIMARY = 24
ENERGY_MAX_KEV = 2400.0
EVENTS_PER_COLUMN = 64


def default_channel_axis() -> Axis:
    return channel_axis(N_CHANNELS)


def default_primary_axis() -> Axis:
    edges = np.linspace(0.0, ENERGY_MAX_KEV, N_PRIMARY + 1)
    return energy_axis(edges, name="primary_energy_kev")


def default_deposition_axis() -> Axis:
    edges = np.linspace(0.0, ENERGY_MAX_KEV, N_CHANNELS + 1)
    return energy_axis(edges, name="deposition_energy_kev")


def synthetic_spectrum(seed: int = SEED) -> Histogram1D:
    """Small channel spectrum with fSumw2 = max(counts, 1)."""
    rng = np.random.default_rng(seed)
    values = rng.poisson(12.0, size=N_CHANNELS).astype(np.float64)
    return Histogram1D(
        axis=default_channel_axis(),
        values=values,
        variances=np.maximum(values, 1.0),
    )


def synthetic_unfolded_spectrum(seed: int = SEED) -> Histogram1D:
    """Small primary-energy spectrum with independent Poisson-like variances."""
    rng = np.random.default_rng(seed + 1)
    values = rng.poisson(20.0, size=N_PRIMARY).astype(np.float64)
    return Histogram1D(
        axis=default_primary_axis(),
        values=values,
        variances=np.maximum(values, 1.0),
    )


def synthetic_response() -> Histogram2D:
    """Deposition-to-channel matrix with Gaussian ridge and normalized columns."""
    channel = np.arange(N_CHANNELS)[:, None]
    deposition = np.arange(N_CHANNELS)[None, :]
    values = np.exp(-0.5 * ((channel - deposition) / 1.5) ** 2)
    values /= values.sum(axis=0, keepdims=True)
    return Histogram2D(
        x=default_channel_axis(),
        y=default_deposition_axis(),
        values=values,
        variances=np.zeros_like(values),
    )


def synthetic_simulation(seed: int = SEED) -> tuple[Histogram2D, Histogram1D]:
    """Matrix-mode counts and per-column totals with exact binomial variances."""
    rng = np.random.default_rng(seed)
    totals = np.full(N_PRIMARY, float(EVENTS_PER_COLUMN))
    counts = np.zeros((N_CHANNELS, N_PRIMARY), dtype=np.float64)
    for column in range(N_PRIMARY):
        probabilities = rng.dirichlet(np.full(N_CHANNELS, 0.4))
        counts[:, column] = rng.multinomial(int(totals[column]), probabilities)
    variances = counts * (1.0 - counts / totals[None, :])
    matrix = Histogram2D(
        x=default_deposition_axis(),
        y=default_primary_axis(),
        values=counts,
        variances=variances,
    )
    column_totals = Histogram1D(axis=default_primary_axis(), values=totals)
    return matrix, column_totals


def synthetic_provenance(producer: str = "synthetic-fixture") -> Provenance:
    return Provenance(
        created_utc="1970-01-01T00:00:00Z",
        producer=producer,
        command="fixture",
        arguments=(("seed", str(SEED)),),
        git_revision="0" * 7,
        git_dirty=False,
        python_version="3.12",
        dependency_versions=(),
        inputs=(InputFingerprint(path="synthetic", sha256="0" * 64),),
    )


def make_calib_product() -> CalibProduct:
    """Calibration product fixture (parameters are placeholders, not fits)."""
    covariance = np.diag([0.25, 1e-4, 1e-6, 1e-8, 0.04, 0.09, 0.16])
    return CalibProduct(
        format_version=SCHEMA_VERSION,
        deposition_to_channel=synthetic_response(),
        param_cov=Histogram2D(
            x=reported_parameter_axis(),
            y=reported_parameter_axis(),
            values=covariance,
            variances=np.zeros_like(covariance),
        ),
        params_reported=(100.0, 2.5, 1e-5, 1e-9),
        resol_params=(2.0, 12.0, 30.0),
        channel_max=float(N_CHANNELS - 1),
        provenance=synthetic_provenance("synthetic-calib"),
    )


def make_sim_product() -> SimProduct:
    matrix, totals = synthetic_simulation()
    return SimProduct(
        format_version=SCHEMA_VERSION,
        primary_to_deposition=matrix,
        primary_column_totals=totals,
        mode=0,
        mode_name="plane-front-gamma",
        geometry_name="plane",
        geometry_param_mm=13.7,
        angular_distribution="lambertian",
        seed=SEED,
        n_events=N_PRIMARY * EVENTS_PER_COLUMN,
        workers=1,
        provenance=synthetic_provenance("synthetic-sim"),
    )


def make_compose_product() -> ComposeProduct:
    """A true ``R = C . p_tilde . diag(eta)`` artifact (D-43/F-RESP-2)."""
    calib = make_calib_product()
    sim = make_sim_product()
    counts = sim.primary_to_deposition.values
    totals = sim.primary_column_totals.values
    detected = counts.sum(axis=0)
    efficiency = np.divide(detected, totals, out=np.zeros_like(totals), where=totals > 0.0)
    normalized = np.divide(
        counts, totals[None, :], out=np.zeros_like(counts), where=totals[None, :] > 0.0
    )
    response = calib.deposition_to_channel.values @ normalized
    return ComposeProduct(
        format_version=SCHEMA_VERSION,
        response_matrix=Histogram2D(
            x=default_channel_axis(),
            y=default_primary_axis(),
            values=response,
            variances=None,
        ),
        deposition_to_channel=calib.deposition_to_channel,
        primary_to_deposition=sim.primary_to_deposition,
        primary_column_totals=sim.primary_column_totals,
        primary_efficiency=Histogram1D(axis=default_primary_axis(), values=efficiency),
        provenance=synthetic_provenance("synthetic-compose"),
    )


def _unfold_settings() -> tuple[tuple[str, str], ...]:
    return (
        ("alpha", "0.01"),
        ("difference_order", "2"),
        ("energy_low_kev", "100.0"),
        ("energy_high_kev", "2400.0"),
        ("channel_low", "0"),
        ("channel_high", "23"),
        ("pad_nsigma", "5.0"),
        ("syst_frac", "0.1"),
        ("snip_enabled", "0"),
        ("snip_threshold_sigma", "5.0"),
        ("snip_protect_sigma", "2.0"),
        ("snip_floor", "0.1"),
        ("snip_iterations", "0"),
        ("snip_max_iterations", "8"),
        ("snip_clipped_bins", "0"),
        ("snip_clipped_index_range", "-1:-1"),
        ("snip_baseline_sha256", ""),
        ("snip_mask_sha256", ""),
        ("chi2", "12.5"),
        ("dof", "10"),
        ("covariance_scale", "1.25"),
    )


def make_unfold_product() -> UnfoldProduct:
    """Full unfolding product with a valid band decomposition (F-UNC-3/D-15)."""
    spectrum = synthetic_unfolded_spectrum()
    sigma_stat = 0.8 * np.sqrt(np.maximum(spectrum.values, 1.0))
    sigma_syst = 0.6 * np.sqrt(np.maximum(spectrum.values, 1.0))
    sigma_total = np.hypot(sigma_stat, sigma_syst)
    refolded = synthetic_spectrum()
    return UnfoldProduct(
        format_version=SCHEMA_VERSION,
        mode="unfold",
        spectrum=spectrum,
        sigma_statistical=Histogram1D(
            axis=spectrum.axis, values=sigma_stat, variances=sigma_stat**2
        ),
        sigma_systematic=Histogram1D(
            axis=spectrum.axis, values=sigma_syst, variances=sigma_syst**2
        ),
        sigma_total=Histogram1D(
            axis=spectrum.axis, values=sigma_total, variances=sigma_total**2
        ),
        refolded=Histogram1D(
            axis=refolded.axis, values=refolded.values, variances=refolded.variances
        ),
        settings=_unfold_settings(),
        provenance=synthetic_provenance("synthetic-unfold"),
    )


def make_unfold_calib_only_product() -> UnfoldProduct:
    """Calibration-only relabeling: one spectrum, no bands, no settings."""
    return UnfoldProduct(
        format_version=SCHEMA_VERSION,
        mode="calib_only",
        spectrum=synthetic_spectrum(seed=SEED + 2),
        sigma_statistical=None,
        sigma_systematic=None,
        sigma_total=None,
        refolded=None,
        settings=(),
        provenance=synthetic_provenance("synthetic-unfold-calib-only"),
    )


def make_spectrum_product() -> SpectrumProduct:
    return SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=synthetic_spectrum(seed=SEED + 3),
        daq_time_s=123.0,
        source_file="synthetic.csv",
        provenance=synthetic_provenance("synthetic-spectrum"),
    )


def expected_binomial_variances(counts: NDArray[np.float64], totals: NDArray[np.float64]):
    """Reference F-SIM-2 variance used by fixture self-checks."""
    return counts * (1.0 - counts / totals[None, :])
