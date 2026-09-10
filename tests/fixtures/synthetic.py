"""Deterministic small-scale synthetic inputs for certificate checks.

Everything here is fixed-seed and deliberately tiny (tens of bins, tens of
counts) so strict-mode certificates run in seconds. The fixtures build
contract nodes directly from numpy arrays; they do not call ``kc761.core``
functions (stubs until W1) and therefore stay usable across all workstreams.

Content:

* ``synthetic_response`` - deposition-to-channel matrix with normalized columns;
* ``synthetic_simulation`` - matrix-mode counts plus per-column totals whose
  binomial variances satisfy the F-SIM-2 relation;
* ``synthetic_spectrum`` - a small channel spectrum with variances;
* product builders for calib, sim, compose and unfold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kc761.schema.axes import Axis, channel_axis, energy_axis
from kc761.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    ComposeProduct,
    Histogram1D,
    Histogram2D,
    InputFingerprint,
    Provenance,
    SimProduct,
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
            x=Axis(
                name="reported_parameter",
                edges=np.arange(-0.5, 7.5, 1.0),
                unit="dimensionless",
            ),
            y=Axis(
                name="reported_parameter",
                edges=np.arange(-0.5, 7.5, 1.0),
                unit="dimensionless",
            ),
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
        mode="plane-front-gamma",
        seed=SEED,
        n_events=N_PRIMARY * EVENTS_PER_COLUMN,
        workers=1,
        provenance=synthetic_provenance("synthetic-sim"),
    )


def make_compose_product() -> ComposeProduct:
    calib = make_calib_product()
    sim = make_sim_product()
    efficiency = Histogram1D(
        axis=default_primary_axis(),
        values=sim.primary_to_deposition.values.sum(axis=0) / sim.primary_column_totals.values,
    )
    return ComposeProduct(
        format_version=SCHEMA_VERSION,
        response_matrix=synthetic_response(),
        deposition_to_channel=calib.deposition_to_channel,
        primary_to_deposition=sim.primary_to_deposition,
        primary_column_totals=sim.primary_column_totals,
        primary_efficiency=efficiency,
        provenance=synthetic_provenance("synthetic-compose"),
    )


def make_unfold_product() -> UnfoldProduct:
    spectrum = synthetic_spectrum()
    scale = 1.1
    sigma = scale * np.sqrt(np.maximum(spectrum.values, 1.0))
    return UnfoldProduct(
        format_version=SCHEMA_VERSION,
        mode="unfold",
        spectrum=spectrum,
        sigma_statistical=Histogram1D(
            axis=spectrum.axis, values=sigma, variances=sigma**2
        ),
        sigma_systematic=Histogram1D(
            axis=spectrum.axis, values=sigma, variances=sigma**2
        ),
        sigma_total=Histogram1D(
            axis=spectrum.axis, values=np.hypot(sigma, sigma), variances=2.0 * sigma**2
        ),
        refolded=Histogram1D(axis=spectrum.axis, values=spectrum.values.copy()),
        settings=(("alpha", "0.01"), ("difference_order", "2")),
        provenance=synthetic_provenance("synthetic-unfold"),
    )


def expected_binomial_variances(counts: NDArray[np.float64], totals: NDArray[np.float64]):
    """Reference F-SIM-2 variance used by fixture self-checks."""
    return counts * (1.0 - counts / totals[None, :])
