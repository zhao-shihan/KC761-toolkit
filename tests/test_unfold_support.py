"""Deterministic fixtures for the unfold tests.

Small, fixed-seed inputs: a real calibration product on the channel-derived
deposition axis ``E(i +- 1/2)``, a matrix-mode simulation whose ``G.x`` equals
``C.y``, and a measured channel spectrum with ``fSumw2``. Everything is built
from ``numpy`` plus the public product/core interfaces; no Geant4 and no ROOT
executable are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kc761tool.calib.product import build_calib_product
from kc761tool.core.model import InternalCalibration
from kc761tool.core.response import compose_response
from kc761tool.schema.axes import channel_axis, energy_axis
from kc761tool.schema.io import write_product
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    Histogram1D,
    Histogram2D,
    Provenance,
    SimProduct,
    SpectrumProduct,
)
from kc761tool.unfold.compose import response_from_product

SEED = 20260910
N_CHANNELS = 32
CHANNEL_MAX = 31.0
CORE_INTERNAL = (-50.0, 10.0, 20.0, 30.0)
RESOL_PARAMS = (1.0, 8.0, 15.0)
PARAM_COV = np.diag([1.0, 1e-2, 1e-4, 1e-6, 0.04, 0.5, 1.0])
PRIMARY_MAX_KEV = 900.0
N_PRIMARY = 90
EVENTS_PER_COLUMN = 4000
DEPOSITION_WIDTH_KEV = 10.0
DEPOSITION_TAIL_SCALE_KEV = 150.0
DEPOSITION_TAIL_FRACTION = 0.05


def synthetic_provenance(producer: str = "unfold-test") -> Provenance:
    return Provenance(
        created_utc="1970-01-01T00:00:00Z",
        producer=producer,
        command="fixture",
        arguments=(),
        git_revision="0" * 7,
        git_dirty=False,
        python_version="3.12",
        dependency_versions=(),
        inputs=(),
    )


def primary_edges_kev() -> NDArray[np.float64]:
    return np.linspace(0.0, PRIMARY_MAX_KEV, N_PRIMARY + 1)


def primary_axis():
    return energy_axis(primary_edges_kev(), name="primary_energy_kev")


def make_calib_product(*, strict: bool = False) -> CalibProduct:
    """Calibration product with a real channel-derived deposition axis."""
    return build_calib_product(
        core_internal=np.asarray(CORE_INTERNAL, dtype=np.float64),
        resol_params=np.asarray(RESOL_PARAMS, dtype=np.float64),
        param_cov=PARAM_COV,
        chi2=10.0,
        dof=8,
        covariance_scale=1.0,
        fit_status="converged",
        scales=(),
        scale_bound_flags=(),
        n_channels=N_CHANNELS,
        channel_max=CHANNEL_MAX,
        provenance=synthetic_provenance("unfold-test-calib"),
        strict=strict,
    )


def calibration(product: CalibProduct) -> InternalCalibration:
    from kc761tool.core.model import ReportedCalibration, reported_to_internal

    return reported_to_internal(
        ReportedCalibration.from_array(np.asarray(product.params_reported)),
        channel_max=product.channel_max,
    )


def make_sim_product(
    calib: CalibProduct,
    *,
    zero_columns: tuple[int, ...] = (),
    events: int = EVENTS_PER_COLUMN,
    seed: int = SEED,
) -> SimProduct:
    """Matrix-mode counts with ``G.x == C.y``; listed columns are all-zero."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(calib.deposition_to_channel.y.edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_centers = 0.5 * (primary_edges_kev()[:-1] + primary_edges_kev()[1:])
    counts = np.zeros((centers.size, p_centers.size), dtype=np.float64)
    blocked = {int(index) for index in zero_columns}
    for column, energy in enumerate(p_centers):
        if column in blocked:
            continue
        peak = np.exp(-0.5 * ((centers - energy) / DEPOSITION_WIDTH_KEV) ** 2)
        tail = np.exp(-np.maximum(centers, 0.0) / DEPOSITION_TAIL_SCALE_KEV)
        probabilities = peak + DEPOSITION_TAIL_FRACTION * tail
        probabilities /= probabilities.sum()
        counts[:, column] = rng.multinomial(int(events), probabilities)
    totals = np.full(p_centers.size, float(events), dtype=np.float64)
    variances = counts * (1.0 - counts / totals[None, :])
    matrix = Histogram2D(
        x=energy_axis(edges, name="deposition_energy_kev"),
        y=primary_axis(),
        values=counts,
        variances=variances,
    )
    return SimProduct(
        format_version=SCHEMA_VERSION,
        primary_to_deposition=matrix,
        primary_column_totals=Histogram1D(axis=primary_axis(), values=totals),
        mode=0,
        mode_name="plane-front-gamma",
        geometry_name="plane",
        geometry_param_mm=13.7,
        angular_distribution="lambertian",
        seed=seed,
        n_events=int(totals.sum()),
        workers=1,
        provenance=synthetic_provenance("unfold-test-sim"),
    )


def response_of(calib: CalibProduct, sim: SimProduct) -> NDArray[np.float64]:
    """The composed full-primary response ``R = C G diag(1/N)``."""
    response = response_from_product(calib)
    composed = compose_response(
        response,
        np.asarray(sim.primary_to_deposition.values, dtype=np.float64),
        np.asarray(sim.primary_column_totals.values, dtype=np.float64),
        primary_edges_kev=np.asarray(sim.primary_to_deposition.y.edges, dtype=np.float64),
    )
    return np.asarray(composed.matrix.toarray(), dtype=np.float64)


def truth_vector(indices: tuple[int, ...], amplitudes: tuple[float, ...]) -> NDArray[np.float64]:
    truth = np.zeros(N_PRIMARY, dtype=np.float64)
    for index, amplitude in zip(indices, amplitudes, strict=True):
        truth[index] = amplitude
    return truth


def make_spectrum_product(
    values: NDArray[np.float64],
    variances: NDArray[np.float64] | None = None,
) -> SpectrumProduct:
    counts = np.asarray(values, dtype=np.float64)
    if variances is None:
        variances = np.maximum(counts, 1.0)
    return SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=channel_axis(N_CHANNELS),
            values=counts,
            variances=np.asarray(variances, dtype=np.float64),
        ),
        daq_time_s=100.0,
        source_file="unfold-test.csv",
        provenance=synthetic_provenance("unfold-test-spectrum"),
    )


def write(path: Path, product) -> Path:
    """Write a product strictly and return the path."""
    return write_product(product, path, strict=True)


__all__ = [
    "CHANNEL_MAX",
    "N_CHANNELS",
    "N_PRIMARY",
    "PARAM_COV",
    "PRIMARY_MAX_KEV",
    "calibration",
    "make_calib_product",
    "make_sim_product",
    "make_spectrum_product",
    "primary_axis",
    "primary_edges_kev",
    "response_of",
    "synthetic_provenance",
    "truth_vector",
    "write",
]
