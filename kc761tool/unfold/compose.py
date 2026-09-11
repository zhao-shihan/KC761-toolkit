"""Compose orchestration: ``R = C . p_tilde . diag(eta)`` (F-RESP-2/F-RESP-3).

``run_compose`` is the entry for ``kc761tool compose``; it reads a calibration
product and a matrix-mode simulation product, checks their axes bitwise
(D-114), composes the full-primary response and writes the inspection artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from kc761tool.core.response import (
    ComposedResponse,
    ResponseMatrix,
    compose_response,
    verify_composed_columns,
    verify_response_columns,
)
from kc761tool.schema.io import build_provenance, validate_output_path, write_product
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    ComposeProduct,
    Histogram1D,
    Histogram2D,
    SimProduct,
)
from kc761tool.unfold.inputs import (
    check_deposition_axes,
    check_recorded_input,
    coerce_calib,
    coerce_sim,
)
from kc761tool.unfold.types import ComposeResult


def response_from_product(product: CalibProduct) -> ResponseMatrix:
    """Rebuild the sparse :class:`ResponseMatrix` from the stored dense C."""
    values = np.asarray(product.deposition_to_channel.values, dtype=np.float64)
    matrix = sparse.csr_matrix(values)
    column_sums = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
    return ResponseMatrix(
        matrix=matrix,
        column_sums=column_sums,
        deposition_edges_kev=np.asarray(
            product.deposition_to_channel.y.edges, dtype=np.float64
        ),
    )


def compose_from_products(
    calib: CalibProduct,
    sim: SimProduct,
    *,
    strict: bool = False,
) -> tuple[ResponseMatrix, ComposedResponse, Histogram1D]:
    """Compose the full-primary response and its derived efficiency."""
    check_deposition_axes(calib, sim)
    response = response_from_product(calib)
    verify_response_columns(response, strict=strict)
    counts = np.asarray(sim.primary_to_deposition.values, dtype=np.float64)
    totals = np.asarray(sim.primary_column_totals.values, dtype=np.float64)
    composed = compose_response(
        response,
        counts,
        totals,
        primary_edges_kev=np.asarray(sim.primary_to_deposition.y.edges, dtype=np.float64),
    )
    verify_composed_columns(composed, response, counts, totals, strict=strict)
    efficiency = Histogram1D(
        axis=sim.primary_to_deposition.y,
        values=np.asarray(composed.efficiency, dtype=np.float64),
        variances=None,
    )
    return response, composed, efficiency


def run_compose(
    calib: str | Path | CalibProduct,
    sim: str | Path | SimProduct,
    *,
    output: str | Path | None = None,
    force: bool = False,
    strict: bool = False,
    command: str = "kc761tool compose",
    arguments: Sequence[tuple[str, str]] = (),
    producer: str = "kc761tool-compose",
    extra_inputs: Sequence[str | Path] = (),
) -> ComposeResult:
    """Compose full-primary ``R`` and optionally write the compose artifact."""
    if output is not None:
        validate_output_path(output, force=force)
    calib_product, calib_path = coerce_calib(calib, strict=strict)
    sim_product, sim_path = coerce_sim(sim, strict=strict)
    check_recorded_input(sim_product, calib_path)

    response, composed, efficiency = compose_from_products(
        calib_product, sim_product, strict=strict
    )

    product: ComposeProduct | None = None
    product_path: Path | None = None
    if output is not None:
        provenance = build_provenance(
            producer=producer,
            command=command,
            arguments=tuple(arguments),
            inputs=[path for path in (calib_path, sim_path) if path is not None]
            + list(extra_inputs),
        )
        product = ComposeProduct(
            format_version=SCHEMA_VERSION,
            response_matrix=Histogram2D(
                x=calib_product.deposition_to_channel.x,
                y=sim_product.primary_to_deposition.y,
                values=np.asarray(composed.matrix.toarray(), dtype=np.float64),
                variances=None,
            ),
            deposition_to_channel=calib_product.deposition_to_channel,
            primary_to_deposition=sim_product.primary_to_deposition,
            primary_column_totals=sim_product.primary_column_totals,
            primary_efficiency=efficiency,
            provenance=provenance,
        )
        product_path = write_product(product, output, force=force, strict=strict)

    return ComposeResult(
        response=response,
        composed=composed,
        efficiency=efficiency,
        product=product,
        product_path=product_path,
        calib_path=calib_path,
        sim_path=sim_path,
    )


__all__ = ["compose_from_products", "response_from_product", "run_compose"]
