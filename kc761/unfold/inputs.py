"""Product loading and cross-product axis/provenance validation.

All product access goes through :mod:`kc761.schema.io` (AGENTS hard rule 10).
The helpers here enforce the axis contract bitwise:

* ``C.y`` (deposition) equals ``G.x``;
* ``C.x`` equals the measured channel axis;
* ``N_j`` lives on ``G.y`` (the primary axis).

When an upstream product recorded the sha256 of one of our inputs, the digest
is re-checked against the file on disk (``fingerprint_for``/``input_sha256``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kc761.core.model import (
    InternalCalibration,
    ReportedCalibration,
    reported_to_internal,
)
from kc761.errors import ProvenanceError, SchemaError, ValidationError
from kc761.schema.io import input_sha256, read_product, sha256_file
from kc761.schema.products import (
    CalibProduct,
    Product,
    SimProduct,
    SpectrumProduct,
    dispatch_key_for,
)


def _load_path(path: str | Path, *, strict: bool, expected: str, cls: type) -> Product:
    product = read_product(path, strict=strict)
    if not isinstance(product, cls):
        raise SchemaError(
            f"{path}: expected a {expected} product, got {dispatch_key_for(product)!r}"
        )
    return product


def coerce_calib(
    value: str | Path | CalibProduct, *, strict: bool
) -> tuple[CalibProduct, Path | None]:
    """Return a calibration product and the path it came from (if any)."""
    if isinstance(value, CalibProduct):
        return value, None
    path = Path(value)
    product = _load_path(path, strict=strict, expected="calib", cls=CalibProduct)
    assert isinstance(product, CalibProduct)
    return product, path


def coerce_sim(
    value: str | Path | SimProduct, *, strict: bool
) -> tuple[SimProduct, Path | None]:
    """Return a simulation product and the path it came from (if any)."""
    if isinstance(value, SimProduct):
        return value, None
    path = Path(value)
    product = _load_path(path, strict=strict, expected="sim", cls=SimProduct)
    assert isinstance(product, SimProduct)
    return product, path


def coerce_spectrum(
    value: str | Path | SpectrumProduct, *, strict: bool
) -> tuple[SpectrumProduct, Path | None]:
    """Return a spectrum product and the path it came from (if any)."""
    if isinstance(value, SpectrumProduct):
        return value, None
    path = Path(value)
    product = _load_path(path, strict=strict, expected="spectrum", cls=SpectrumProduct)
    assert isinstance(product, SpectrumProduct)
    return product, path


def internal_calibration(product: CalibProduct) -> InternalCalibration:
    """Rebuild the internal basis ``(c0, k1, k2, k3)`` from the stored cubic."""
    params = np.asarray(product.params_reported, dtype=np.float64)
    reported = ReportedCalibration.from_array(params)
    return reported_to_internal(reported, channel_max=product.channel_max)


def resolution_params(product: CalibProduct) -> NDArray[np.float64]:
    """Return the three resolution parameters ``(b0, b1, b2)``."""
    return np.asarray(product.resol_params, dtype=np.float64)


def covariance_matrix(product: CalibProduct) -> NDArray[np.float64]:
    """Return the stored reported-basis 7x7 covariance (F-CAL-5)."""
    return np.asarray(product.param_cov.values, dtype=np.float64)


def primary_edges_kev(product: SimProduct) -> NDArray[np.float64]:
    """Return the simulation primary energy axis edges (``G.y``)."""
    return np.asarray(product.primary_to_deposition.y.edges, dtype=np.float64)


def check_deposition_axes(calib: CalibProduct, sim: SimProduct) -> None:
    """``C.y == G.x`` and ``N_j`` on ``G.y``, bitwise (D-114)."""
    c_axis = calib.deposition_to_channel.y
    g_axis = sim.primary_to_deposition.x
    if c_axis.unit != g_axis.unit:
        raise ValidationError(
            f"deposition axis unit mismatch: C.y is {c_axis.unit!r}, G.x is {g_axis.unit!r}"
        )
    c_edges = np.asarray(c_axis.edges, dtype=np.float64)
    g_edges = np.asarray(g_axis.edges, dtype=np.float64)
    if not np.array_equal(c_edges, g_edges):
        raise ValidationError(
            "deposition axis mismatch: the calibration C.y edges do not match "
            "the simulation G.x edges; the matrix-mode run used a different "
            "calibration"
        )
    primary = np.asarray(sim.primary_to_deposition.y.edges, dtype=np.float64)
    totals = np.asarray(sim.primary_column_totals.axis.edges, dtype=np.float64)
    if not np.array_equal(primary, totals):
        raise ValidationError(
            "primary axis mismatch: primary_column_totals does not use the G.y axis"
        )


def check_data_channel_axis(calib: CalibProduct, data: SpectrumProduct) -> None:
    """``C.x`` equals the measured channel axis, bitwise (D-114)."""
    c_axis = calib.deposition_to_channel.x
    d_axis = data.spectrum.axis
    if c_axis.unit != d_axis.unit:
        raise ValidationError(
            f"channel axis unit mismatch: C.x is {c_axis.unit!r}, data is {d_axis.unit!r}"
        )
    c_edges = np.asarray(c_axis.edges, dtype=np.float64)
    d_edges = np.asarray(d_axis.edges, dtype=np.float64)
    if not np.array_equal(c_edges, d_edges):
        raise ValidationError(
            "channel axis mismatch: the data spectrum axis does not match the "
            "calibration C.x channel axis"
        )


def check_recorded_input(product: Product, path: Path | None) -> None:
    """Verify a recorded input digest when the upstream product stored one.

    ``product`` is the upstream product whose provenance may reference
    ``path`` (for example a simulation generated against a calibration file).
    Matching is exact/resolved-path, never by basename (schema.io).
    """
    if path is None:
        return
    recorded = input_sha256(product.provenance, path)
    if recorded is None:
        return
    actual = sha256_file(path)
    if recorded != actual:
        raise ProvenanceError(
            f"input digest mismatch for {path}: the upstream {dispatch_key_for(product)!r} "
            f"product recorded {recorded}, the file is {actual}"
        )


__all__ = [
    "check_data_channel_axis",
    "check_deposition_axes",
    "check_recorded_input",
    "coerce_calib",
    "coerce_sim",
    "coerce_spectrum",
    "covariance_matrix",
    "internal_calibration",
    "primary_edges_kev",
    "resolution_params",
]
