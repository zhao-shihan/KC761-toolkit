"""Strict-mode product certificates and always-on rejection tests.

The always-on checks (version, object set/type, axes, shape, finiteness, meta
fields) must reject malformed files in both modes. The product-level
certificates (F-RESP-1/F-RESP-2, F-COV-2, F-SIM-1/F-SIM-2, F-UNC-3, F-IO-1)
run only with ``strict=True``; the counterexamples below therefore write in
normal mode and assert a :class:`CertificateError` with the expected formula ID
under strict verification.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import uproot
from uproot.writing.identify import to_TAxis, to_TH2x

from kc761tool.core.model import PARAM_NAMES_REPORTED
from kc761tool.errors import CertificateError, SchemaError
from kc761tool.schema import _uproot, io
from kc761tool.schema.axes import Axis, channel_axis, energy_axis, reported_parameter_axis
from kc761tool.schema.products import (
    OBJ_DEPOSITION_TO_CHANNEL,
    OBJ_PARAM_COV,
    CalibProduct,
    Histogram1D,
    SpectrumProduct,
    dispatch_key_for,
    meta_field_types,
)
from tests.fixtures import synthetic

N_CHANNELS = synthetic.N_CHANNELS
N_PRIMARY = synthetic.N_PRIMARY

BUILDERS = (
    synthetic.make_calib_product,
    synthetic.make_sim_product,
    synthetic.make_compose_product,
    synthetic.make_unfold_product,
    synthetic.make_unfold_calib_only_product,
    synthetic.make_spectrum_product,
)


@pytest.mark.parametrize("build", BUILDERS)
def test_strict_certificates_pass(tmp_path: Path, build) -> None:
    product = build()
    path = io.write_product(product, tmp_path / "product.root", strict=True)
    io.verify_product(path, strict=True)


def _write_normal(product, path: Path) -> Path:
    """Write with always-on validation only; strict certificates are skipped."""
    return io.write_product(product, path, force=True)


def _expect_strict_failure(path: Path, formula_id: str, match: str = "") -> None:
    io.verify_product(path, strict=False)  # the normal mode must pass
    with pytest.raises(CertificateError) as info:
        io.verify_product(path, strict=True)
    assert info.value.formula_id == formula_id
    if match:
        assert match in str(info.value)


# --------------------------------------------------------------------------
# calib certificates
# --------------------------------------------------------------------------
def test_calib_column_sum_certificate(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    values = product.deposition_to_channel.values.copy()
    values[:, 0] *= 0.5
    bad = replace(
        product, deposition_to_channel=replace(product.deposition_to_channel, values=values)
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "c.root"), "F-RESP-1")


def test_calib_covariance_psd_certificate(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    covariance = np.diag([0.25, 1e-4, 1e-6, 1e-8, 0.04, 0.09, -0.16])
    bad = replace(product, param_cov=replace(product.param_cov, values=covariance))
    _expect_strict_failure(_write_normal(bad, tmp_path / "c.root"), "F-COV-2")


def test_calib_parameter_finiteness_certificate(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    bad = replace(product, params_reported=(math.nan, 2.5, 1e-5, 1e-9))
    _expect_strict_failure(_write_normal(bad, tmp_path / "c.root"), "F-MODEL-2")


def test_calib_parameter_labels_are_always_on(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    path = tmp_path / "c.root"
    _write_calib_raw(
        path,
        product,
        x_labels=("wrong",) * len(PARAM_NAMES_REPORTED),
    )
    with pytest.raises(SchemaError, match="param_cov bin labels"):
        io.verify_product(path, strict=False)
    with pytest.raises(SchemaError, match="param_cov bin labels"):
        io.verify_product(path, strict=True)


# --------------------------------------------------------------------------
# sim certificates
# --------------------------------------------------------------------------
def test_sim_count_bound_certificate(tmp_path: Path) -> None:
    product = synthetic.make_sim_product()
    values = product.primary_to_deposition.values.copy()
    values[0, 0] = product.primary_column_totals.values[0] + 5.0
    bad = replace(
        product, primary_to_deposition=replace(product.primary_to_deposition, values=values)
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "s.root"), "F-SIM-1")


def test_sim_variance_certificate(tmp_path: Path) -> None:
    product = synthetic.make_sim_product()
    variances = np.zeros_like(product.primary_to_deposition.variances)
    bad = replace(
        product,
        primary_to_deposition=replace(product.primary_to_deposition, variances=variances),
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "s.root"), "F-SIM-2")


# --------------------------------------------------------------------------
# compose certificates
# --------------------------------------------------------------------------
def test_compose_axis_certificate(tmp_path: Path) -> None:
    product = synthetic.make_compose_product()
    shifted = Axis(
        name="primary_energy_kev",
        edges=product.response_matrix.y.edges + 1.0,
        unit="kev",
    )
    bad = replace(product, response_matrix=replace(product.response_matrix, y=shifted))
    _expect_strict_failure(_write_normal(bad, tmp_path / "p.root"), "F-RESP-3")


def test_compose_column_sum_certificate(tmp_path: Path) -> None:
    product = synthetic.make_compose_product()
    values = product.response_matrix.values.copy()
    values[:, 0] *= 0.5
    bad = replace(product, response_matrix=replace(product.response_matrix, values=values))
    _expect_strict_failure(_write_normal(bad, tmp_path / "p.root"), "F-RESP-2")


# --------------------------------------------------------------------------
# unfold certificates
# --------------------------------------------------------------------------
def test_unfold_band_decomposition_certificate(tmp_path: Path) -> None:
    product = synthetic.make_unfold_product()
    assert product.sigma_statistical is not None
    broken = product.sigma_statistical.values.copy()
    bad = replace(
        product, sigma_total=replace(product.sigma_total, values=broken, variances=broken**2)
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "u.root"), "F-UNC-3")


def test_unfold_band_storage_certificate(tmp_path: Path) -> None:
    product = synthetic.make_unfold_product()
    assert product.sigma_statistical is not None
    bad = replace(
        product,
        sigma_statistical=replace(
            product.sigma_statistical,
            variances=np.ones_like(product.sigma_statistical.values),
        ),
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "u.root"), "F-IO-1")


def test_unfold_refolded_nonnegativity_certificate(tmp_path: Path) -> None:
    product = synthetic.make_unfold_product()
    assert product.refolded is not None
    bad = replace(
        product, refolded=replace(product.refolded, values=-product.refolded.values - 1.0)
    )
    _expect_strict_failure(_write_normal(bad, tmp_path / "u.root"), "F-RESP-2")


# --------------------------------------------------------------------------
# spectrum certificates
# --------------------------------------------------------------------------
def test_spectrum_daq_time_certificate(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    bad = replace(product, daq_time_s=0.0)
    _expect_strict_failure(_write_normal(bad, tmp_path / "x.root"), "F-IO-1")


def test_spectrum_negative_counts_are_allowed(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    subtracted = Histogram1D(
        axis=product.spectrum.axis,
        values=product.spectrum.values - 100.0,
        variances=product.spectrum.variances,
    )
    bad = SpectrumProduct(
        format_version=product.format_version,
        spectrum=subtracted,
        daq_time_s=product.daq_time_s,
        source_file=product.source_file,
        provenance=product.provenance,
    )
    path = io.write_product(bad, tmp_path / "net.root", strict=True)
    reloaded = io.read_product(path, strict=True)
    assert isinstance(reloaded, SpectrumProduct)
    assert np.any(reloaded.spectrum.values < 0.0)


# --------------------------------------------------------------------------
# raw-file helpers
# --------------------------------------------------------------------------
def _raw_axis(name: str, edges, unit: str):
    values = np.asarray(edges, dtype=np.float64)
    return to_TAxis(
        fName=name,
        # D-170: fTitle is display-only and never parsed, so raw fixtures can
        # carry a deliberately meaningless label.
        fTitle=f"display only ({unit})",
        fNbins=values.size - 1,
        fXmin=float(values[0]),
        fXmax=float(values[-1]),
        fXbins=values.astype(">f8"),
    )


def _raw_th2d(name: str, values: np.ndarray, x_axis, y_axis):
    values = np.asarray(values, dtype=np.float64)
    n_x, n_y = values.shape
    grid = np.zeros((n_x + 2, n_y + 2))
    grid[1:-1, 1:-1] = values
    data = grid.T.reshape(-1).astype(">f8")
    entries = float(values.sum())
    return to_TH2x(
        fName=None,
        fTitle="",
        data=data,
        fEntries=entries,
        fTsumw=entries,
        fTsumw2=entries,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=0.0,
        fTsumwy2=0.0,
        fTsumwxy=0.0,
        fSumw2=None,
        fXaxis=x_axis,
        fYaxis=y_axis,
    )


def _write_calib_raw(
    path: Path,
    product: CalibProduct,
    *,
    values: np.ndarray | None = None,
    x_edges=None,
    x_unit: str | None = None,
    x_name: str = "channel",
    x_labels=None,
    drop_param_cov: bool = False,
    extra_object: str | None = None,
    c_as_th1: bool = False,
    meta: dict[str, object] | None = None,
) -> None:
    c_values = product.deposition_to_channel.values if values is None else values
    y_axis = _raw_axis(
        product.deposition_to_channel.y.name,
        product.deposition_to_channel.y.edges,
        product.deposition_to_channel.y.unit,
    )
    with uproot.recreate(path) as file:
        if c_as_th1:
            _uproot.write_hist1d(
                file,
                OBJ_DEPOSITION_TO_CHANNEL,
                Histogram1D(axis=product.deposition_to_channel.x, values=c_values[:, 0]),
            )
        else:
            edges = (
                product.deposition_to_channel.x.edges if x_edges is None else x_edges
            )
            unit = x_unit or product.deposition_to_channel.x.unit
            file[OBJ_DEPOSITION_TO_CHANNEL] = _raw_th2d(
                OBJ_DEPOSITION_TO_CHANNEL,
                c_values,
                _raw_axis(x_name, edges, unit),
                y_axis,
            )
        if not drop_param_cov:
            _uproot.write_hist2d(
                file,
                OBJ_PARAM_COV,
                product.param_cov,
                x_labels=x_labels if x_labels is not None else PARAM_NAMES_REPORTED,
                y_labels=PARAM_NAMES_REPORTED,
            )
        if extra_object is not None:
            _uproot.write_hist1d(
                file,
                extra_object,
                Histogram1D(axis=channel_axis(3), values=np.zeros(3)),
            )
        _uproot.write_meta_tree(
            file, meta if meta is not None else io._encode_meta(product, "calib")
        )


# --------------------------------------------------------------------------
# always-on rejection (normal mode must already fail)
# --------------------------------------------------------------------------
def test_missing_object_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.root"
    _write_calib_raw(path, synthetic.make_calib_product(), drop_param_cov=True)
    with pytest.raises(SchemaError, match="object set mismatch"):
        io.read_product(path)


def test_extra_object_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.root"
    _write_calib_raw(path, synthetic.make_calib_product(), extra_object="extra_histogram")
    with pytest.raises(SchemaError, match="object set mismatch"):
        io.read_product(path)


def test_wrong_object_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.root"
    _write_calib_raw(path, synthetic.make_calib_product(), c_as_th1=True)
    with pytest.raises(SchemaError, match="expected th2"):
        io.read_product(path)


def test_wrong_format_version_is_rejected(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    meta = dict(io._encode_meta(product, "calib"))
    meta["format_version"] = 2
    path = tmp_path / "c.root"
    _write_calib_raw(path, product, meta=meta)
    with pytest.raises(SchemaError, match="format_version"):
        io.read_product(path)


def test_missing_meta_field_is_rejected(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    meta = dict(io._encode_meta(product, "calib"))
    del meta["producer"]
    path = tmp_path / "c.root"
    _write_calib_raw(path, product, meta=meta)
    with pytest.raises(SchemaError, match="meta field mismatch"):
        io.read_product(path)


def test_multi_entry_meta_field_is_rejected(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    duplicate = {
        key: np.array([value, value])
        for key, value in io._encode_meta(product, "calib").items()
    }
    path = tmp_path / "c.root"
    with uproot.recreate(path) as file:
        _uproot.write_hist2d(file, OBJ_DEPOSITION_TO_CHANNEL, product.deposition_to_channel)
        _uproot.write_hist2d(
            file,
            OBJ_PARAM_COV,
            product.param_cov,
            x_labels=PARAM_NAMES_REPORTED,
            y_labels=PARAM_NAMES_REPORTED,
        )
        file.mkrntuple("meta", duplicate, description="bad meta")
    with pytest.raises(SchemaError):
        io.read_product(path)


def test_duplicate_object_version_is_rejected(tmp_path: Path) -> None:
    product = synthetic.make_spectrum_product()
    path = io.write_product(product, tmp_path / "x.root")
    with uproot.update(path) as file:
        fields = {
            key: np.array([value])
            for key, value in io._encode_meta(product, "spectrum").items()
        }
        file.mkrntuple("meta", fields, description="duplicate")
    with pytest.raises(SchemaError, match="duplicate object"):
        io.read_product(path)


def test_unknown_axis_name_is_rejected(tmp_path: Path) -> None:
    """D-170: units come from the canonical fName, not the display title."""
    path = tmp_path / "c.root"
    _write_calib_raw(path, synthetic.make_calib_product(), x_name="parsec")
    with pytest.raises(SchemaError, match="unknown axis name"):
        io.read_product(path)


def test_axis_reversal_is_rejected(tmp_path: Path) -> None:
    product = synthetic.make_calib_product()
    path = tmp_path / "c.root"
    _write_calib_raw(
        path,
        product,
        x_edges=product.deposition_to_channel.x.edges[::-1],
    )
    with pytest.raises(SchemaError, match="strictly increasing"):
        io.read_product(path)


# --------------------------------------------------------------------------
# exactness / contract-shape checks
# --------------------------------------------------------------------------
def test_meta_field_types_are_frozen() -> None:
    sim_types = meta_field_types("sim")
    assert sim_types["mode"] is int
    assert sim_types["mode_name"] is str
    assert "calib_sha256" not in sim_types
    assert "sim_sha256" not in sim_types
    spectrum_types = meta_field_types("spectrum")
    assert spectrum_types["daq_time_s"] is float
    assert spectrum_types["source_file"] is str
    unfold_types = meta_field_types("unfold")
    assert unfold_types["mode"] is str
    assert unfold_types["dof"] is int
    assert meta_field_types("unfold_calib_only")["mode"] is str


def test_unfold_settings_must_be_complete(tmp_path: Path) -> None:
    product = synthetic.make_unfold_product()
    bad = replace(product, settings=product.settings[:-1])
    with pytest.raises(SchemaError):
        io.write_product(bad, tmp_path / "bad.root")


def test_unfold_calib_only_rejects_bands(tmp_path: Path) -> None:
    product = synthetic.make_unfold_calib_only_product()
    full = synthetic.make_unfold_product()
    bad = replace(product, sigma_total=full.sigma_total)
    with pytest.raises(SchemaError, match="must not carry bands"):
        io.write_product(bad, tmp_path / "bad.root")


def test_reported_parameter_axis_shape() -> None:
    axis = reported_parameter_axis()
    assert axis.n_bins == len(PARAM_NAMES_REPORTED)
    assert energy_axis(np.array([0.0, 1.0, 2.0])).unit == "kev"


def test_dispatch_roundtrip_for_all_kinds(tmp_path: Path) -> None:
    for build in BUILDERS:
        product = build()
        path = io.write_product(product, tmp_path / "x.root", force=True)
        assert dispatch_key_for(io.read_product(path)) == dispatch_key_for(product)


def test_sim_axes_match_fixture_shapes() -> None:
    product = synthetic.make_sim_product()
    assert product.primary_to_deposition.x.n_bins == N_CHANNELS
    assert product.primary_to_deposition.y.n_bins == N_PRIMARY
    assert product.primary_column_totals.axis.n_bins == N_PRIMARY
