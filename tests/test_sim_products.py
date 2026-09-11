"""Product tests: mc_spectrum/SimProduct round-trips and physical certificates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kc761tool import errors
from kc761tool.core import binning
from kc761tool.schema import io
from kc761tool.schema.axes import energy_axis
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    Histogram1D,
    InputFingerprint,
    McSpectrumProduct,
    Provenance,
)
from kc761tool.sim import certificates
from tests.fixtures import synthetic


def _provenance() -> Provenance:
    return Provenance(
        created_utc="1970-01-01T00:00:00Z",
        producer="sim-test",
        command="test",
        arguments=(("seed", "1"),),
        git_revision="0" * 7,
        git_dirty=False,
        python_version="3.12",
        dependency_versions=(),
        inputs=(InputFingerprint(path="synthetic", sha256="0" * 64),),
    )


def _mc_spectrum_product() -> McSpectrumProduct:
    edges = binning.source_mode_deposition_edges_kev()
    values = np.zeros(edges.size - 1, dtype=np.float64)
    values[:8] = [10.0, 4.0, 0.0, 1.0, 6.0, 0.0, 0.0, 2.0]
    return McSpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=energy_axis(edges, name="energy_kev"),
            values=values,
            variances=certificates.source_spectrum_variance(values),
        ),
        source_key="am241",
        mode_name="source_decay",
        geometry_name="sandwich",
        geometry_param_mm=1.0,
        n_events=23,
        seed=7,
        workers=1,
        provenance=_provenance(),
    )


def test_mc_spectrum_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    product = _mc_spectrum_product()
    path = io.write_product(product, tmp_path / "mc.root", strict=True)
    back = io.read_product(path, strict=True)
    assert isinstance(back, McSpectrumProduct)
    assert back.source_key == product.source_key
    assert back.geometry_name == product.geometry_name
    assert back.n_events == product.n_events
    assert back.workers == product.workers
    assert np.array_equal(back.spectrum.values, product.spectrum.values)
    assert np.array_equal(back.spectrum.variances, product.spectrum.variances)


def test_mc_spectrum_requires_the_fixed_source_mode_variance(tmp_path: Path) -> None:
    product = _mc_spectrum_product()
    bad = Histogram1D(
        axis=product.spectrum.axis,
        values=product.spectrum.values,
        variances=np.ones_like(product.spectrum.values),
    )
    with pytest.raises(errors.CertificateError, match="F-SIM-2"):
        io.write_product(
            replace(product, spectrum=bad),
            tmp_path / "bad.root",
            strict=True,
        )


def test_mc_spectrum_axis_must_be_energy(tmp_path: Path) -> None:
    from kc761tool.schema.axes import channel_axis

    product = _mc_spectrum_product()
    bad = replace(
        product,
        spectrum=Histogram1D(
            axis=channel_axis(product.spectrum.axis.n_bins),
            values=product.spectrum.values,
            variances=product.spectrum.variances,
        ),
    )
    with pytest.raises(errors.SchemaError, match="energy axis unit"):
        io.write_product(bad, tmp_path / "bad.root", strict=True)


def test_mc_spectrum_axis_must_be_the_canonical_source_mode_axis(tmp_path: Path) -> None:
    """D-148-adjacent D6: the fixed source-mode axis is always-on enforced."""
    product = _mc_spectrum_product()
    bad = replace(
        product,
        spectrum=Histogram1D(
            axis=energy_axis(np.linspace(0.0, 4000.0, product.spectrum.axis.n_bins + 1)),
            values=product.spectrum.values,
            variances=product.spectrum.variances,
        ),
    )
    with pytest.raises(errors.SchemaError, match="fixed source-mode axis"):
        io.write_product(bad, tmp_path / "bad.root")


def test_sim_product_strict_roundtrip_carries_exact_variance(tmp_path: Path) -> None:
    product = synthetic.make_sim_product()
    path = io.write_product(product, tmp_path / "sim.root", strict=True)
    back = io.read_product(path, strict=True)
    assert back.primary_to_deposition.variances is not None
    assert np.allclose(
        back.primary_to_deposition.variances,
        synthetic.expected_binomial_variances(
            back.primary_to_deposition.values, back.primary_column_totals.values
        ),
    )


def test_binomial_variance_matches_definition() -> None:
    totals = np.array([4.0, 0.0, 10.0])
    counts = np.array([[1.0, 0.0, 7.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    variance = certificates.binomial_variance(counts, totals)
    with np.errstate(invalid="ignore"):
        expected = counts * (1.0 - counts / totals[None, :])
    expected[:, 1] = 0.0
    assert np.array_equal(variance, expected)


def test_event_accounting_positive_and_negative() -> None:
    counts = np.array([[3.0, 0.0], [1.0, 2.0]])
    zero = np.array([1.0, 0.0])
    totals = counts.sum(axis=0) + zero
    certificates.verify_event_accounting(counts, totals, zero, 7)
    with pytest.raises(errors.CertificateError, match="sum\\(N_j\\)"):
        certificates.verify_event_accounting(counts, totals, zero, 8)
    lost = totals.copy()
    lost[0] += 1.0
    with pytest.raises(errors.CertificateError, match="events lost"):
        certificates.verify_event_accounting(counts, lost, zero, 7)


def test_physical_boundary_certificate_is_strict_only() -> None:
    deposition_edges = np.array([0.0, 100.0, 200.0])
    primary_edges = np.array([0.0, 50.0, 150.0])
    # deposition bin [100, 200) above primary bin [0, 50): forbidden.
    counts = np.array([[0.0, 1.0], [1.0, 0.0]])
    certificates.verify_physical_boundary(
        counts, deposition_edges, primary_edges, strict=False
    )
    with pytest.raises(errors.CertificateError, match="F-SIM-6"):
        certificates.verify_physical_boundary(
            counts, deposition_edges, primary_edges, strict=True
        )
    clean = np.array([[0.0, 1.0], [0.0, 0.0]])
    certificates.verify_physical_boundary(
        clean, deposition_edges, primary_edges, strict=True
    )


def test_source_spectrum_variance_positive_and_negative() -> None:
    values = np.array([5.0, 0.0, 5.0])
    variance = certificates.source_spectrum_variance(values)
    assert np.allclose(variance, values * (1.0 - values / values.sum()))
    certificates.verify_source_spectrum_variance(values, variance)
    with pytest.raises(errors.CertificateError, match="F-SIM-2"):
        certificates.verify_source_spectrum_variance(values, variance + 1.0)
    empty = np.zeros(4)
    assert np.array_equal(certificates.source_spectrum_variance(empty), empty)


def test_efficiency_certificate() -> None:
    counts = np.array([[1.0, 0.0], [3.0, 2.0]])
    totals = np.array([4.0, 2.0])
    certificates.verify_efficiency(counts, totals)
    with pytest.raises(errors.CertificateError, match="F-SIM-3"):
        certificates.verify_efficiency(counts, np.array([1.0, 2.0]))


def test_default_primary_axis_matches_core_binning() -> None:
    assert binning.SOURCE_MODE_DEPOSITION_BINS == 4096
    assert binning.source_mode_deposition_edges_kev().size == 4097
