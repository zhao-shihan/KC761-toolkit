"""W0 uproot spike tests: TH1D/TH2D variance and meta RNTuple round-trip.

These tests freeze the behaviour that ``docs/formats.md`` section 6 records:
uproot's tuple histogram syntax cannot carry ``fSumw2``, while the model
constructors used by ``kc761.schema._uproot`` can; metadata is written as an
RNTuple (D-12 as revised) via the explicit ``mkrntuple`` call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot

from kc761.schema import _uproot
from kc761.schema.axes import UNIT_KEV, energy_axis
from kc761.schema.products import Histogram1D, Histogram2D


def _energy_axis(start: float, stop: float, n_bins: int, name: str):
    return energy_axis(np.linspace(start, stop, n_bins + 1), name=name)


def test_th2d_variance_roundtrip(tmp_path: Path) -> None:
    x_axis = _energy_axis(0.0, 400.0, 4, "primary_energy_kev")
    y_axis = _energy_axis(0.0, 800.0, 5, "deposition_energy_kev")
    values = np.arange(20.0).reshape(4, 5)
    variances = values * 0.5 + 1.0
    matrix = Histogram2D(x=x_axis, y=y_axis, values=values, variances=variances)

    path = tmp_path / "spike.root"
    with uproot.recreate(path) as file:
        _uproot.write_hist2d(file, "matrix", matrix, title="spike matrix")
        band_values = np.sqrt(variances[:, 0])
        _uproot.write_hist1d(
            file,
            "band",
            Histogram1D(
                axis=x_axis, values=band_values, variances=band_values**2
            ),
        )
        _uproot.write_meta_tree(
            file,
            {"format_version": 1, "mode": "plane-front-gamma", "alpha": 0.01},
        )

    with uproot.open(path) as file:
        assert _uproot.histogram_has_variance(file["matrix"])
        back = _uproot.read_hist2d(
            file, "matrix", x_unit=UNIT_KEV, y_unit=UNIT_KEV
        )
        assert np.array_equal(back.x.edges, x_axis.edges)
        assert np.array_equal(back.y.edges, y_axis.edges)
        assert np.allclose(back.values, values)
        assert back.variances is not None
        assert np.allclose(back.variances, variances)
        assert np.allclose(file["matrix"].errors() ** 2, variances)

        band = _uproot.read_hist1d(file, "band", unit=UNIT_KEV)
        assert band.axis.n_bins == x_axis.n_bins
        assert np.allclose(band.values, band_values)
        assert band.variances is not None
        assert np.allclose(file["band"].errors(), band_values)

        meta = _uproot.read_meta_tree(file)
        assert meta["format_version"] == 1
        assert meta["mode"] == "plane-front-gamma"
        assert meta["alpha"] == pytest.approx(0.01)

        # D-12 as revised: the metadata object must be an RNTuple. Field
        # order is name-sorted, so compare as a set.
        meta_object = file["meta"]
        assert isinstance(meta_object, uproot.behaviors.RNTuple.RNTuple)
        assert set(meta_object.keys()) == {"format_version", "mode", "alpha"}


def test_hist1d_without_variance_has_no_buffer(tmp_path: Path) -> None:
    axis = _energy_axis(0.0, 100.0, 3, "energy_kev")
    values = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "counts.root"
    with uproot.recreate(path) as file:
        _uproot.write_hist1d(file, "counts", Histogram1D(axis=axis, values=values))
    with uproot.open(path) as file:
        assert not _uproot.histogram_has_variance(file["counts"])
        back = _uproot.read_hist1d(file, "counts", unit=UNIT_KEV)
        assert back.variances is None
        assert np.allclose(back.values, values)
