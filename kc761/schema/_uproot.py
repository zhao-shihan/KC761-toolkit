"""Verified low-level uproot helpers (W0 spike).

Keep the head comment in sync with ``docs/formats.md`` section 6 (AGENTS.md).

Spike conclusions (uproot 5.7.6, validated by ``tests/test_uproot_spike.py``):

* Uproot's tuple histogram syntax cannot carry ``fSumw2``; variance requires
  the model constructors ``to_TH1x`` / ``to_TH2x`` with flow-padded arrays.
* ``fSumw2`` must follow the same flow layout as ``data``; for TH2D that is the
  transposed, flattened array.
* ``hist.errors()`` returns ``sqrt(fSumw2)`` when the buffer exists;
  ``len(hist.member("fSumw2")) > 0`` detects it.
* Scalar/string metadata uses a ``meta`` RNTuple (D-12 as revised
  2026-09-10), written explicitly with ``file.mkrntuple(...)``. Uproot's dict
  assignment already defaults to RNTuple, but the explicit call pins the type.
  ``TParameter``/``TNamed``/``TMatrixDSym`` have no writable uproot model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import uproot
from numpy.typing import NDArray
from uproot.writing.identify import to_TAxis, to_TH1x, to_TH2x

from kc761.errors import SchemaError, UnsupportedError
from kc761.schema.axes import Axis
from kc761.schema.products import META_NTUPLE_NAME, Histogram1D, Histogram2D

MIN_UPROOT = (5, 0)


def _check_uproot_version() -> None:
    parts = uproot.__version__.split(".")
    version = (int(parts[0]), int(parts[1]))
    if version < MIN_UPROOT:
        raise UnsupportedError(
            f"uproot >= {MIN_UPROOT[0]}.{MIN_UPROOT[1]} is required, "
            f"found {uproot.__version__}"
        )


def histogram_has_variance(hist: Any) -> bool:
    """True when a ROOT histogram carries a non-empty ``fSumw2`` buffer."""
    return len(hist.member("fSumw2")) > 0


def _flow1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros(values.size + 2, dtype=">f8")
    out[1:-1] = values
    return out


def _flow2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros((values.shape[0] + 2, values.shape[1] + 2), dtype=">f8")
    out[1:-1, 1:-1] = values
    return out.T.reshape(-1)


def _to_axis(axis: Axis) -> Any:
    edges = np.asarray(axis.edges, dtype=np.float64)
    return to_TAxis(
        fName=axis.name,
        fTitle=f"{axis.name} [{axis.unit}]",
        fNbins=edges.size - 1,
        fXmin=float(edges[0]),
        fXmax=float(edges[-1]),
        fXbins=edges.astype(">f8"),
    )


def _require_shape(values: Any, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise SchemaError(f"{label}: expected shape {shape}, got {array.shape}")
    return array


def write_hist1d(
    file: Any,
    name: str,
    hist: Histogram1D,
    *,
    title: str = "",
) -> None:
    """Write a TH1D, with ``fSumw2`` when variances are present."""
    _check_uproot_version()
    values = _require_shape(hist.values, (hist.axis.n_bins,), f"{name}.values")
    variances = None
    if hist.variances is not None:
        variances = _require_shape(hist.variances, values.shape, f"{name}.variances")
    centers = hist.axis.centers()
    entries = float(values.sum())
    file[name] = to_TH1x(
        fName=None,
        fTitle=title,
        data=_flow1d(values),
        fEntries=entries,
        fTsumw=entries,
        fTsumw2=float(variances.sum()) if variances is not None else entries,
        fTsumwx=float(values @ centers),
        fTsumwx2=float(values @ (centers**2)),
        fSumw2=_flow1d(variances).astype(">f8") if variances is not None else None,
        fXaxis=_to_axis(hist.axis),
    )


def write_hist2d(
    file: Any,
    name: str,
    hist: Histogram2D,
    *,
    title: str = "",
) -> None:
    """Write a TH2D, with ``fSumw2`` when variances are present."""
    _check_uproot_version()
    shape = (hist.x.n_bins, hist.y.n_bins)
    values = _require_shape(hist.values, shape, f"{name}.values")
    variances = None
    if hist.variances is not None:
        variances = _require_shape(hist.variances, shape, f"{name}.variances")
    x_centers = hist.x.centers()
    y_centers = hist.y.centers()
    entries = float(values.sum())
    file[name] = to_TH2x(
        fName=None,
        fTitle=title,
        data=_flow2d(values),
        fEntries=entries,
        fTsumw=entries,
        fTsumw2=float(variances.sum()) if variances is not None else entries,
        fTsumwx=float(values.sum(axis=1) @ x_centers),
        fTsumwx2=float(values.sum(axis=1) @ (x_centers**2)),
        fTsumwy=float(values.sum(axis=0) @ y_centers),
        fTsumwy2=float(values.sum(axis=0) @ (y_centers**2)),
        fTsumwxy=float(x_centers @ (values @ y_centers)),
        fSumw2=_flow2d(variances).astype(">f8") if variances is not None else None,
        fXaxis=_to_axis(hist.x),
        fYaxis=_to_axis(hist.y),
    )


def write_meta_tree(
    file: Any,
    scalars: Mapping[str, float | int | str] | None = None,
) -> None:
    """Write the ``meta`` RNTuple from length-1 scalar/string fields.

    The explicit ``mkrntuple`` call pins the object type (D-12 as revised):
    ``file[name] = dict`` also creates an RNTuple in uproot 5.7.6, but the
    contract must not follow a library default that may change.
    """
    fields: dict[str, NDArray[Any]] = {}
    for key, value in (scalars or {}).items():
        fields[key] = np.array([value])
    file.mkrntuple(META_NTUPLE_NAME, fields, description="kc761 metadata")


def read_hist1d(
    file: Any,
    name: str,
    *,
    unit: str,
    axis_name: str | None = None,
) -> Histogram1D:
    """Read a TH1D (values, optional variances and edges) into a contract node."""
    hist = file[name]
    values = np.asarray(hist.values(), dtype=np.float64)
    edges = np.asarray(hist.axis(0).edges(), dtype=np.float64)
    variances = (
        np.asarray(hist.errors(), dtype=np.float64) ** 2
        if histogram_has_variance(hist)
        else None
    )
    axis = Axis(name=axis_name or name, edges=edges, unit=unit)
    return Histogram1D(axis=axis, values=values, variances=variances)


def read_hist2d(
    file: Any,
    name: str,
    *,
    x_unit: str,
    y_unit: str,
    x_name: str | None = None,
    y_name: str | None = None,
) -> Histogram2D:
    """Read a TH2D (values, optional variances and both axes)."""
    hist = file[name]
    values = np.asarray(hist.values(), dtype=np.float64)
    x_edges = np.asarray(hist.axis(0).edges(), dtype=np.float64)
    y_edges = np.asarray(hist.axis(1).edges(), dtype=np.float64)
    variances = (
        np.asarray(hist.errors(), dtype=np.float64) ** 2
        if histogram_has_variance(hist)
        else None
    )
    return Histogram2D(
        x=Axis(name=x_name or name, edges=x_edges, unit=x_unit),
        y=Axis(name=y_name or name, edges=y_edges, unit=y_unit),
        values=values,
        variances=variances,
    )


def read_meta_tree(file: Any) -> dict[str, Any]:
    """Read the ``meta`` RNTuple into a flat dict of Python scalars."""
    tree = file[META_NTUPLE_NAME]
    out: dict[str, Any] = {}
    # RNTuple iteration yields RField objects and keys() is name-sorted;
    # always address meta fields by name.
    field_names = list(tree.keys())
    for key in field_names:
        array = np.asarray(tree[key].array())
        if array.size == 1:
            item = array.reshape(-1)[0]
            out[key] = item.item() if isinstance(item, np.generic) else item
        else:  # pragma: no cover - contract forbids multi-entry meta fields
            out[key] = array
    return out
