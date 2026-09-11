"""Verified low-level uproot helpers.

Keep the head comment in sync with ``docs/formats.md`` section 6 (AGENTS.md).

Spike conclusions (uproot 5.7.6, validated by ``tests/test_uproot_spike.py``):

* Uproot's tuple histogram syntax cannot carry ``fSumw2``; variance requires
  the model constructors ``to_TH1x`` / ``to_TH2x`` with flow-padded arrays.
* ``fSumw2`` must follow the same flow layout as ``data``; for TH2D that is the
  transposed, flattened array.
* ``hist.errors()`` returns ``sqrt(fSumw2)`` when the buffer exists;
  ``len(hist.member("fSumw2")) > 0`` detects it.
* Scalar/string metadata uses a ``meta`` RNTuple (D-12 as revised), written
  explicitly with ``file.mkrntuple(...)``. Uproot's dict assignment already
  defaults to RNTuple, but the explicit call pins the type.
  ``TParameter``/``TNamed``/``TMatrixDSym`` have no writable uproot model.
* Axis bin labels *are* writable through ``to_THashList``/``to_TObjString``;
  ``param_cov`` uses them to carry ``c0 c1 c2 c3 b0 b1 b2`` (verified).
* Axis name and unit travel in the axis ``fTitle`` as ``"<name> [<unit>]"``;
  :func:`axis_from_hist` parses them back, so units are self-describing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import uproot
from numpy.typing import NDArray
from uproot.writing.identify import (
    to_TAxis,
    to_TH1x,
    to_TH2x,
    to_THashList,
    to_TObjString,
)

from kc761tool.errors import SchemaError, UnsupportedError
from kc761tool.schema.axes import Axis, human_axis_title, unit_for_axis_name
from kc761tool.schema.products import (
    HUMAN_TITLES,
    META_NTUPLE_NAME,
    Histogram1D,
    Histogram2D,
)

MIN_UPROOT = (5, 0)


def _check_uproot_version() -> None:
    parts = uproot.__version__.split(".")
    version = (int(parts[0]), int(parts[1]))
    if version < MIN_UPROOT:
        raise UnsupportedError(
            f"uproot >= {MIN_UPROOT[0]}.{MIN_UPROOT[1]} is required, found {uproot.__version__}"
        )


def histogram_has_variance(hist: Any) -> bool:
    """True when a ROOT histogram carries a non-empty ``fSumw2`` buffer."""
    return len(hist.member("fSumw2")) > 0


def _raw_variances_1d(hist: Any) -> NDArray[np.float64] | None:
    """Return the exact ``fSumw2`` buffer of a TH1D without flow bins."""
    if not histogram_has_variance(hist):
        return None
    raw = np.asarray(hist.member("fSumw2"), dtype=np.float64)
    return raw[1:-1].copy()


def _raw_variances_2d(hist: Any, shape: tuple[int, int]) -> NDArray[np.float64] | None:
    """Return the exact ``fSumw2`` buffer of a TH2D without flow bins."""
    if not histogram_has_variance(hist):
        return None
    raw = np.asarray(hist.member("fSumw2"), dtype=np.float64)
    n_x, n_y = shape
    # _flow2d stores ``out.T.reshape(-1)`` for an (n_x+2, n_y+2) flow grid.
    grid = raw.reshape(n_y + 2, n_x + 2).T
    return grid[1:-1, 1:-1].copy()


def _flow1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros(values.size + 2, dtype=">f8")
    out[1:-1] = values
    return out


def _flow2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros((values.shape[0] + 2, values.shape[1] + 2), dtype=">f8")
    out[1:-1, 1:-1] = values
    return out.T.reshape(-1)


def _bin_labels(labels: Sequence[str]) -> Any:
    """Build the writable ``THashList`` of bin labels used by ``fLabels``."""
    objects = [to_TObjString(str(label)) for label in labels]
    label_list = to_THashList(objects)
    # ROOT's TAxis::SetBinLabel sets TObject.fUniqueID to the 1-based bin index.
    for index, label in enumerate(label_list, start=1):
        label._bases[0]._members["@fUniqueID"] = index
    return label_list


def _to_axis(axis: Axis, *, labels: Sequence[str] | None = None) -> Any:
    edges = np.asarray(axis.edges, dtype=np.float64)
    return to_TAxis(
        fName=axis.name,
        fTitle=human_axis_title(axis),
        fNbins=edges.size - 1,
        fXmin=float(edges[0]),
        fXmax=float(edges[-1]),
        fXbins=edges.astype(">f8"),
        fLabels=_bin_labels(labels) if labels is not None else None,
    )


def axis_from_hist(hist: Any, index: int, *, fallback_name: str) -> Axis:
    """Read one axis back from a histogram (D-170).

    The canonical name travels in ``fName``; the unit is derived from that name
    through :data:`kc761tool.schema.axes.AXIS_UNITS`. ``fTitle`` is a human-readable
    display label only and is never parsed. A missing ``fName`` is a
    :class:`kc761tool.errors.SchemaError`, never a silent default.
    """
    axis = hist.axis(index)
    raw_name = axis.member("fName") if axis.has_member("fName") else None
    if not isinstance(raw_name, str) or not raw_name:
        raise SchemaError(f"axis {index} of {fallback_name!r}: missing axis name")
    unit = unit_for_axis_name(raw_name)
    edges = np.asarray(axis.edges(), dtype=np.float64)
    return Axis(name=raw_name, edges=edges, unit=unit)


def axis_labels(hist: Any, index: int) -> tuple[str, ...] | None:
    """Return the bin labels of one axis, or ``None`` when it has none."""
    axis = hist.axis(index)
    if not axis.has_member("fLabels"):
        return None
    labels = axis.member("fLabels")
    if labels is None:
        return None
    return tuple(str(label) for label in labels)


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
    title: str | None = None,
) -> None:
    """Write a TH1D, with ``fSumw2`` when variances are present."""
    _check_uproot_version()
    resolved_title = HUMAN_TITLES.get(name, "") if title is None else title
    values = _require_shape(hist.values, (hist.axis.n_bins,), f"{name}.values")
    variances = None
    if hist.variances is not None:
        variances = _require_shape(hist.variances, values.shape, f"{name}.variances")
    centers = hist.axis.centers()
    entries = float(values.sum())
    file[name] = to_TH1x(
        fName=None,
        fTitle=resolved_title,
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
    title: str | None = None,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
) -> None:
    """Write a TH2D, with ``fSumw2`` when variances are present.

    ``x_labels``/``y_labels`` attach TAxis bin labels (used by ``param_cov``).
    """
    _check_uproot_version()
    resolved_title = HUMAN_TITLES.get(name, "") if title is None else title
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
        fTitle=resolved_title,
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
        fXaxis=_to_axis(hist.x, labels=x_labels),
        fYaxis=_to_axis(hist.y, labels=y_labels),
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
    file.mkrntuple(META_NTUPLE_NAME, fields, description="kc761tool metadata")


def read_hist1d(
    file: Any,
    name: str,
    *,
    unit: str | None = None,
    axis_name: str | None = None,
) -> Histogram1D:
    """Read a TH1D (values, optional variances and edges) into a contract node.

    ``unit``/``axis_name`` override the self-described axis; when omitted they
    are parsed from the stored axis title and name.
    """
    hist = file[name]
    axis = axis_from_hist(hist, 0, fallback_name=axis_name or name)
    if axis_name is not None and axis_name != axis.name:
        raise SchemaError(f"{name}: stored axis name {axis.name!r} != {axis_name!r}")
    if unit is not None and unit != axis.unit:
        raise SchemaError(f"{name}: stored axis unit {axis.unit!r} != {unit!r}")
    values = np.asarray(hist.values(), dtype=np.float64)
    variances = _raw_variances_1d(hist)
    return Histogram1D(axis=axis, values=values, variances=variances)


def read_hist2d(
    file: Any,
    name: str,
    *,
    x_unit: str | None = None,
    y_unit: str | None = None,
    x_name: str | None = None,
    y_name: str | None = None,
) -> Histogram2D:
    """Read a TH2D (values, optional variances and both axes).

    Units and names are parsed from the stored axis titles unless overridden.
    """
    hist = file[name]
    x_axis = axis_from_hist(hist, 0, fallback_name=x_name or name)
    y_axis = axis_from_hist(hist, 1, fallback_name=y_name or name)
    if x_name is not None and x_name != x_axis.name:
        raise SchemaError(f"{name}: stored x-axis name {x_axis.name!r} != {x_name!r}")
    if y_name is not None and y_name != y_axis.name:
        raise SchemaError(f"{name}: stored y-axis name {y_axis.name!r} != {y_name!r}")
    if x_unit is not None and x_unit != x_axis.unit:
        raise SchemaError(f"{name}: stored x-axis unit {x_axis.unit!r} != {x_unit!r}")
    if y_unit is not None and y_unit != y_axis.unit:
        raise SchemaError(f"{name}: stored y-axis unit {y_axis.unit!r} != {y_unit!r}")
    values = np.asarray(hist.values(), dtype=np.float64)
    variances = _raw_variances_2d(hist, values.shape)
    return Histogram2D(
        x=x_axis,
        y=y_axis,
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
