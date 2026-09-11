"""Product IO contract: atomic write, reopen validation and overwrite policy.

Formula ID F-IO-1 (docs/derivations.md). Frozen protocol (docs/formats.md):

1. refuse an existing target unless ``force`` is set (D-17);
2. write to ``<target>.part``;
3. close, reopen and validate every object and the ``meta`` tree;
4. rename onto the target (atomic on the same filesystem) (D-16);
5. on any failure remove the partial file and raise.

All product reading and writing goes through this module (AGENTS.md rule 10);
callers must not open product files themselves.

Validation is two-tiered (D-61/D-62): schema/version/axes/units/shape/finiteness
checks always run, while the product-level certificates (F-RESP-1/F-RESP-2,
F-COV-2, F-SIM-1..3, F-UNC-3, F-IO-1) run only with ``strict=True``. Every
raise is a :class:`kc761.errors.Kc761Error` subclass with an actionable message.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import uproot
from scipy import sparse
from uproot.behaviors.RNTuple import RNTuple
from uproot.behaviors.TH1 import TH1
from uproot.behaviors.TH2 import TH2

from kc761.core.binning import source_mode_deposition_edges_kev
from kc761.core.covariance import CovarianceEstimate, verify_covariance_psd
from kc761.core.model import PARAM_NAMES_REPORTED
from kc761.core.response import (
    ComposedResponse,
    ResponseMatrix,
    verify_composed_columns,
    verify_response_columns,
)
from kc761.core.uncertainty import UncertaintyBands, verify_band_decomposition
from kc761.errors import (
    CertificateError,
    Kc761Error,
    ProvenanceError,
    SchemaError,
    UsageError,
    ValidationError,
)
from kc761.schema import _uproot
from kc761.schema.axes import UNIT_KEV, Axis, check_same_edges
from kc761.schema.products import (
    META_ANGULAR_DISTRIBUTION,
    META_ARGUMENTS_JSON,
    META_CHANNEL_MAX,
    META_CHI2,
    META_COMMAND,
    META_COVARIANCE_SCALE,
    META_CREATED_UTC,
    META_DAQ_TIME_S,
    META_DEPENDENCY_VERSIONS,
    META_DOF,
    META_FIT_STATUS,
    META_FORMAT_VERSION,
    META_GEOMETRY_NAME,
    META_GEOMETRY_PARAM_MM,
    META_GIT_DIRTY,
    META_GIT_REVISION,
    META_INPUTS_JSON,
    META_MODE,
    META_MODE_NAME,
    META_N_EVENTS,
    META_NTUPLE_NAME,
    META_PARAMS_REPORTED_JSON,
    META_PRODUCER,
    META_PRODUCT_KIND,
    META_PYTHON_VERSION,
    META_RESOL_CLAMP_COUNT,
    META_RESOL_CLAMP_ENERGY_HIGH_KEV,
    META_RESOL_CLAMP_ENERGY_LOW_KEV,
    META_RESOL_PARAMS_JSON,
    META_SCALE_BOUND_FLAGS_JSON,
    META_SCALES_JSON,
    META_SEED,
    META_SOURCE_FILE,
    META_SOURCE_KEY,
    META_WORKERS,
    OBJ_DEPOSITION_TO_CHANNEL,
    OBJ_MC_SPECTRUM,
    OBJ_PARAM_COV,
    OBJ_PRIMARY_COLUMN_TOTALS,
    OBJ_PRIMARY_EFFICIENCY,
    OBJ_PRIMARY_TO_DEPOSITION,
    OBJ_RESPONSE_MATRIX,
    OBJ_SIGMA_STATISTICAL,
    OBJ_SIGMA_SYSTEMATIC,
    OBJ_SIGMA_TOTAL,
    OBJ_SPECTRUM,
    OBJ_SPECTRUM_CALIBRATED,
    OBJ_SPECTRUM_REFOLDED,
    OBJ_SPECTRUM_UNFOLDED,
    OBJECT_NAMES,
    SCHEMA_VERSION,
    TRACKED_DEPENDENCIES,
    UNFOLD_MODE_CALIB_ONLY,
    UNFOLD_SETTING_TYPES,
    CalibProduct,
    ComposeProduct,
    Histogram1D,
    Histogram2D,
    InputFingerprint,
    McSpectrumProduct,
    Product,
    Provenance,
    SimProduct,
    SpectrumProduct,
    UnfoldProduct,
    dispatch_key_for,
    dispatch_key_for_meta,
    meta_field_types,
    product_kind_for,
)

PART_SUFFIX = ".part"
"""Suffix of the in-progress file; a stale one is overwritten (F-IO-1)."""

_CERT_RTOL = 1e-9
"""Relative tolerance of the product-level certificates."""

_SHA256 = re.compile(r"[0-9a-f]{64}")

_OBJECT_KIND: Mapping[str, str] = {
    OBJ_DEPOSITION_TO_CHANNEL: "th2",
    OBJ_PARAM_COV: "th2",
    OBJ_PRIMARY_TO_DEPOSITION: "th2",
    OBJ_PRIMARY_COLUMN_TOTALS: "th1",
    OBJ_RESPONSE_MATRIX: "th2",
    OBJ_PRIMARY_EFFICIENCY: "th1",
    OBJ_SPECTRUM_UNFOLDED: "th1",
    OBJ_SIGMA_STATISTICAL: "th1",
    OBJ_SIGMA_SYSTEMATIC: "th1",
    OBJ_SIGMA_TOTAL: "th1",
    OBJ_SPECTRUM_REFOLDED: "th1",
    OBJ_SPECTRUM_CALIBRATED: "th1",
    OBJ_SPECTRUM: "th1",
    OBJ_MC_SPECTRUM: "th1",
    META_NTUPLE_NAME: "rntuple",
}

_REQUIRED_VARIANCE: Mapping[str, frozenset[str]] = {
    "calib": frozenset(),
    "sim": frozenset({OBJ_PRIMARY_TO_DEPOSITION}),
    "compose": frozenset(),
    "unfold": frozenset(
        {OBJ_SIGMA_STATISTICAL, OBJ_SIGMA_SYSTEMATIC, OBJ_SIGMA_TOTAL}
    ),
    "unfold_calib_only": frozenset(),
    "spectrum": frozenset({OBJ_SPECTRUM}),
    "mc_spectrum": frozenset({OBJ_MC_SPECTRUM}),
}


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(text: str, field: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"meta field {field!r} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of one input file (D-18)."""
    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"input file not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    """Return ``(revision, dirty)``; fall back to ``("unknown", False)``."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        warnings.warn(
            "git metadata unavailable; recording git_revision='unknown', "
            "git_dirty=0 (D-93)",
            RuntimeWarning,
            stacklevel=3,
        )
        return "unknown", False
    return revision or "unknown", bool(status.strip())


def _dependency_versions(extra: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Resolve installed versions in the single-source ordered dependency list."""
    names = list(TRACKED_DEPENDENCIES)
    names.extend(name for name in extra if name not in TRACKED_DEPENDENCIES)
    versions: list[tuple[str, str]] = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        versions.append((name, version))
    return tuple(versions)


def build_provenance(
    *,
    producer: str,
    command: str,
    arguments: Sequence[tuple[str, str]],
    inputs: Sequence[str | Path],
    extra_dependencies: Sequence[str] = (),
) -> Provenance:
    """Assemble the complete provenance record for one run (F-IO-1/D-18).

    ``arguments`` is serialized in the given order; ``inputs`` are hashed and
    stored as :class:`InputFingerprint` records. A missing git repository is a
    warning, not an error: ``git_revision='unknown'`` and ``git_dirty=0`` are
    recorded instead (D-93).
    """
    if not producer:
        raise ProvenanceError("producer must be a non-empty string")
    revision, dirty = _git_state()
    fingerprints = tuple(
        InputFingerprint(path=str(Path(path)), sha256=sha256_file(path))
        for path in inputs
    )
    return Provenance(
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        producer=producer,
        command=command,
        arguments=tuple((str(key), str(value)) for key, value in arguments),
        git_revision=revision,
        git_dirty=dirty,
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(extra_dependencies),
        inputs=fingerprints,
    )


def fingerprint_for(
    provenance: Provenance, path: str | Path
) -> InputFingerprint | None:
    """Return the recorded fingerprint for an input path, if any.

    Matching is by exact string, by the given path string, or by resolved
    absolute path; there is no basename fallback because that could match two
    different inputs.
    """
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    for fingerprint in provenance.inputs:
        if fingerprint.path == str(path) or fingerprint.path == str(candidate):
            return fingerprint
        try:
            if Path(fingerprint.path).resolve() == resolved:
                return fingerprint
        except OSError:
            continue
    return None


def input_sha256(provenance: Provenance, path: str | Path) -> str | None:
    """Convenience wrapper around :func:`fingerprint_for` for downstream checks."""
    fingerprint = fingerprint_for(provenance, path)
    return None if fingerprint is None else fingerprint.sha256


# --------------------------------------------------------------------------
# Overwrite policy and atomic write
# --------------------------------------------------------------------------
def refuse_overwrite(path: str | Path, *, force: bool = False) -> None:
    """Raise when ``path`` exists and ``force`` is not set (F-IO-1/D-17)."""
    target = Path(path)
    if target.exists() and not force:
        raise SchemaError(
            f"refusing to overwrite existing {target}; pass --force to replace it"
        )
    if target.is_dir():
        raise SchemaError(f"target {target} is a directory, not a product file")


def validate_output_path(path: str | Path, *, force: bool = False) -> None:
    """Fail fast on an unusable output target before an expensive run (D-168).

    Applies the same overwrite policy as :func:`refuse_overwrite` but raises
    :class:`kc761.errors.UsageError` (CLI exit code 2, "pass --force or fix the
    path"), and additionally creates the parent directory and checks that it is
    writable. ``write_product`` still re-checks at write time as the last line
    of defence.
    """
    try:
        refuse_overwrite(path, force=force)
    except SchemaError as exc:
        raise UsageError(str(exc)) from exc
    target = Path(path)
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(f"cannot create output directory {parent}: {exc}") from exc
    if not os.access(parent, os.W_OK):
        raise UsageError(f"output directory is not writable: {parent}")


def atomic_write(
    target: str | Path,
    writer: Callable[[Path], None],
    *,
    verify: Callable[[Path], None],
) -> Path:
    """Write, verify and atomically move a product into place (F-IO-1/D-16).

    ``writer`` receives the ``.part`` path; ``verify`` reopens it and checks
    the full contract. A stale ``.part`` is removed before writing. On any
    failure the partial file is removed and an existing target is left intact.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + PART_SUFFIX)
    if part.exists():
        if part.is_dir():
            raise SchemaError(f"stale partial path {part} is a directory")
        part.unlink()
    try:
        writer(part)
        verify(part)
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------
# Meta encoding / decoding
# --------------------------------------------------------------------------
def _encode_common_meta(product: Product) -> dict[str, float | int | str]:
    provenance = product.provenance
    return {
        META_FORMAT_VERSION: int(product.format_version),
        META_PRODUCT_KIND: product_kind_for(dispatch_key_for(product)),
        META_PRODUCER: provenance.producer,
        META_CREATED_UTC: provenance.created_utc,
        META_GIT_REVISION: provenance.git_revision,
        META_GIT_DIRTY: 1 if provenance.git_dirty else 0,
        META_PYTHON_VERSION: provenance.python_version,
        META_DEPENDENCY_VERSIONS: _json_dumps(
            [[name, version] for name, version in provenance.dependency_versions]
        ),
        META_COMMAND: provenance.command,
        META_ARGUMENTS_JSON: _json_dumps(
            [[name, value] for name, value in provenance.arguments]
        ),
        META_INPUTS_JSON: _json_dumps(
            [{"path": fp.path, "sha256": fp.sha256} for fp in provenance.inputs]
        ),
    }


def _coerce_setting(field: str, type_: type, raw: str) -> float | int | str:
    text = str(raw).strip()
    if type_ is str:
        return str(raw)
    try:
        if type_ is int:
            return int(text)
        if type_ is float:
            value = float(text)
            if not np.isfinite(value):
                raise ValidationError(f"unfold setting {field!r} must be finite, got {raw!r}")
            return value
    except ValueError as exc:
        raise SchemaError(
            f"unfold setting {field!r} is not a valid {type_.__name__}: {raw!r}"
        ) from exc
    raise SchemaError(f"unfold setting {field!r} has unsupported type {type_!r}")


def _encode_meta(product: Product, dispatch: str) -> dict[str, float | int | str]:
    meta = _encode_common_meta(product)
    if dispatch == "calib":
        assert isinstance(product, CalibProduct)
        meta[META_CHANNEL_MAX] = float(product.channel_max)
        meta[META_PARAMS_REPORTED_JSON] = _json_dumps(
            [float(value) for value in product.params_reported]
        )
        meta[META_RESOL_PARAMS_JSON] = _json_dumps(
            [float(value) for value in product.resol_params]
        )
        meta[META_CHI2] = float(product.chi2)
        meta[META_DOF] = int(product.dof)
        meta[META_COVARIANCE_SCALE] = float(product.covariance_scale)
        meta[META_FIT_STATUS] = product.fit_status
        meta[META_SCALES_JSON] = _json_dumps(
            [
                [str(label), [float(value) for value in params]]
                for label, params in product.scales
            ]
        )
        meta[META_SCALE_BOUND_FLAGS_JSON] = _json_dumps(
            [
                [str(label), [int(flag) for flag in flags]]
                for label, flags in product.scale_bound_flags
            ]
        )
        meta[META_RESOL_CLAMP_COUNT] = int(product.resol_clamp_count)
        meta[META_RESOL_CLAMP_ENERGY_LOW_KEV] = float(product.resol_clamp_energy_low_kev)
        meta[META_RESOL_CLAMP_ENERGY_HIGH_KEV] = float(product.resol_clamp_energy_high_kev)
    elif dispatch == "sim":
        assert isinstance(product, SimProduct)
        meta[META_MODE] = int(product.mode)
        meta[META_MODE_NAME] = product.mode_name
        meta[META_GEOMETRY_NAME] = product.geometry_name
        meta[META_GEOMETRY_PARAM_MM] = float(product.geometry_param_mm)
        meta[META_ANGULAR_DISTRIBUTION] = product.angular_distribution
        meta[META_SEED] = int(product.seed)
        meta[META_N_EVENTS] = int(product.n_events)
        meta[META_WORKERS] = int(product.workers)
    elif dispatch in ("unfold", "unfold_calib_only"):
        assert isinstance(product, UnfoldProduct)
        meta[META_MODE] = product.mode
        if dispatch == "unfold":
            provided = dict(product.settings)
            if len(provided) != len(product.settings):
                raise SchemaError("unfold settings contain duplicate field names")
            expected = set(UNFOLD_SETTING_TYPES)
            if set(provided) != expected:
                raise SchemaError(
                    "unfold settings mismatch: "
                    f"missing={sorted(expected - set(provided))} "
                    f"extra={sorted(set(provided) - expected)}"
                )
            for field, type_ in UNFOLD_SETTING_TYPES.items():
                meta[field] = _coerce_setting(field, type_, provided[field])
    elif dispatch == "spectrum":
        assert isinstance(product, SpectrumProduct)
        meta[META_DAQ_TIME_S] = float(product.daq_time_s)
        meta[META_SOURCE_FILE] = product.source_file
    elif dispatch == "mc_spectrum":
        assert isinstance(product, McSpectrumProduct)
        meta[META_SOURCE_KEY] = product.source_key
        meta[META_MODE_NAME] = product.mode_name
        meta[META_GEOMETRY_NAME] = product.geometry_name
        meta[META_GEOMETRY_PARAM_MM] = float(product.geometry_param_mm)
        meta[META_N_EVENTS] = int(product.n_events)
        meta[META_SEED] = int(product.seed)
        meta[META_WORKERS] = int(product.workers)
    return meta


def _decode_float_list(meta: Mapping[str, Any], field: str, length: int) -> tuple[float, ...]:
    raw = _json_loads(meta[field], field)
    if not isinstance(raw, list) or len(raw) != length:
        raise SchemaError(
            f"meta field {field!r} must be a JSON list of {length} numbers"
        )
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SchemaError(f"meta field {field!r} contains a non-numeric entry")
        values.append(float(item))
    return tuple(values)


def _decode_scales(
    meta: Mapping[str, Any], field: str
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    """Decode the calib ``scales_json`` field into labelled Bezier parameters."""
    raw = _json_loads(meta[field], field)
    if not isinstance(raw, list):
        raise SchemaError(f"meta field {field!r} must be a JSON list")
    scales: list[tuple[str, tuple[float, float, float, float]]] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise SchemaError(
                f"meta field {field!r} entries must be [label, [s0, s1, s2, s3]]"
            )
        label, params = entry
        if not isinstance(label, str) or not label:
            raise SchemaError(f"meta field {field!r} has a non-string or empty label")
        if not isinstance(params, list) or len(params) != 4:
            raise SchemaError(
                f"meta field {field!r} label {label!r} must carry four s parameters"
            )
        values: list[float] = []
        for item in params:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise SchemaError(
                    f"meta field {field!r} label {label!r} has a non-numeric s parameter"
                )
            values.append(float(item))
        scales.append((label, (values[0], values[1], values[2], values[3])))
    return tuple(scales)


def _decode_scale_flags(
    meta: Mapping[str, Any], field: str
) -> tuple[tuple[str, tuple[bool, bool, bool, bool]], ...]:
    """Decode the calib ``scale_bound_flags_json`` field (bound-hit flags)."""
    raw = _json_loads(meta[field], field)
    if not isinstance(raw, list):
        raise SchemaError(f"meta field {field!r} must be a JSON list")
    flags: list[tuple[str, tuple[bool, bool, bool, bool]]] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise SchemaError(
                f"meta field {field!r} entries must be [label, [f0, f1, f2, f3]]"
            )
        label, values = entry
        if not isinstance(label, str) or not label:
            raise SchemaError(f"meta field {field!r} has a non-string or empty label")
        if not isinstance(values, list) or len(values) != 4:
            raise SchemaError(
                f"meta field {field!r} label {label!r} must carry four flags"
            )
        parsed = tuple(bool(value) for value in values)
        flags.append((label, (parsed[0], parsed[1], parsed[2], parsed[3])))
    return tuple(flags)


def _decode_provenance(meta: Mapping[str, Any]) -> Provenance:
    arguments_raw = _json_loads(meta[META_ARGUMENTS_JSON], META_ARGUMENTS_JSON)
    if not isinstance(arguments_raw, list):
        raise SchemaError(f"meta field {META_ARGUMENTS_JSON!r} must be a JSON list")
    arguments: list[tuple[str, str]] = []
    for entry in arguments_raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise SchemaError(
                f"meta field {META_ARGUMENTS_JSON!r} entries must be [name, value]"
            )
        arguments.append((str(entry[0]), str(entry[1])))

    dependencies_raw = _json_loads(meta[META_DEPENDENCY_VERSIONS], META_DEPENDENCY_VERSIONS)
    if not isinstance(dependencies_raw, list):
        raise SchemaError(
            f"meta field {META_DEPENDENCY_VERSIONS!r} must be a JSON list"
        )
    dependencies: list[tuple[str, str]] = []
    for entry in dependencies_raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise SchemaError(
                f"meta field {META_DEPENDENCY_VERSIONS!r} entries must be [name, version]"
            )
        dependencies.append((str(entry[0]), str(entry[1])))

    inputs_raw = _json_loads(meta[META_INPUTS_JSON], META_INPUTS_JSON)
    if not isinstance(inputs_raw, list):
        raise SchemaError(f"meta field {META_INPUTS_JSON!r} must be a JSON list")
    inputs: list[InputFingerprint] = []
    for entry in inputs_raw:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise SchemaError(
                f"meta field {META_INPUTS_JSON!r} entries must be "
                "{'path': ..., 'sha256': ...}"
            )
        digest = str(entry["sha256"])
        if _SHA256.fullmatch(digest) is None:
            raise SchemaError(
                f"meta field {META_INPUTS_JSON!r} has an invalid sha256 digest {digest!r}"
            )
        inputs.append(InputFingerprint(path=str(entry["path"]), sha256=digest))

    return Provenance(
        created_utc=str(meta[META_CREATED_UTC]),
        producer=str(meta[META_PRODUCER]),
        command=str(meta[META_COMMAND]),
        arguments=tuple(arguments),
        git_revision=str(meta[META_GIT_REVISION]),
        git_dirty=bool(meta[META_GIT_DIRTY]),
        python_version=str(meta[META_PYTHON_VERSION]),
        dependency_versions=tuple(dependencies),
        inputs=tuple(inputs),
    )


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------
def _check_hist1d(name: str, hist: Histogram1D, *, require_variance: bool) -> None:
    values = np.asarray(hist.values, dtype=np.float64)
    if values.shape != (hist.axis.n_bins,):
        raise SchemaError(
            f"{name}: values shape {values.shape} does not match axis "
            f"({hist.axis.n_bins} bins)"
        )
    if not np.isfinite(values).all():
        raise ValidationError(f"{name} contains non-finite values")
    if hist.variances is None:
        if require_variance:
            raise SchemaError(f"{name}: missing required fSumw2 variance buffer")
        return
    variances = np.asarray(hist.variances, dtype=np.float64)
    if variances.shape != values.shape:
        raise SchemaError(
            f"{name}: variance shape {variances.shape} does not match values {values.shape}"
        )
    if not np.isfinite(variances).all():
        raise ValidationError(f"{name} variances contain non-finite values")


def _check_hist2d(name: str, hist: Histogram2D, *, require_variance: bool) -> None:
    shape = (hist.x.n_bins, hist.y.n_bins)
    values = np.asarray(hist.values, dtype=np.float64)
    if values.shape != shape:
        raise SchemaError(f"{name}: values shape {values.shape} does not match axes {shape}")
    if not np.isfinite(values).all():
        raise ValidationError(f"{name} contains non-finite values")
    if hist.variances is None:
        if require_variance:
            raise SchemaError(f"{name}: missing required fSumw2 variance buffer")
        return
    variances = np.asarray(hist.variances, dtype=np.float64)
    if variances.shape != shape:
        raise SchemaError(
            f"{name}: variance shape {variances.shape} does not match axes {shape}"
        )
    if not np.isfinite(variances).all():
        raise ValidationError(f"{name} variances contain non-finite values")


def _check_meta_fields(meta: Mapping[str, Any], dispatch: str) -> None:
    expected = meta_field_types(dispatch)
    if set(meta) != set(expected):
        raise SchemaError(
            f"{dispatch}: meta field mismatch; missing={sorted(set(expected) - set(meta))} "
            f"extra={sorted(set(meta) - set(expected))}"
        )
    for field, type_ in expected.items():
        value = meta[field]
        if isinstance(value, np.ndarray):
            raise SchemaError(f"meta field {field!r} must have exactly one entry")
        if type_ is str:
            if not isinstance(value, str):
                raise SchemaError(f"meta field {field!r} must be a string")
        elif type_ is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaError(f"meta field {field!r} must be an int")
        elif type_ is float:
            if isinstance(value, bool) or not isinstance(value, float):
                raise SchemaError(f"meta field {field!r} must be a float")
        else:  # pragma: no cover - contract table only uses int/float/str
            raise SchemaError(f"meta field {field!r} has unsupported type {type_!r}")
    if meta[META_FORMAT_VERSION] != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported format_version {meta[META_FORMAT_VERSION]!r}; "
            f"this reader implements {SCHEMA_VERSION}"
        )
    if meta[META_PRODUCT_KIND] != product_kind_for(dispatch):
        raise SchemaError(
            f"meta product_kind {meta[META_PRODUCT_KIND]!r} does not match "
            f"dispatch {dispatch!r}"
        )
    if meta[META_GIT_DIRTY] not in (0, 1):
        raise SchemaError(f"meta field {META_GIT_DIRTY!r} must be 0 or 1")


def _check_product_bodies(product: Product) -> None:
    dispatch = dispatch_key_for(product)
    if int(product.format_version) != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported format_version {product.format_version!r}; "
            f"this writer implements {SCHEMA_VERSION}"
        )
    if dispatch == "calib":
        assert isinstance(product, CalibProduct)
        _check_hist2d(
            OBJ_DEPOSITION_TO_CHANNEL,
            product.deposition_to_channel,
            require_variance=OBJ_DEPOSITION_TO_CHANNEL in _REQUIRED_VARIANCE["calib"],
        )
        _check_hist2d(
            OBJ_PARAM_COV,
            product.param_cov,
            require_variance=OBJ_PARAM_COV in _REQUIRED_VARIANCE["calib"],
        )
    elif dispatch == "sim":
        assert isinstance(product, SimProduct)
        _check_hist2d(
            OBJ_PRIMARY_TO_DEPOSITION,
            product.primary_to_deposition,
            require_variance=OBJ_PRIMARY_TO_DEPOSITION
            in _REQUIRED_VARIANCE["sim"],
        )
        _check_hist1d(
            OBJ_PRIMARY_COLUMN_TOTALS,
            product.primary_column_totals,
            require_variance=False,
        )
    elif dispatch == "compose":
        assert isinstance(product, ComposeProduct)
        for name, hist in (
            (OBJ_RESPONSE_MATRIX, product.response_matrix),
            (OBJ_DEPOSITION_TO_CHANNEL, product.deposition_to_channel),
            (OBJ_PRIMARY_TO_DEPOSITION, product.primary_to_deposition),
        ):
            _check_hist2d(name, hist, require_variance=False)
        for name, hist in (
            (OBJ_PRIMARY_COLUMN_TOTALS, product.primary_column_totals),
            (OBJ_PRIMARY_EFFICIENCY, product.primary_efficiency),
        ):
            _check_hist1d(name, hist, require_variance=False)
    elif dispatch == "unfold":
        assert isinstance(product, UnfoldProduct)
        if (
            product.sigma_statistical is None
            or product.sigma_systematic is None
            or product.sigma_total is None
            or product.refolded is None
        ):
            raise SchemaError("unfold product is missing one or more required objects")
        if len(dict(product.settings)) != len(product.settings):
            raise SchemaError("unfold settings contain duplicate field names")
        if set(dict(product.settings)) != set(UNFOLD_SETTING_TYPES):
            raise SchemaError(
                "unfold settings mismatch: expected "
                f"{sorted(UNFOLD_SETTING_TYPES)}"
            )
        _check_hist1d(OBJ_SPECTRUM_UNFOLDED, product.spectrum, require_variance=False)
        for name, band in (
            (OBJ_SIGMA_STATISTICAL, product.sigma_statistical),
            (OBJ_SIGMA_SYSTEMATIC, product.sigma_systematic),
            (OBJ_SIGMA_TOTAL, product.sigma_total),
        ):
            _check_hist1d(name, band, require_variance=name in _REQUIRED_VARIANCE["unfold"])
        _check_hist1d(OBJ_SPECTRUM_REFOLDED, product.refolded, require_variance=False)
    elif dispatch == "unfold_calib_only":
        assert isinstance(product, UnfoldProduct)
        if (
            product.sigma_statistical is not None
            or product.sigma_systematic is not None
            or product.sigma_total is not None
            or product.refolded is not None
        ):
            raise SchemaError("calib_only product must not carry bands or refolded")
        if product.settings:
            raise SchemaError("calib_only product must not carry unfold settings")
        _check_hist1d(OBJ_SPECTRUM_CALIBRATED, product.spectrum, require_variance=False)
    elif dispatch == "spectrum":
        assert isinstance(product, SpectrumProduct)
        _check_hist1d(
            OBJ_SPECTRUM,
            product.spectrum,
            require_variance=OBJ_SPECTRUM in _REQUIRED_VARIANCE["spectrum"],
        )
    elif dispatch == "mc_spectrum":
        assert isinstance(product, McSpectrumProduct)
        if not product.source_key:
            raise SchemaError("mc_spectrum: source_key must be non-empty")
        if product.geometry_name == "":
            raise SchemaError("mc_spectrum: geometry_name must be non-empty")
        if product.spectrum.axis.unit != UNIT_KEV:
            raise SchemaError(
                f"mc_spectrum: energy axis unit must be {UNIT_KEV!r}, "
                f"got {product.spectrum.axis.unit!r}"
            )
        canonical = source_mode_deposition_edges_kev()
        edges = np.asarray(product.spectrum.axis.edges, dtype=np.float64)
        if edges.shape != canonical.shape or not np.allclose(
            edges, canonical, rtol=0.0, atol=1e-9
        ):
            raise SchemaError(
                "mc_spectrum: energy axis must be the fixed source-mode axis "
                f"({canonical.size - 1} bins over "
                f"{canonical[0]}..{canonical[-1]} keV), got {edges.size - 1} bins"
            )
        _check_hist1d(
            OBJ_MC_SPECTRUM,
            product.spectrum,
            require_variance=OBJ_MC_SPECTRUM in _REQUIRED_VARIANCE["mc_spectrum"],
        )


# --------------------------------------------------------------------------
# Strict-mode product certificates
# --------------------------------------------------------------------------
def _require_same_edges(left: Axis, right: Axis, formula_id: str) -> None:
    try:
        check_same_edges(left, right)
    except SchemaError as exc:
        raise CertificateError(
            formula_id, f"axis mismatch: {left.name!r} vs {right.name!r} ({exc})"
        ) from exc


def _certify_calib(product: CalibProduct) -> None:
    params = np.asarray(product.params_reported, dtype=np.float64)
    if params.shape != (4,) or not np.isfinite(params).all():
        raise CertificateError("F-MODEL-2", "reported parameters must be finite (4 values)")
    resol = np.asarray(product.resol_params, dtype=np.float64)
    if resol.shape != (3,) or not np.isfinite(resol).all():
        raise CertificateError("F-MODEL-4", "resolution parameters must be finite (3 values)")
    if not np.isfinite(product.channel_max) or product.channel_max <= 0.0:
        raise CertificateError("F-MODEL-1", "channel_max must be positive and finite")
    if not np.isfinite(product.chi2) or product.chi2 < 0.0:
        raise CertificateError("F-CAL-1", "chi2 must be finite and non-negative")
    if product.dof < 0:
        raise CertificateError("F-CAL-1", "dof must be non-negative")
    if not np.isfinite(product.covariance_scale) or product.covariance_scale <= 0.0:
        raise CertificateError("F-COV-2", "covariance_scale must be positive and finite")
    if product.resol_clamp_count < 0:
        raise CertificateError("F-MODEL-5", "resol_clamp_count must be non-negative")
    for label, params in product.scales:
        if not label:
            raise CertificateError("F-CAL-2", "scale labels must be non-empty")
        if len(params) != 4 or not np.isfinite(np.asarray(params, dtype=np.float64)).all():
            raise CertificateError(
                "F-CAL-2", f"scale parameters for {label!r} must be four finite numbers"
            )

    matrix = sparse.csr_matrix(np.asarray(product.deposition_to_channel.values, dtype=np.float64))
    response = ResponseMatrix(
        matrix=matrix,
        column_sums=np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel(),
        deposition_edges_kev=product.deposition_to_channel.y.edges,
    )
    verify_response_columns(response, strict=True)

    covariance = np.asarray(product.param_cov.values, dtype=np.float64)
    if covariance.shape != (len(PARAM_NAMES_REPORTED), len(PARAM_NAMES_REPORTED)):
        raise CertificateError(
            "F-COV-2",
            f"param_cov must be {len(PARAM_NAMES_REPORTED)}x{len(PARAM_NAMES_REPORTED)}, "
            f"got {covariance.shape}",
        )
    estimate = CovarianceEstimate(
        matrix=covariance, scale=1.0, chi2=0.0, dof=1, estimator="product"
    )
    verify_covariance_psd(estimate, strict=True)


def _certify_sim(product: SimProduct) -> None:
    counts = np.asarray(product.primary_to_deposition.values, dtype=np.float64)
    totals = np.asarray(product.primary_column_totals.values, dtype=np.float64)
    if np.any(counts < 0.0):
        raise CertificateError("F-SIM-2", "simulation counts must be non-negative")
    if not np.isfinite(totals).all() or np.any(totals < 0.0):
        raise CertificateError("F-SIM-1", "per-column totals must be finite and non-negative")
    if np.any(counts > totals[None, :] + _CERT_RTOL):
        raise CertificateError("F-SIM-1", "a deposition bin exceeds its column total")
    column_sums = counts.sum(axis=0)
    if np.any(column_sums > totals + _CERT_RTOL):
        raise CertificateError("F-SIM-1", "a column sum exceeds its column total")
    empty = totals == 0.0
    if np.any(counts[:, empty] != 0.0):
        raise CertificateError("F-SIM-1", "a zero-total column carries non-zero counts")
    if not np.isclose(totals.sum(), float(product.n_events), rtol=0.0, atol=_CERT_RTOL):
        raise CertificateError(
            "F-SIM-1",
            f"sum(N_j)={totals.sum():.12g} != n_events={product.n_events}",
        )
    probabilities = np.divide(
        counts, totals[None, :], out=np.zeros_like(counts), where=totals[None, :] > 0.0
    )
    expected = totals[None, :] * probabilities * (1.0 - probabilities)
    variances = product.primary_to_deposition.variances
    if variances is None:
        raise CertificateError("F-SIM-2", "primary_to_deposition has no fSumw2 buffer")
    if np.any(np.abs(variances - expected) > _CERT_RTOL * np.maximum(1.0, expected)):
        raise CertificateError("F-SIM-2", "fSumw2 does not match N_j p (1 - p)")
    efficiency = np.divide(
        column_sums, totals, out=np.zeros_like(totals), where=totals > 0.0
    )
    if np.any(efficiency < -_CERT_RTOL) or np.any(efficiency > 1.0 + _CERT_RTOL):
        raise CertificateError("F-SIM-3", "derived efficiency leaves [0, 1]")


def _certify_compose(product: ComposeProduct) -> None:
    """F-RESP-2/F-RESP-3: axis consistency, composition identity and eta.

    ``primary_efficiency`` is the detection efficiency ``colsum(G)/N_j``
    (F-SIM-3). ``R`` column sums equal the **reachable** detected mass/N
    (F-RESP-2): they coincide with ``eta`` only when no deposition column has
    an exactly-zero C column, which F-RESP-1 explicitly allows (D-99).
    """
    g_axis = product.primary_to_deposition.y
    _require_same_edges(product.deposition_to_channel.x, product.response_matrix.x, "F-RESP-3")
    _require_same_edges(
        product.primary_to_deposition.x, product.deposition_to_channel.y, "F-RESP-3"
    )
    _require_same_edges(g_axis, product.primary_column_totals.axis, "F-RESP-3")
    _require_same_edges(g_axis, product.response_matrix.y, "F-RESP-3")
    _require_same_edges(g_axis, product.primary_efficiency.axis, "F-RESP-3")

    c_values = np.asarray(product.deposition_to_channel.values, dtype=np.float64)
    g_values = np.asarray(product.primary_to_deposition.values, dtype=np.float64)
    r_values = np.asarray(product.response_matrix.values, dtype=np.float64)
    totals = np.asarray(product.primary_column_totals.values, dtype=np.float64)
    efficiency = np.asarray(product.primary_efficiency.values, dtype=np.float64)

    response = ResponseMatrix(
        matrix=sparse.csr_matrix(c_values),
        column_sums=c_values.sum(axis=0),
        deposition_edges_kev=product.deposition_to_channel.y.edges,
    )
    composed = ComposedResponse(
        matrix=sparse.csr_matrix(r_values),
        column_sums=np.asarray(r_values.sum(axis=0), dtype=np.float64),
        efficiency=efficiency,
        primary_edges_kev=product.primary_to_deposition.y.edges,
    )
    verify_composed_columns(composed, response, g_values, totals, strict=True)

    detected = g_values.sum(axis=0)
    expected_efficiency = np.divide(
        detected, totals, out=np.zeros_like(totals), where=totals > 0.0
    )
    if np.any(np.abs(efficiency - expected_efficiency) > _CERT_RTOL):
        raise CertificateError("F-RESP-2", "eta != column_sum(G) / N_j")


def _check_band_storage(name: str, band: Histogram1D) -> None:
    if band.variances is None:
        raise CertificateError("F-IO-1", f"{name} has no fSumw2 buffer (D-15)")
    if not np.allclose(band.variances, band.values**2, rtol=_CERT_RTOL, atol=_CERT_RTOL):
        raise CertificateError("F-IO-1", f"{name}: fSumw2 != content**2 (D-15)")


def _certify_unfold(product: UnfoldProduct) -> None:
    assert product.sigma_statistical is not None
    assert product.sigma_systematic is not None
    assert product.sigma_total is not None
    for name, band in (
        (OBJ_SIGMA_STATISTICAL, product.sigma_statistical),
        (OBJ_SIGMA_SYSTEMATIC, product.sigma_systematic),
        (OBJ_SIGMA_TOTAL, product.sigma_total),
    ):
        _require_same_edges(band.axis, product.spectrum.axis, "F-UNC-3")
        _check_band_storage(name, band)
    bands = UncertaintyBands(
        sigma_stat=np.asarray(product.sigma_statistical.values, dtype=np.float64),
        sigma_syst=np.asarray(product.sigma_systematic.values, dtype=np.float64),
        sigma_total=np.asarray(product.sigma_total.values, dtype=np.float64),
        components=(),
    )
    verify_band_decomposition(bands, strict=True)
    if product.refolded is not None and np.any(
        np.asarray(product.refolded.values, dtype=np.float64) < -_CERT_RTOL
    ):
        raise CertificateError("F-RESP-2", "refolded spectrum must be non-negative")


def _certify_unfold_calib_only(product: UnfoldProduct) -> None:
    if product.mode != UNFOLD_MODE_CALIB_ONLY:
        raise CertificateError("F-IO-1", "calib_only certificate on a non-calib_only product")


def _certify_spectrum(product: SpectrumProduct) -> None:
    if not np.isfinite(product.daq_time_s) or product.daq_time_s <= 0.0:
        raise CertificateError("F-IO-1", "daq_time_s must be positive and finite")
    variances = product.spectrum.variances
    if variances is None:
        raise CertificateError("F-IO-1", "spectrum has no fSumw2 buffer")
    if np.any(np.asarray(variances, dtype=np.float64) < 0.0):
        raise CertificateError("F-IO-1", "spectrum fSumw2 must be non-negative")


def _certify_mc_spectrum(product: McSpectrumProduct) -> None:
    """F-SIM-2 source variant (D-127/D-128): ``fSumw2 = c (1 - c/P)``."""
    values = np.asarray(product.spectrum.values, dtype=np.float64)
    variances = product.spectrum.variances
    if variances is None:
        raise CertificateError("F-SIM-2", "mc_spectrum has no fSumw2 buffer")
    variances = np.asarray(variances, dtype=np.float64)
    if np.any(values < 0.0):
        raise CertificateError("F-SIM-2", "mc_spectrum counts must be non-negative")
    if np.any(variances < 0.0):
        raise CertificateError("F-SIM-2", "mc_spectrum fSumw2 must be non-negative")
    total = float(values.sum())
    expected = np.zeros_like(values) if total == 0.0 else values * (1.0 - values / total)
    if np.any(np.abs(variances - expected) > _CERT_RTOL * np.maximum(1.0, expected)):
        raise CertificateError(
            "F-SIM-2",
            "mc_spectrum fSumw2 does not match c (1 - c/P) with P = sum(counts)",
        )
    if product.n_events < 0:
        raise CertificateError("F-SIM-1", "mc_spectrum n_events must be non-negative")
    if product.workers < 1:
        raise CertificateError("F-SIM-1", "mc_spectrum workers must be >= 1")
    if not np.isfinite(product.geometry_param_mm):
        raise CertificateError("F-SIM-1", "mc_spectrum geometry_param_mm must be finite")


def _run_certificates(product: Product, dispatch: str) -> None:
    if dispatch == "calib":
        assert isinstance(product, CalibProduct)
        _certify_calib(product)
    elif dispatch == "sim":
        assert isinstance(product, SimProduct)
        _certify_sim(product)
    elif dispatch == "compose":
        assert isinstance(product, ComposeProduct)
        _certify_compose(product)
    elif dispatch == "unfold":
        assert isinstance(product, UnfoldProduct)
        _certify_unfold(product)
    elif dispatch == "unfold_calib_only":
        assert isinstance(product, UnfoldProduct)
        _certify_unfold_calib_only(product)
    elif dispatch == "spectrum":
        assert isinstance(product, SpectrumProduct)
        _certify_spectrum(product)
    elif dispatch == "mc_spectrum":
        assert isinstance(product, McSpectrumProduct)
        _certify_mc_spectrum(product)


def _check_product(product: Product, *, strict: bool) -> None:
    dispatch = dispatch_key_for(product)
    _check_product_bodies(product)
    if strict:
        _run_certificates(product, dispatch)


def _strict_file_checks(path: str | Path, dispatch: str) -> None:
    """Raw-file strict checks that need the on-disk object (fSumw2 content)."""
    if dispatch == "spectrum":
        with uproot.open(path) as file:
            hist = file[OBJ_SPECTRUM]
            if not _uproot.histogram_has_variance(hist):
                raise CertificateError("F-IO-1", "spectrum has no fSumw2 buffer")
            raw = np.asarray(hist.member("fSumw2"), dtype=np.float64)
            if np.any(raw < 0.0):
                raise CertificateError("F-IO-1", "raw spectrum fSumw2 has negative entries")
    elif dispatch == "mc_spectrum":
        with uproot.open(path) as file:
            hist = file[OBJ_MC_SPECTRUM]
            if not _uproot.histogram_has_variance(hist):
                raise CertificateError("F-SIM-2", "mc_spectrum has no fSumw2 buffer")
            raw = np.asarray(hist.member("fSumw2"), dtype=np.float64)
            if np.any(raw < 0.0):
                raise CertificateError("F-SIM-2", "raw mc_spectrum fSumw2 is negative")


# --------------------------------------------------------------------------
# Reader / writer
# --------------------------------------------------------------------------
def _check_object_set(names: set[str], dispatch: str) -> None:
    expected = set(OBJECT_NAMES[dispatch])
    if names != expected:
        raise SchemaError(
            f"{dispatch}: object set mismatch; missing={sorted(expected - names)} "
            f"extra={sorted(names - expected)}"
        )


def _check_object_types(file: Any, dispatch: str) -> None:
    for name in OBJECT_NAMES[dispatch]:
        obj = file[name]
        kind = _OBJECT_KIND[name]
        if kind == "th2":
            ok = isinstance(obj, TH2)
        elif kind == "th1":
            ok = isinstance(obj, TH1) and not isinstance(obj, TH2)
        else:
            ok = isinstance(obj, RNTuple)
        if not ok:
            raise SchemaError(
                f"{dispatch}: object {name!r} has type {type(obj).__name__}, "
                f"expected {kind}"
            )


def _check_calib_labels(file: Any) -> None:
    """Always-on structural check of the ``param_cov`` bin labels (D-148).

    The labels are part of the frozen product contract, so they are verified in
    every read, not only under ``--strict`` (D-148).
    """
    hist = file[OBJ_PARAM_COV]
    expected = tuple(PARAM_NAMES_REPORTED)
    x_labels = _uproot.axis_labels(hist, 0)
    y_labels = _uproot.axis_labels(hist, 1)
    if x_labels != expected or y_labels != expected:
        raise SchemaError(
            f"param_cov bin labels must be {expected}; got x={x_labels} y={y_labels}"
        )


def _build_product(file: Any, dispatch: str, meta: Mapping[str, Any]) -> Product:
    format_version = int(meta[META_FORMAT_VERSION])
    provenance = _decode_provenance(meta)
    if dispatch == "calib":
        return CalibProduct(
            format_version=format_version,
            deposition_to_channel=_uproot.read_hist2d(file, OBJ_DEPOSITION_TO_CHANNEL),
            param_cov=_uproot.read_hist2d(file, OBJ_PARAM_COV),
            params_reported=_decode_float_list(meta, META_PARAMS_REPORTED_JSON, 4),
            resol_params=_decode_float_list(meta, META_RESOL_PARAMS_JSON, 3),
            channel_max=float(meta[META_CHANNEL_MAX]),
            provenance=provenance,
            chi2=float(meta[META_CHI2]),
            dof=int(meta[META_DOF]),
            covariance_scale=float(meta[META_COVARIANCE_SCALE]),
            fit_status=str(meta[META_FIT_STATUS]),
            scales=_decode_scales(meta, META_SCALES_JSON),
            scale_bound_flags=_decode_scale_flags(meta, META_SCALE_BOUND_FLAGS_JSON),
            resol_clamp_count=int(meta[META_RESOL_CLAMP_COUNT]),
            resol_clamp_energy_low_kev=float(meta[META_RESOL_CLAMP_ENERGY_LOW_KEV]),
            resol_clamp_energy_high_kev=float(meta[META_RESOL_CLAMP_ENERGY_HIGH_KEV]),
        )
    if dispatch == "sim":
        return SimProduct(
            format_version=format_version,
            primary_to_deposition=_uproot.read_hist2d(file, OBJ_PRIMARY_TO_DEPOSITION),
            primary_column_totals=_uproot.read_hist1d(file, OBJ_PRIMARY_COLUMN_TOTALS),
            mode=int(meta[META_MODE]),
            mode_name=str(meta[META_MODE_NAME]),
            geometry_name=str(meta[META_GEOMETRY_NAME]),
            geometry_param_mm=float(meta[META_GEOMETRY_PARAM_MM]),
            angular_distribution=str(meta[META_ANGULAR_DISTRIBUTION]),
            seed=int(meta[META_SEED]),
            n_events=int(meta[META_N_EVENTS]),
            workers=int(meta[META_WORKERS]),
            provenance=provenance,
        )
    if dispatch == "compose":
        return ComposeProduct(
            format_version=format_version,
            response_matrix=_uproot.read_hist2d(file, OBJ_RESPONSE_MATRIX),
            deposition_to_channel=_uproot.read_hist2d(file, OBJ_DEPOSITION_TO_CHANNEL),
            primary_to_deposition=_uproot.read_hist2d(file, OBJ_PRIMARY_TO_DEPOSITION),
            primary_column_totals=_uproot.read_hist1d(file, OBJ_PRIMARY_COLUMN_TOTALS),
            primary_efficiency=_uproot.read_hist1d(file, OBJ_PRIMARY_EFFICIENCY),
            provenance=provenance,
        )
    if dispatch == "unfold":
        settings = tuple((field, str(meta[field])) for field in UNFOLD_SETTING_TYPES)
        return UnfoldProduct(
            format_version=format_version,
            mode=str(meta[META_MODE]),
            spectrum=_uproot.read_hist1d(file, OBJ_SPECTRUM_UNFOLDED),
            sigma_statistical=_uproot.read_hist1d(file, OBJ_SIGMA_STATISTICAL),
            sigma_systematic=_uproot.read_hist1d(file, OBJ_SIGMA_SYSTEMATIC),
            sigma_total=_uproot.read_hist1d(file, OBJ_SIGMA_TOTAL),
            refolded=_uproot.read_hist1d(file, OBJ_SPECTRUM_REFOLDED),
            settings=settings,
            provenance=provenance,
        )
    if dispatch == "unfold_calib_only":
        return UnfoldProduct(
            format_version=format_version,
            mode=str(meta[META_MODE]),
            spectrum=_uproot.read_hist1d(file, OBJ_SPECTRUM_CALIBRATED),
            sigma_statistical=None,
            sigma_systematic=None,
            sigma_total=None,
            refolded=None,
            settings=(),
            provenance=provenance,
        )
    if dispatch == "spectrum":
        return SpectrumProduct(
            format_version=format_version,
            spectrum=_uproot.read_hist1d(file, OBJ_SPECTRUM),
            daq_time_s=float(meta[META_DAQ_TIME_S]),
            source_file=str(meta[META_SOURCE_FILE]),
            provenance=provenance,
        )
    if dispatch == "mc_spectrum":
        return McSpectrumProduct(
            format_version=format_version,
            spectrum=_uproot.read_hist1d(file, OBJ_MC_SPECTRUM),
            source_key=str(meta[META_SOURCE_KEY]),
            mode_name=str(meta[META_MODE_NAME]),
            geometry_name=str(meta[META_GEOMETRY_NAME]),
            geometry_param_mm=float(meta[META_GEOMETRY_PARAM_MM]),
            n_events=int(meta[META_N_EVENTS]),
            seed=int(meta[META_SEED]),
            workers=int(meta[META_WORKERS]),
            provenance=provenance,
        )
    raise SchemaError(f"unsupported dispatch key {dispatch!r}")  # pragma: no cover


def _read(path: str | Path) -> Product:
    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"no such product file: {source}")
    try:
        with uproot.open(source) as file:
            raw_names = [str(key).rsplit(";", 1)[0] for key in file]
            names = set(raw_names)
            if len(raw_names) != len(names):
                raise SchemaError(
                    f"{source}: duplicate object versions on disk; a product must "
                    "contain exactly one object per name"
                )
            if META_NTUPLE_NAME not in names:
                raise SchemaError(f"{source}: product has no {META_NTUPLE_NAME!r} metadata")
            meta = _uproot.read_meta_tree(file)
            if META_PRODUCT_KIND not in meta or META_FORMAT_VERSION not in meta:
                raise SchemaError(
                    f"{source}: {META_NTUPLE_NAME!r} lacks product_kind/format_version"
                )
            dispatch = dispatch_key_for_meta(
                meta[META_PRODUCT_KIND], meta.get(META_MODE)
            )
            _check_object_set(names, dispatch)
            _check_object_types(file, dispatch)
            _check_meta_fields(meta, dispatch)
            if dispatch == "calib":
                _check_calib_labels(file)
            return _build_product(file, dispatch, meta)
    except Kc761Error:
        raise
    except Exception as exc:  # uproot raises several unrelated exception types
        raise SchemaError(f"{source}: not a readable product file: {exc}") from exc


def _write_objects(file: Any, product: Product, dispatch: str) -> None:
    if dispatch == "calib":
        assert isinstance(product, CalibProduct)
        _uproot.write_hist2d(file, OBJ_DEPOSITION_TO_CHANNEL, product.deposition_to_channel)
        _uproot.write_hist2d(
            file,
            OBJ_PARAM_COV,
            product.param_cov,
            x_labels=PARAM_NAMES_REPORTED,
            y_labels=PARAM_NAMES_REPORTED,
        )
    elif dispatch == "sim":
        assert isinstance(product, SimProduct)
        _uproot.write_hist2d(file, OBJ_PRIMARY_TO_DEPOSITION, product.primary_to_deposition)
        _uproot.write_hist1d(file, OBJ_PRIMARY_COLUMN_TOTALS, product.primary_column_totals)
    elif dispatch == "compose":
        assert isinstance(product, ComposeProduct)
        _uproot.write_hist2d(file, OBJ_RESPONSE_MATRIX, product.response_matrix)
        _uproot.write_hist2d(file, OBJ_DEPOSITION_TO_CHANNEL, product.deposition_to_channel)
        _uproot.write_hist2d(file, OBJ_PRIMARY_TO_DEPOSITION, product.primary_to_deposition)
        _uproot.write_hist1d(file, OBJ_PRIMARY_COLUMN_TOTALS, product.primary_column_totals)
        _uproot.write_hist1d(file, OBJ_PRIMARY_EFFICIENCY, product.primary_efficiency)
    elif dispatch == "unfold":
        assert isinstance(product, UnfoldProduct)
        _uproot.write_hist1d(file, OBJ_SPECTRUM_UNFOLDED, product.spectrum)
        assert product.sigma_statistical is not None
        assert product.sigma_systematic is not None
        assert product.sigma_total is not None
        assert product.refolded is not None
        _uproot.write_hist1d(file, OBJ_SIGMA_STATISTICAL, product.sigma_statistical)
        _uproot.write_hist1d(file, OBJ_SIGMA_SYSTEMATIC, product.sigma_systematic)
        _uproot.write_hist1d(file, OBJ_SIGMA_TOTAL, product.sigma_total)
        _uproot.write_hist1d(file, OBJ_SPECTRUM_REFOLDED, product.refolded)
    elif dispatch == "unfold_calib_only":
        assert isinstance(product, UnfoldProduct)
        _uproot.write_hist1d(file, OBJ_SPECTRUM_CALIBRATED, product.spectrum)
    elif dispatch == "spectrum":
        assert isinstance(product, SpectrumProduct)
        _uproot.write_hist1d(file, OBJ_SPECTRUM, product.spectrum)
    elif dispatch == "mc_spectrum":
        assert isinstance(product, McSpectrumProduct)
        _uproot.write_hist1d(file, OBJ_MC_SPECTRUM, product.spectrum)
    _uproot.write_meta_tree(file, _encode_meta(product, dispatch))


def read_product(path: str | Path, *, strict: bool = False) -> Product:
    """Load and validate a product, dispatching on its ``meta`` tree (F-IO-1)."""
    product = _read(path)
    _check_product(product, strict=strict)
    if strict:
        _strict_file_checks(path, dispatch_key_for(product))
    return product


def verify_product(path: str | Path, *, strict: bool = False) -> None:
    """Reopen a product and validate schema, versions, axes and metadata (F-IO-1)."""
    product = _read(path)
    _check_product(product, strict=strict)
    if strict:
        _strict_file_checks(path, dispatch_key_for(product))


def write_product(
    product: Product, path: str | Path, *, force: bool = False, strict: bool = False
) -> Path:
    """Persist one product atomically and return its final path (F-IO-1)."""
    dispatch = dispatch_key_for(product)
    _check_product(product, strict=strict)
    refuse_overwrite(path, force=force)

    def _writer(part: Path) -> None:
        with uproot.recreate(part) as file:
            _write_objects(file, product, dispatch)

    return atomic_write(
        path,
        _writer,
        verify=lambda part: verify_product(part, strict=strict),
    )
