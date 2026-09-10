"""Product contracts: containers, object names, meta fields and provenance.

The product kinds and their object names are frozen here and described in
``docs/formats.md``. Changes go through the contract-change process
(``AGENTS.md``).

Node classes hold numpy arrays and axes only; validation and IO live in
``kc761.schema.io`` (W2). Every name, unit and meta field literal used by the
IO layer is defined here or in :mod:`kc761.schema.axes` so there is a single
source for the product contract (AGENTS hard rule 11).

Provenance is complete by contract (D-18): git revision, dependency versions,
input hashes and the full CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761.errors import SchemaError
from kc761.schema.axes import Axis

SCHEMA_VERSION = 1

#: Metadata object name; it is an RNTuple (D-12 as revised 2026-09-10).
META_NTUPLE_NAME = "meta"

# --- product kinds (D-12 / W2 decision 1) ---------------------------------
PRODUCT_KIND_CALIB: Final = "calib"
PRODUCT_KIND_SIM: Final = "sim"
PRODUCT_KIND_COMPOSE: Final = "compose"
PRODUCT_KIND_UNFOLD: Final = "unfold"
PRODUCT_KIND_SPECTRUM: Final = "spectrum"

PRODUCT_KINDS: Final[tuple[str, ...]] = (
    PRODUCT_KIND_CALIB,
    PRODUCT_KIND_SIM,
    PRODUCT_KIND_COMPOSE,
    PRODUCT_KIND_UNFOLD,
    PRODUCT_KIND_SPECTRUM,
)

#: Full unfolding versus calibration-only relabeling; they share one product
#: kind and are separated by the ``mode`` meta field.
UNFOLD_MODE_FULL: Final = "unfold"
UNFOLD_MODE_CALIB_ONLY: Final = "calib_only"
UNFOLD_MODES: Final[tuple[str, ...]] = (UNFOLD_MODE_FULL, UNFOLD_MODE_CALIB_ONLY)

# --- object names (single source) -----------------------------------------
OBJ_DEPOSITION_TO_CHANNEL: Final = "deposition_to_channel"
OBJ_PARAM_COV: Final = "param_cov"
OBJ_PRIMARY_TO_DEPOSITION: Final = "primary_to_deposition"
OBJ_PRIMARY_COLUMN_TOTALS: Final = "primary_column_totals"
OBJ_RESPONSE_MATRIX: Final = "response_matrix"
OBJ_PRIMARY_EFFICIENCY: Final = "primary_efficiency"
OBJ_SPECTRUM_UNFOLDED: Final = "kc761_spectrum_unfolded"
OBJ_SIGMA_STATISTICAL: Final = "sigma_statistical"
OBJ_SIGMA_SYSTEMATIC: Final = "sigma_systematic"
OBJ_SIGMA_TOTAL: Final = "sigma_total"
OBJ_SPECTRUM_REFOLDED: Final = "kc761_spectrum_refolded"
OBJ_SPECTRUM_CALIBRATED: Final = "kc761_spectrum_calibrated"
OBJ_SPECTRUM: Final = "kc761_spectrum"

#: Object collection per dispatch key. The set on disk must match exactly
#: (no extra, no missing object; W2 decision 6).
OBJECT_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "calib": (OBJ_DEPOSITION_TO_CHANNEL, OBJ_PARAM_COV, META_NTUPLE_NAME),
    "sim": (OBJ_PRIMARY_TO_DEPOSITION, OBJ_PRIMARY_COLUMN_TOTALS, META_NTUPLE_NAME),
    "compose": (
        OBJ_RESPONSE_MATRIX,
        OBJ_DEPOSITION_TO_CHANNEL,
        OBJ_PRIMARY_TO_DEPOSITION,
        OBJ_PRIMARY_COLUMN_TOTALS,
        OBJ_PRIMARY_EFFICIENCY,
        META_NTUPLE_NAME,
    ),
    "unfold": (
        OBJ_SPECTRUM_UNFOLDED,
        OBJ_SIGMA_STATISTICAL,
        OBJ_SIGMA_SYSTEMATIC,
        OBJ_SIGMA_TOTAL,
        OBJ_SPECTRUM_REFOLDED,
        META_NTUPLE_NAME,
    ),
    "unfold_calib_only": (OBJ_SPECTRUM_CALIBRATED, META_NTUPLE_NAME),
    "spectrum": (OBJ_SPECTRUM, META_NTUPLE_NAME),
}

#: ``product_kind`` value written for each dispatch key.
PRODUCT_KIND_FOR_DISPATCH: Final[dict[str, str]] = {
    "calib": PRODUCT_KIND_CALIB,
    "sim": PRODUCT_KIND_SIM,
    "compose": PRODUCT_KIND_COMPOSE,
    "unfold": PRODUCT_KIND_UNFOLD,
    "unfold_calib_only": PRODUCT_KIND_UNFOLD,
    "spectrum": PRODUCT_KIND_SPECTRUM,
}

# --- meta field names (single source) -------------------------------------
META_FORMAT_VERSION: Final = "format_version"
META_PRODUCT_KIND: Final = "product_kind"
META_PRODUCER: Final = "producer"
META_CREATED_UTC: Final = "created_utc"
META_GIT_REVISION: Final = "git_revision"
META_GIT_DIRTY: Final = "git_dirty"
META_PYTHON_VERSION: Final = "python_version"
META_DEPENDENCY_VERSIONS: Final = "dependency_versions"
META_COMMAND: Final = "command"
META_ARGUMENTS_JSON: Final = "arguments_json"
META_INPUTS_JSON: Final = "inputs_json"

META_CHANNEL_MAX: Final = "channel_max"
META_PARAMS_REPORTED_JSON: Final = "params_reported_json"
META_RESOL_PARAMS_JSON: Final = "resol_params_json"
#: W3 calibration fit diagnostics (D-49/D-106). ``chi2``/``dof`` and
#: ``covariance_scale`` are shared with the unfold settings; the remaining
#: fields are calib-only.
META_FIT_STATUS: Final = "fit_status"
META_SCALES_JSON: Final = "scales_json"
META_SCALE_BOUND_FLAGS_JSON: Final = "scale_bound_flags_json"
META_RESOL_CLAMP_COUNT: Final = "resol_clamp_count"
META_RESOL_CLAMP_ENERGY_LOW_KEV: Final = "resol_clamp_energy_low_kev"
META_RESOL_CLAMP_ENERGY_HIGH_KEV: Final = "resol_clamp_energy_high_kev"

META_MODE: Final = "mode"
META_MODE_NAME: Final = "mode_name"
META_GEOMETRY_NAME: Final = "geometry_name"
META_GEOMETRY_PARAM_MM: Final = "geometry_param_mm"
META_ANGULAR_DISTRIBUTION: Final = "angular_distribution"
META_SEED: Final = "seed"
META_N_EVENTS: Final = "n_events"
META_WORKERS: Final = "workers"

META_ALPHA: Final = "alpha"
META_DIFFERENCE_ORDER: Final = "difference_order"
META_ENERGY_LOW_KEV: Final = "energy_low_kev"
META_ENERGY_HIGH_KEV: Final = "energy_high_kev"
META_CHANNEL_LOW: Final = "channel_low"
META_CHANNEL_HIGH: Final = "channel_high"
META_PAD_NSIGMA: Final = "pad_nsigma"
META_SYST_FRAC: Final = "syst_frac"
META_CHI2: Final = "chi2"
META_DOF: Final = "dof"
META_COVARIANCE_SCALE: Final = "covariance_scale"

META_DAQ_TIME_S: Final = "daq_time_s"
META_SOURCE_FILE: Final = "source_file"

#: Ordered dependency list whose versions are recorded in provenance (D-18).
#: W5 passes Geant4 through ``extra_dependencies``.
TRACKED_DEPENDENCIES: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "numba",
    "sympy",
    "uproot",
    "matplotlib",
)

#: Types of the unfold settings. The keys are the meta field names and their
#: order is the canonical ``settings`` order.
UNFOLD_SETTING_TYPES: Final[dict[str, type]] = {
    META_ALPHA: float,
    META_DIFFERENCE_ORDER: int,
    META_ENERGY_LOW_KEV: float,
    META_ENERGY_HIGH_KEV: float,
    META_CHANNEL_LOW: int,
    META_CHANNEL_HIGH: int,
    META_PAD_NSIGMA: float,
    META_SYST_FRAC: float,
    META_CHI2: float,
    META_DOF: int,
    META_COVARIANCE_SCALE: float,
}

_COMMON_META_TYPES: Final[dict[str, type]] = {
    META_FORMAT_VERSION: int,
    META_PRODUCT_KIND: str,
    META_PRODUCER: str,
    META_CREATED_UTC: str,
    META_GIT_REVISION: str,
    META_GIT_DIRTY: int,
    META_PYTHON_VERSION: str,
    META_DEPENDENCY_VERSIONS: str,
    META_COMMAND: str,
    META_ARGUMENTS_JSON: str,
    META_INPUTS_JSON: str,
}

_SIM_META_TYPES: Final[dict[str, type]] = {
    META_MODE: int,
    META_MODE_NAME: str,
    META_GEOMETRY_NAME: str,
    META_GEOMETRY_PARAM_MM: float,
    META_ANGULAR_DISTRIBUTION: str,
    META_SEED: int,
    META_N_EVENTS: int,
    META_WORKERS: int,
}

_CALIB_META_TYPES: Final[dict[str, type]] = {
    META_CHANNEL_MAX: float,
    META_PARAMS_REPORTED_JSON: str,
    META_RESOL_PARAMS_JSON: str,
    META_CHI2: float,
    META_DOF: int,
    META_COVARIANCE_SCALE: float,
    META_FIT_STATUS: str,
    META_SCALES_JSON: str,
    META_SCALE_BOUND_FLAGS_JSON: str,
    META_RESOL_CLAMP_COUNT: int,
    META_RESOL_CLAMP_ENERGY_LOW_KEV: float,
    META_RESOL_CLAMP_ENERGY_HIGH_KEV: float,
}

_SPECTRUM_META_TYPES: Final[dict[str, type]] = {
    META_DAQ_TIME_S: float,
    META_SOURCE_FILE: str,
}

_EXTRA_META_TYPES: Final[dict[str, dict[str, type]]] = {
    "calib": _CALIB_META_TYPES,
    "sim": _SIM_META_TYPES,
    "compose": {},
    "unfold": {META_MODE: str, **UNFOLD_SETTING_TYPES},
    "unfold_calib_only": {META_MODE: str},
    "spectrum": _SPECTRUM_META_TYPES,
}


def meta_field_types(dispatch_key: str) -> dict[str, type]:
    """Return the exact ``meta`` field -> type contract for a dispatch key."""
    if dispatch_key not in OBJECT_NAMES:
        raise SchemaError(
            f"unknown dispatch key {dispatch_key!r}; expected one of {tuple(OBJECT_NAMES)}"
        )
    types: dict[str, type] = dict(_COMMON_META_TYPES)
    types.update(_EXTRA_META_TYPES[dispatch_key])
    return types


def meta_fields(dispatch_key: str) -> tuple[str, ...]:
    """Return the exact ordered ``meta`` field list for a dispatch key."""
    return tuple(meta_field_types(dispatch_key))


def product_kind_for(dispatch_key: str) -> str:
    """Map a dispatch key to the ``product_kind`` written on disk."""
    if dispatch_key not in PRODUCT_KIND_FOR_DISPATCH:
        raise SchemaError(f"unknown dispatch key {dispatch_key!r}")
    return PRODUCT_KIND_FOR_DISPATCH[dispatch_key]


def dispatch_key_for_meta(product_kind: object, mode: object) -> str:
    """Resolve the dispatch key from the on-disk ``product_kind`` and ``mode``.

    The two unfold variants share ``product_kind = 'unfold'`` and are separated
    by ``mode``; every other kind is its own dispatch key.
    """
    if product_kind == PRODUCT_KIND_UNFOLD:
        if mode not in UNFOLD_MODES:
            raise SchemaError(
                f"unfold product has invalid mode {mode!r}; expected one of {UNFOLD_MODES}"
            )
        return "unfold_calib_only" if mode == UNFOLD_MODE_CALIB_ONLY else "unfold"
    if isinstance(product_kind, str) and product_kind in OBJECT_NAMES:
        return product_kind
    raise SchemaError(
        f"unknown product_kind {product_kind!r}; expected one of {PRODUCT_KINDS}"
    )


@dataclass(frozen=True)
class InputFingerprint:
    """One input file and its sha256 digest (D-18)."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Provenance:
    """Complete provenance record written into every ``meta`` tree (D-18)."""

    created_utc: str
    producer: str
    command: str
    arguments: tuple[tuple[str, str], ...]
    git_revision: str
    git_dirty: bool
    python_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    inputs: tuple[InputFingerprint, ...]


@dataclass(frozen=True)
class Histogram1D:
    """One-dimensional histogram: axis, values and optional variances."""

    axis: Axis
    values: NDArray[np.float64]
    variances: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class Histogram2D:
    """Two-dimensional histogram with x = output side, y = input side (D-20)."""

    x: Axis
    y: Axis
    values: NDArray[np.float64]
    variances: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class CalibProduct:
    """Calibration export: response matrix, covariance and reported parameters.

    The parameter vectors are stored as JSON lists in ``meta`` in the frozen
    ``PARAM_NAMES_REPORTED`` order; ``param_cov`` carries the same names as
    axis bin labels (D-13).
    """

    format_version: int
    deposition_to_channel: Histogram2D
    param_cov: Histogram2D
    params_reported: tuple[float, float, float, float]
    resol_params: tuple[float, float, float]
    channel_max: float
    provenance: Provenance
    #: W3 fit diagnostics (D-49/D-106). Defaulted so pre-W3 synthetic fixtures
    #: and the W2 round-trip tests keep constructing valid products.
    chi2: float = 0.0
    dof: int = 0
    covariance_scale: float = 1.0
    fit_status: str = "unknown"
    scales: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    scale_bound_flags: tuple[tuple[str, tuple[bool, bool, bool, bool]], ...] = ()
    resol_clamp_count: int = 0
    resol_clamp_energy_low_kev: float = 0.0
    resol_clamp_energy_high_kev: float = 0.0


@dataclass(frozen=True)
class SimProduct:
    """Matrix-mode simulation export: counts plus per-column totals (D-35/D-36)."""

    format_version: int
    primary_to_deposition: Histogram2D
    primary_column_totals: Histogram1D
    mode: int
    mode_name: str
    geometry_name: str
    geometry_param_mm: float
    angular_distribution: str
    seed: int
    n_events: int
    workers: int
    provenance: Provenance


@dataclass(frozen=True)
class ComposeProduct:
    """Inspection artifact for ``R = C . p_tilde . diag(eta)`` (D-43)."""

    format_version: int
    response_matrix: Histogram2D
    deposition_to_channel: Histogram2D
    primary_to_deposition: Histogram2D
    primary_column_totals: Histogram1D
    primary_efficiency: Histogram1D
    provenance: Provenance


@dataclass(frozen=True)
class UnfoldProduct:
    """Unfold (or calibration-only) export with strictly split bands (D-50)."""

    format_version: int
    mode: str  # "unfold" or "calib_only"
    spectrum: Histogram1D
    sigma_statistical: Histogram1D | None
    sigma_systematic: Histogram1D | None
    sigma_total: Histogram1D | None
    refolded: Histogram1D | None
    settings: tuple[tuple[str, str], ...]
    provenance: Provenance


@dataclass(frozen=True)
class SpectrumProduct:
    """Measured or background-subtracted channel spectrum (D-72)."""

    format_version: int
    spectrum: Histogram1D
    daq_time_s: float
    source_file: str
    provenance: Provenance


Product = (
    CalibProduct
    | SimProduct
    | ComposeProduct
    | UnfoldProduct
    | SpectrumProduct
)


def dispatch_key_for(product: Product) -> str:
    """Return the dispatch key of an in-memory product."""
    if isinstance(product, CalibProduct):
        return "calib"
    if isinstance(product, SimProduct):
        return "sim"
    if isinstance(product, ComposeProduct):
        return "compose"
    if isinstance(product, SpectrumProduct):
        return "spectrum"
    if isinstance(product, UnfoldProduct):
        if product.mode not in UNFOLD_MODES:
            raise SchemaError(
                f"unfold product has invalid mode {product.mode!r}; "
                f"expected one of {UNFOLD_MODES}"
            )
        return "unfold_calib_only" if product.mode == UNFOLD_MODE_CALIB_ONLY else "unfold"
    raise SchemaError(f"unsupported product type {type(product).__name__}")
